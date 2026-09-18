"""The constitution bridge: a frozen base LLM coupled to a frozen model that
carries Claude's constitution, writing only into the base model's value neurons.

Shape of the channel (mirrors psilm/mlx/model.py's physics coupling, with the
FNO replaced by a language model and the injection restricted to a subspace):

    base stream at l_fwd  --fwd-->  K soft tokens
                                      |
                      [prefix] + soft + [suffix]  ->  constitution model
                                      |            (full causal pass, final norm)
                            M readout hidden states
                                      |
                                    --rev-->  M tokens
                                      |
    base stream at l_rev  <--gated cross-attention, MASKED to the value dims

Why the mask. arXiv:2602.00986 finds a sparse (~1%) set of residual-stream
dimensions whose activation predicts state value: zeroing them crashes accuracy
while zeroing random dimensions of the same count does nothing. Those dimensions
are the coupling point, so the injection is multiplied by a frozen 0/1 mask over
them and the write-location ablations (value neurons / random dims of the same
count / the whole stream) are separate training runs that differ only in the
mask. The cap and the receiver's local scale are therefore measured over the
WRITTEN dimensions only: "5% of the local stream" has to mean 5% per written
coordinate, otherwise a 9-dim write into a 896-dim stream is capped ~10x tighter
than a full-stream write (RMS over all dims of a vector supported on 9 of 896 is
sqrt(9/896) = 0.1 of its RMS over those 9) and the ablation would compare
channel strengths instead of write locations.

Training is self-distillation: the teacher is the same frozen backbone reading a
constitution excerpt as its system prompt, the student is the backbone plus these
bridges reading the plain system prompt. See eval/mlx_constitution_train.py.

Nothing here loads weights at import time. ``python -m psilm.mlx.constitution``
runs the CPU-only self-test on a tiny synthetic backbone.
"""

import json
import math
import random
import time
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .bridges import _rms
from .model import cross_entropy_masked
from .staged import MlxStream, padded_causal_mask

#: Fixed framing around the soft tokens inside the constitution model. The soft
#: tokens stand in for the situation; the suffix is the question the constitution
#: model is being asked, and its last M positions are the readout. Fixed text
#: (not learned) so a checkpoint's features mean the same thing in every run.
PREFIX_TEXT = "Situation:"
SUFFIX_TEXT = "\nAccording to Claude's constitution, in this situation Claude should"

#: What a checkpoint carries besides its tensors (see load_constitution_stack).
META_KEYS = ("d_model", "d_const", "k_fwd", "m_tokens", "write_dims", "read_dims",
             "l_fwd", "l_rev", "const_model", "gate_bias", "inj_cap", "readout_norm",
             "emb_rms", "step")

#: (model, tokenizers, constitution model, bridges, checkpoint meta) - what the
#: trainer and the evaluator both need, so their --self-test can inject a tiny
#: synthetic one in place of real weights.
ConstStack = namedtuple("ConstStack", "model tok hf_tok const bridges meta")


# ----------------------------------------------------------------------------
# write / read dimension sets
# ----------------------------------------------------------------------------

def parse_dims(spec: Optional[str], d_model: int) -> Optional[List[int]]:
    """Resolve a --write-dims / --read-dims spelling to a sorted dim list.

    "all" (or None)            -> None, meaning every dimension
    "<path.json>"              -> that value-neuron file's "top1pct"
    "<path.json>:top5pct"      -> its "top5pct"
    "<path.json>:topN"         -> the first N entries of its "ranking"
    "random:<seed>:<n>"        -> n seeded random dims (the control arm)

    The file is agent A1's results/value_neurons/<tag>/layer<l>.json with keys
    "layer", "d_model", "ranking" (most important first), "top1pct", "top5pct".
    """
    if spec is None or spec in ("", "all"):
        return None
    if spec.startswith("random:"):
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"{spec!r}: expected random:<seed>:<n>")
        seed, n = int(parts[1]), int(parts[2])
        if not 0 < n <= d_model:
            raise ValueError(f"{spec!r}: n must be in 1..{d_model}")
        # random.Random.sample is stable across CPython versions for a given
        # seed, so the control arm is reproducible from the spelling alone.
        return sorted(random.Random(seed).sample(range(d_model), n))
    path, key = (spec.rsplit(":", 1) if ":" in spec else (spec, "top1pct"))
    rec = json.loads(Path(path).read_text())
    if int(rec.get("d_model", d_model)) != d_model:
        raise ValueError(f"{path}: d_model {rec['d_model']} != backbone's {d_model}")
    if key in ("top1pct", "top5pct"):
        dims = list(rec[key])
    elif key.startswith("top"):
        n = int(key[3:])
        dims = list(rec["ranking"])[:n]
        if len(dims) < n:
            raise ValueError(f"{path}: ranking has {len(dims)} dims, asked for top{n}")
    else:
        raise ValueError(f"{spec!r}: unknown key {key!r} (top1pct, top5pct, topN)")
    out = sorted({int(d) for d in dims})
    if not out or out[0] < 0 or out[-1] >= d_model:
        raise ValueError(f"{path}:{key}: dims out of range for d_model {d_model}")
    return out


def dims_label(dims: Optional[List[int]], d_model: int) -> str:
    if dims is None or len(dims) == d_model:
        return f"all {d_model}"
    return f"{len(dims)} of {d_model} ({100.0 * len(dims) / d_model:.2f}%)"


# ----------------------------------------------------------------------------
# the constitution model
# ----------------------------------------------------------------------------

def _embedding_row_rms(emb, vocab: int, sample: int = 4096) -> float:
    """Mean over rows of the per-row RMS of the embedding matrix: the scale the
    soft tokens have to land at to look like tokens to the constitution model.

    A quantized embedding keeps its rows packed, so those are sampled through the
    module (which dequantizes); a mean of row norms needs no more than a sample.
    """
    if hasattr(emb, "scales"):
        n = min(vocab, sample)
        idx = mx.linspace(0, vocab - 1, n).astype(mx.int32)
        rows = emb(idx).astype(mx.float32)
    else:
        rows = emb.weight.astype(mx.float32)
    r = mx.sqrt((rows * rows).mean(axis=-1) + 1e-12)
    return float(r.mean().item())


class ConstitutionModelMLX:
    """A frozen mlx-lm model whose weights "contain" the constitution, driven on
    soft tokens instead of ids.

    Not an nn.Module: it holds no trainable state, and keeping it out of the
    bridges' parameter tree is what keeps save_weights/load_weights strict.
    Gradients still flow through it to the soft tokens, which is the point --
    the forward bridge learns to write something this model can read.
    """

    def __init__(self, path: str, m_tokens: int = 8, prefix_text: str = PREFIX_TEXT,
                 suffix_text: str = SUFFIX_TEXT):
        import mlx_lm
        model, tok = mlx_lm.load(str(path))
        self.path = str(path)
        self._init(model, tok, m_tokens, prefix_text, suffix_text)

    @classmethod
    def from_loaded(cls, model, tok, m_tokens: int = 8, path: str = "<in-memory>",
                    prefix_text: str = PREFIX_TEXT, suffix_text: str = SUFFIX_TEXT):
        """Wrap an already-loaded (model, tokenizer): the self-tests' tiny model."""
        self = cls.__new__(cls)
        self.path = path
        self._init(model, tok, m_tokens, prefix_text, suffix_text)
        return self

    def _init(self, model, tok, m_tokens, prefix_text, suffix_text):
        model.freeze()
        self.model = model
        self.inner = model.model
        self.tok = tok
        self.d_const = int(model.args.hidden_size)
        self.vocab = int(model.args.vocab_size)
        self.emb_rms = _embedding_row_rms(self.inner.embed_tokens, self.vocab)
        self.prefix_text, self.suffix_text = prefix_text, suffix_text
        # the constitution model's OWN tokenizer, no chat template: these ids are
        # a fixed frame, not a conversation
        self.prefix_ids = mx.array(_encode(tok, prefix_text), dtype=mx.int32)
        self.suffix_ids = mx.array(_encode(tok, suffix_text), dtype=mx.int32)
        self.m_tokens = int(m_tokens)
        # the readout is the tail of the suffix, so it cannot be longer than it
        self.n_readout = min(int(m_tokens), int(self.suffix_ids.size))
        assert self.n_readout >= 1, "suffix must tokenize to at least one token"
        # mlx-lm decoder stacks expose model.model(inputs, cache, input_embeddings)
        # and return the FINAL-NORMED hidden state. Prefer that (it is the
        # architecture's own forward, so any embedding scale -- Gemma's sqrt(d) --
        # and per-layer mask pattern is applied as the author intended); fall back
        # to the explicit staged loop for a stack that cannot take embeddings.
        import inspect
        try:
            self._native = "input_embeddings" in inspect.signature(self.inner.__call__).parameters
        except (TypeError, ValueError):                     # pragma: no cover
            self._native = False

    def features(self, soft: mx.array) -> mx.array:
        """soft (B, K, d_const) -> (B, M, d_const) final-norm hidden states at the
        last M positions of [prefix] + soft + [suffix], M = n_readout.

        One causal pass, no cache, all positions valid (the frame is fixed and K
        is constant, so every row of the batch has the same length -- no padding
        mask is needed). The soft tokens are cast to the embedding dtype so the
        model runs in its own precision; gradients pass back through the cast.
        """
        B = soft.shape[0]
        pre = self.inner.embed_tokens(self.prefix_ids)[None]      # (1, P, d)
        suf = self.inner.embed_tokens(self.suffix_ids)[None]      # (1, S, d)
        embs = mx.concatenate([mx.broadcast_to(pre, (B,) + pre.shape[1:]),
                               soft.astype(pre.dtype),
                               mx.broadcast_to(suf, (B,) + suf.shape[1:])], axis=1)
        if self._native:
            h = self.inner(None, cache=None, input_embeddings=embs)
        else:                                                     # pragma: no cover
            h = embs
            attn = mx.ones(embs.shape[:2], dtype=mx.int32)
            mask = padded_causal_mask(attn, embs.dtype)
            for layer in self.inner.layers:
                h = layer(h, mask=mask, cache=None)
            h = self.inner.norm(h)
        return h[:, -self.n_readout:, :]


def _encode(tok, text: str) -> List[int]:
    """Plain token ids, no special tokens, whatever kind of tokenizer this is."""
    try:
        ids = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(text)
    return [int(i) for i in ids]


# ----------------------------------------------------------------------------
# bridges
# ----------------------------------------------------------------------------

class ConstitutionForwardBridge(nn.Module):
    """base residual stream at l_fwd -> K soft tokens for the constitution model.

    Reads either the whole stream or only the value-neuron dimensions (the strict
    variant: if the sparse subsystem is where value lives, reading it should be
    enough). Per-dimension calibration then a parameter-free RMS, exactly as
    ForwardBridgeMLX: the standardization is what stops a backbone's constant
    massive-activation dimensions from swallowing the across-prompt variance, and
    the RMS is what keeps the pooling softmax off saturation at any width.

    K learned queries pool over the prompt positions (one softmax each) and a
    shared MLP maps each pooled vector into the constitution model's width. The
    output is renormalized to emb_rms * gain so the soft tokens arrive at the
    scale of real embedding rows -- a model that has never seen an activation ten
    times the size of a token embedding reads one as noise.
    """

    def __init__(self, d_model: int, d_const: int, k_fwd: int = 8, d_hidden: int = 512,
                 read_idx: Optional[List[int]] = None, readout_norm: str = "dim",
                 emb_rms: float = 1.0):
        super().__init__()
        self.readout_norm = readout_norm
        self.emb_rms = float(emb_rms)
        self.k_fwd = int(k_fwd)
        if read_idx is None:
            self.read_idx = None
            d_read = d_model
        else:
            idx = sorted({int(i) for i in read_idx})
            assert idx and 0 <= idx[0] and idx[-1] < d_model, "read dims out of range"
            self.read_idx = mx.array(idx, dtype=mx.int32)
            self.freeze(keys=["read_idx"], recurse=False)
            d_read = len(idx)
        self.d_read = d_read
        self.dim_mu = mx.zeros((d_read,))
        self.dim_sigma = mx.ones((d_read,))
        self.freeze(keys=["dim_mu", "dim_sigma"], recurse=False)
        self.queries = mx.random.normal((k_fwd, d_read)) / math.sqrt(d_read)
        self.key = nn.Linear(d_read, d_read)
        self.mlp1 = nn.Linear(d_read, d_hidden)
        self.mlp2 = nn.Linear(d_hidden, d_const)
        self.gain = mx.array(1.0)

    # -- pieces -------------------------------------------------------------
    def _read(self, hidden):
        h = hidden.astype(mx.float32)
        if self.read_idx is not None:
            h = mx.take(h, self.read_idx, axis=-1)
        return h

    def _normalize(self, hidden):
        h = self._read(hidden)
        if self.readout_norm == "dim":
            h = (h - self.dim_mu) / self.dim_sigma
        return _rms(h)

    def calibrate_readout(self, hidden, valid):
        """hidden (B, L, d_model) at l_fwd, valid (B, L) bool: per-dim mean/std
        over the valid positions, of the READ dimensions."""
        h = self._read(hidden)
        w = valid.astype(mx.float32)[..., None]
        n = w.sum()
        mu = (h * w).sum(axis=(0, 1)) / n
        var = (((h - mu) ** 2) * w).sum(axis=(0, 1)) / n
        self.dim_mu = mu
        self.dim_sigma = mx.sqrt(var + 1e-6)
        self.freeze(keys=["dim_mu", "dim_sigma"], recurse=False)
        mx.eval(self.dim_mu, self.dim_sigma)

    def __call__(self, hidden, prompt_mask):
        h = self._normalize(hidden)                              # (B, L, d_read)
        scores = (self.key(h) @ self.queries.T) / math.sqrt(h.shape[-1])   # (B, L, K)
        scores = mx.where(prompt_mask[..., None], scores,
                          mx.array(-1e9, dtype=scores.dtype))
        w = mx.softmax(scores, axis=1)                           # over positions
        weights = w.transpose(0, 2, 1)                           # (B, K, L)
        pooled = weights @ h                                     # (B, K, d_read)
        soft = self.mlp2(nn.gelu(self.mlp1(pooled)))             # (B, K, d_const)
        return _rms(soft) * (self.emb_rms * self.gain), weights


class ConstitutionReverseBridge(nn.Module):
    """M constitution-model hidden states -> M tokens for the base model.

    Deliberately thin: one linear map plus a per-position bias. The features are
    already the output of a 24-layer read of the constitution; what is left to
    learn is where in the base model's basis they belong, and a per-position bias
    lets the M tokens differentiate even before the linear map has moved.
    """

    def __init__(self, d_const: int, d_model: int, m_tokens: int):
        super().__init__()
        self.m_tokens = int(m_tokens)
        self.proj = nn.Linear(d_const, d_model)
        self.pos_bias = mx.zeros((m_tokens, d_model))

    def __call__(self, feats):
        assert feats.shape[1] == self.m_tokens, (feats.shape, self.m_tokens)
        return self.proj(_rms(feats.astype(mx.float32))) + self.pos_bias[None]


class MaskedGatedCrossAttentionMLX(nn.Module):
    """GatedCrossAttentionMLX, writing only into a fixed set of dimensions.

    Identical computation (same attention, same gate, same leaky-gate and
    return_ratio semantics) with three differences, all forced by the mask:

      * the injection is multiplied by a frozen 0/1 buffer over the write dims;
      * the cap ratio is RMS over the WRITTEN dims, so --inj-cap 0.05 means the
        injection is 5% of the local stream *per written coordinate* rather than
        5% of a stream it mostly does not touch;
      * the receiver's local scale is likewise RMS over the written dims, so the
        channel is scale-free in the subspace it writes to.

    With every dimension written this is bit-for-bit GatedCrossAttentionMLX
    (sum * (1/n) is exactly mean in MLX for the widths in play; asserted by the
    self-test), which is what makes the full-stream ablation arm a fair control.
    """

    def __init__(self, d_model: int, write_idx: Optional[List[int]] = None,
                 d_attn: int = 256, g_hidden: int = 256, gate_bias: float = 0.0,
                 inj_cap=None):
        super().__init__()
        self.inj_cap = inj_cap
        self.gate_bias = float(gate_bias)   # as constructed, not as trained
        self.gate_floor = None              # leaky gate, set at inference only
        self.contentless = None             # int seed, set at inference only: see __call__
        idx = (list(range(d_model)) if write_idx is None
               else sorted({int(i) for i in write_idx}))
        assert idx and 0 <= idx[0] and idx[-1] < d_model, "write dims out of range"
        self.write_idx = mx.array(idx, dtype=mx.int32)
        mask = mx.zeros((d_model,), dtype=mx.float32)
        mask[self.write_idx] = 1.0
        self.write_mask = mask.astype(mx.bool_)
        self.freeze(keys=["write_idx", "write_mask"], recurse=False)
        self.n_write = len(idx)
        self._inv_n = 1.0 / self.n_write
        self.to_q = nn.Linear(d_model, d_attn)
        self.to_k = nn.Linear(d_model, d_attn)
        self.to_v = nn.Linear(d_model, d_attn)
        self.to_out = nn.Linear(d_attn, d_model)
        self.to_out.weight = 1e-3 * mx.random.normal(self.to_out.weight.shape)
        self.to_out.bias = mx.zeros_like(self.to_out.bias)
        self.g1 = nn.Linear(d_model, g_hidden)
        self.g2 = nn.Linear(g_hidden, 1)
        self.g2.weight = mx.zeros_like(self.g2.weight)
        self.g2.bias = mx.full(self.g2.bias.shape, gate_bias)

    def __call__(self, hidden, tokens, return_ratio=False):
        dtype = hidden.dtype
        h_raw = hidden.astype(mx.float32)
        h = _rms(h_raw)                      # the gate and the query read the
        q = self.to_q(h)                     # WHOLE stream; only the write does not
        k, v = self.to_k(tokens), self.to_v(tokens)
        attn = mx.softmax(q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1]), axis=-1)
        inj = self.to_out(attn @ v) * self.write_mask
        if self.contentless is not None:
            # The CONTENTLESS control. Malla et al. (2609.06951) find that
            # steering's off-target movement lands on a fixed set of default
            # sinks -- chiefly refusal -- largely regardless of what is steered,
            # and that a magnitude-matched contentless direction moves the same
            # behaviours in the same order. Below 10B the pull is strongest, so a
            # 9B backbone with refusal as the readout is exactly the case that
            # needs this control before a refusal shift is attributed to content.
            #
            # So: one fixed random direction over the written coordinates,
            # rescaled at EVERY position to the real injection's own
            # per-coordinate RMS. Same channel, same written coordinates, same
            # gate (sigma reads h, not the payload), same receiver scale, same
            # magnitude -- no content. Note this is matched to what the real
            # injection actually does here, not merely to inj_cap, so it stays
            # matched at any position where the cap does not bind.
            r_real = mx.sqrt((inj * inj).sum(axis=-1, keepdims=True) * self._inv_n + 1e-12)
            d = mx.random.normal(shape=(1, 1, inj.shape[-1]),
                                 key=mx.random.key(int(self.contentless))) * self.write_mask
            r_d = mx.sqrt((d * d).sum(axis=-1, keepdims=True) * self._inv_n + 1e-12)
            inj = d * (r_real / r_d)
        if self.inj_cap is not None:
            r = mx.sqrt((inj * inj).sum(axis=-1, keepdims=True) * self._inv_n + 1e-12)
            inj = inj * mx.minimum(mx.array(1.0), self.inj_cap / r)
        sigma = mx.sigmoid(self.g2(nn.relu(self.g1(h))))
        scale = mx.sqrt((h_raw * h_raw * self.write_mask).sum(axis=-1, keepdims=True)
                        * self._inv_n + 1e-6)
        sigma_eff = sigma if self.gate_floor is None else \
            self.gate_floor + (1.0 - self.gate_floor) * sigma
        delta = sigma_eff * inj * scale
        out = (h_raw + delta).astype(dtype)
        if return_ratio:
            ratio = mx.sqrt((delta * delta).sum(axis=-1) * self._inv_n + 1e-12) / scale[..., 0]
            return out, sigma, ratio
        return out, sigma


class ConstitutionBridgesMLX(nn.Module):
    """fwd + rev + masked injection: everything trainable in a constitution run."""

    def __init__(self, d_model: int, d_const: int, write_idx: Optional[List[int]] = None,
                 read_idx: Optional[List[int]] = None, k_fwd: int = 8, m_tokens: int = 8,
                 gate_bias: float = 0.0, inj_cap=0.2, readout_norm: str = "dim",
                 emb_rms: float = 1.0, d_hidden: int = 512):
        super().__init__()
        self.d_model, self.d_const = int(d_model), int(d_const)
        self.k_fwd, self.m_tokens = int(k_fwd), int(m_tokens)
        self.fwd = ConstitutionForwardBridge(d_model, d_const, k_fwd=k_fwd,
                                            d_hidden=d_hidden, read_idx=read_idx,
                                            readout_norm=readout_norm, emb_rms=emb_rms)
        self.rev = ConstitutionReverseBridge(d_const, d_model, m_tokens)
        self.inject = MaskedGatedCrossAttentionMLX(d_model, write_idx=write_idx,
                                                  gate_bias=gate_bias, inj_cap=inj_cap)

    @property
    def write_dims(self) -> List[int]:
        return [int(v) for v in self.inject.write_idx.tolist()]

    @property
    def read_dims(self) -> Optional[List[int]]:
        ri = self.fwd.read_idx
        return None if ri is None else [int(v) for v in ri.tolist()]


def load_constitution_bridge_weights(bridges: ConstitutionBridgesMLX, path,
                                     strict: bool = True):
    """Strict load: every tensor the module defines must be in the file and
    nothing else, buffers included, and the write/read dim buffers must carry the
    same dimensions the module was built with.

    The dim check is the one that matters in practice: a checkpoint trained on
    the value neurons and one trained on random dims of the same count have
    identical shapes, so without it an ablation could be evaluated against the
    wrong mask and never complain.
    """
    weights = dict(mx.load(str(path)))
    params = dict(tree_flatten(bridges.parameters()))
    missing = sorted(k for k in params if k not in weights)
    unexpected = sorted(k for k in weights if k not in params)
    wrong = [(k, tuple(weights[k].shape), tuple(params[k].shape))
             for k in weights if k in params
             and tuple(weights[k].shape) != tuple(params[k].shape)]
    if wrong:
        lines = "; ".join(f"{k}: file {a} vs module {b}" for k, a, b in wrong[:4])
        raise ValueError(f"{path}: tensor shapes do not match the module: {lines}")
    if strict and (missing or unexpected):
        raise ValueError(f"{path}: missing {missing}, unexpected {unexpected}")
    for key in ("inject.write_idx", "fwd.read_idx"):
        if key in weights and key in params:
            if not bool(mx.all(weights[key] == params[key]).item()):
                raise ValueError(
                    f"{path}: {key} in the checkpoint is a different dimension set "
                    f"than the module was built with -- rebuild from the .meta")
    bridges.load_weights(list(weights.items()), strict=False)
    return missing


# ----------------------------------------------------------------------------
# the coupled model
# ----------------------------------------------------------------------------

@dataclass
class Generation:
    gen_ids: List[int]
    text: str
    stopped_eos: bool
    sigma_prompt: List[float] = field(default_factory=list)
    sigma_gen: List[float] = field(default_factory=list)
    ratio_gen: List[float] = field(default_factory=list)
    sec: float = 0.0


def _eos_id_set(tok) -> set:
    ids = set()
    for attr in ("eos_token_ids", "eos_token_id"):
        v = getattr(tok, attr, None)
        if isinstance(v, int):
            ids.add(int(v))
        elif v:
            try:
                ids.update(int(x) for x in v)
            except TypeError:                      # pragma: no cover
                pass
    return ids


class PsiConstitutionMLX:
    """Frozen backbone + frozen constitution model + trainable bridges.

    Same layer convention as PsiLMMLX (l_fwd = round(n*10/24), l_rev =
    round(n*15/24) by default, "layer l" = the residual stream entering block l).
    """

    def __init__(self, model, tok, const_model: ConstitutionModelMLX,
                 bridges: ConstitutionBridgesMLX, l_fwd=None, l_rev=None,
                 lam_gate: float = 1.0):
        self.model = model
        self.tok = tok
        self.const = const_model
        self.phi = bridges
        n = len(model.model.layers)
        self.n_layers = n
        self.l_fwd = l_fwd if l_fwd is not None else round(n * 10 / 24)
        self.l_rev = l_rev if l_rev is not None else round(n * 15 / 24)
        assert 0 < self.l_fwd <= self.l_rev <= n, (self.l_fwd, self.l_rev, n)
        assert bridges.m_tokens == const_model.n_readout, \
            f"bridges expect {bridges.m_tokens} readout tokens, model gives {const_model.n_readout}"
        assert bridges.d_const == const_model.d_const, (bridges.d_const, const_model.d_const)
        model.freeze()
        self.lam_gate = lam_gate
        self.eos_ids = _eos_id_set(tok)
        self._last_ratio = None

    # -- coupling -----------------------------------------------------------
    def _apply_inject(self, h, tokens, mode, floor=None):
        """Mirrors bench_common.StagedDecoder._inject: the floor is set for the
        call and always restored, and 'zeroed' measures the gate on a stream the
        injection never touches."""
        self.phi.inject.gate_floor = floor
        try:
            h_inj, sigma, ratio = self.phi.inject(h, tokens, return_ratio=True)
        finally:
            self.phi.inject.gate_floor = None
        return (h_inj if mode == "psilm" else h), sigma, ratio

    def _couple(self, stream: MlxStream, prompt_mask, mode: str = "psilm", floor=None):
        """Run the stream to the end, exchanging state with the bridges.

        mode "base" runs the backbone alone (the bridges are not touched);
        "zeroed" builds the tokens and measures the gate but does not write.
        Returns (sigma, weights) with sigma None in the base arm.
        """
        if mode == "base":
            stream.run(0, self.n_layers)
            self._last_ratio = None
            return None, None
        stream.run(0, self.l_fwd)
        soft, weights = self.phi.fwd(stream.hidden, prompt_mask)
        feats = self.const.features(soft)
        tokens = self.phi.rev(feats)
        stream.run(self.l_fwd, self.l_rev)
        stream.hidden, sigma, ratio = self._apply_inject(stream.hidden, tokens, mode, floor)
        stream.run(self.l_rev, self.n_layers)
        self._last_ratio = ratio
        return sigma, weights

    def logits(self, ids, attn=None, prompt_mask=None, mode="psilm", floor=None):
        """Full-sequence logits plus (sigma, ratio). ids (B, L) mx array."""
        s = MlxStream(self.model, ids, attn)
        if prompt_mask is None:
            prompt_mask = mx.ones(ids.shape, dtype=mx.bool_)
        sigma, _ = self._couple(s, prompt_mask, mode, floor)
        return s.finish(), sigma, self._last_ratio

    # -- training -----------------------------------------------------------
    def loss_fn(self, batch):
        """CE on the labelled positions, plus the gate penalty on no-harm batches.

        Positives: the teacher's continuation, plain shifted CE (there is no
        digit cargo here; every token of the constitution-conditioned answer
        counts the same). No-harm: the backbone's OWN continuation on off-task
        prompts through the full coupled pipeline plus lam_gate * mean gate over
        the real tokens, exactly as PsiLMMLX._noharm_loss -- the only ways down
        are to shut the gate or to make the injection harmless where the
        constitution is irrelevant.
        """
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        sigma, _ = self._couple(s, batch["prompt_mask"])
        logits = s.finish()
        ce = cross_entropy_masked(logits, batch["p_labels"], None, 1.0)
        g = sigma[..., 0]
        valid = batch["p_attn"].astype(mx.float32)
        gate_all = (g * valid).sum() / (valid.sum() + 1e-6)      # real tokens only
        resp = (batch["p_labels"] != -100).astype(mx.float32)
        n_resp = resp.sum() + 1e-6
        gate_ans = (g * resp).sum() / n_resp
        ratio_ans = (self._last_ratio * resp).sum() / n_resp
        pm = batch["prompt_mask"].astype(mx.float32)
        gate_prompt = (pm * g).sum() / (pm.sum() + 1e-6)
        loss = ce + (self.lam_gate * gate_all if batch.get("noharm") else 0.0)
        return loss, (ce, gate_all, gate_ans, ratio_ans, gate_prompt)

    # -- evaluation ---------------------------------------------------------
    def teacher_forced(self, prompt_ids, cont_ids, mode="psilm", floor=None):
        """(mean CE, top-1 agreement, gate stats) on cont_ids in ONE pass.

        cont_ids is the teacher's greedy continuation, so CE is the distillation
        loss on held-out prompts and agreement is how often the coupled student's
        argmax IS the teacher's token. The prompt mask covers the prompt only,
        which is what the forward bridge pooled over in training.
        """
        prompt_ids, cont_ids = list(prompt_ids), list(cont_ids)
        if not cont_ids:
            return None, None, {}
        ids = prompt_ids + cont_ids
        t = mx.array([ids], dtype=mx.int32)
        pmask = mx.array([[True] * len(prompt_ids) + [False] * len(cont_ids)])
        lg, sigma, ratio = self.logits(t, None, pmask, mode, floor)
        lo, hi = len(prompt_ids) - 1, len(ids) - 1
        lg = lg[:, lo:hi].astype(mx.float32)
        tgt = mx.array([cont_ids], dtype=mx.int32)
        ce = nn.losses.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                     reduction="mean")
        agree = (lg.argmax(-1) == tgt).astype(mx.float32).mean()
        mx.eval(ce, agree)
        stats = {}
        if sigma is not None:
            g = sigma[0, :, 0].astype(mx.float32)
            r = ratio[0].astype(mx.float32)
            mx.eval(g, r)
            p = len(prompt_ids)
            stats = {"prompt_mean": float(g[:p].mean().item()),
                     "cont_mean": float(g[p:].mean().item()) if len(ids) > p else None,
                     "all_mean": float(g.mean().item()),
                     "max": float(g.max().item()),
                     "ratio_cont": float(r[p:].mean().item()) if len(ids) > p else None}
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        return float(ce.item()), float(agree.item()), stats

    def generate(self, prompt_ids, max_new: int = 64, mode: str = "psilm",
                 floor=None) -> Generation:
        """Greedy decode, recomputing the whole sequence at every step.

        No KV cache: the bridges read the prompt and inject at every position, so
        a cached decode is possible (bench_common.StagedDecoder does exactly that)
        but needs the decoder's machinery. At 0.5B and 128 new tokens the quadratic
        recompute is seconds; at 9B it is not, which is what the bench is for.
        """
        t0 = time.perf_counter()
        prompt_ids = list(prompt_ids)
        ids = list(prompt_ids)
        gen: List[int] = []
        sig_p: List[float] = []
        sig_g: List[float] = []
        rat_g: List[float] = []
        stopped = False
        for step in range(max_new):
            t = mx.array([ids], dtype=mx.int32)
            pmask = mx.array([[True] * len(prompt_ids)
                              + [False] * (len(ids) - len(prompt_ids))])
            lg, sigma, ratio = self.logits(t, None, pmask, mode)
            nxt = int(lg[:, -1].argmax(-1).item())
            if sigma is not None:
                if step == 0:
                    g = sigma[0, :, 0].astype(mx.float32)
                    mx.eval(g)
                    sig_p = [float(v) for v in g.tolist()]
                else:
                    sig_g.append(float(sigma[0, -1, 0].item()))
                    rat_g.append(float(ratio[0, -1].item()))
            ids.append(nxt)
            gen.append(nxt)
            if nxt in self.eos_ids:
                stopped = True
                break
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        text_ids = gen[:-1] if stopped else gen
        return Generation(gen, self._decode(text_ids), stopped, sig_p, sig_g, rat_g,
                          time.perf_counter() - t0)

    def _decode(self, ids) -> str:
        try:
            return self.tok.decode(ids, skip_special_tokens=True)
        except TypeError:
            return self.tok.decode(ids)


# ----------------------------------------------------------------------------
# the bench-side coupler (eval/bench_common.py's StagedDecoder drives this)
# ----------------------------------------------------------------------------

class ConstitutionCoupler:
    """The constitution channel behind the same two calls StagedDecoder makes of
    the physics channel: tokens(h_prompt) once per question, inject(h, tokens)
    once per position. x0_span/sub_value exist only to match that signature --
    there is no pointer here (the whole prompt is the situation) and no scalar
    value to substitute (the content control for this channel is a different
    excerpt, not a different number).
    """

    def __init__(self, bridges: ConstitutionBridgesMLX, const_model: ConstitutionModelMLX):
        self.phi = bridges
        self.const = const_model
        # The harness hands inject() a per-question seed already combined with the
        # question id, so generation and the teacher-forced KL pass -- which each
        # call tokens() once -- draw the SAME direction. An earlier version keyed
        # the seed on a call counter here, which gave the KL pass the next
        # question's direction and shifted every seed across a resume boundary.

    #: the harness checks for this before running a contentless arm, so an arm
    #: name this channel cannot honour fails loudly instead of quietly becoming
    #: the psilm arm again.
    supports_contentless = True

    def tokens(self, h_prompt, x0_span=None, sub_value=None):
        if sub_value is not None:
            raise ValueError("the constitution channel has no scalar to substitute")
        L = h_prompt.shape[1]
        pmask = mx.ones((1, L), dtype=mx.bool_)
        soft, weights = self.phi.fwd(h_prompt, pmask)
        feats = self.const.features(soft)
        tokens = self.phi.rev(feats)
        mx.eval(tokens, soft, weights)
        srms = mx.sqrt((soft * soft).mean(axis=-1) + 1e-12)
        trms = mx.sqrt((tokens * tokens).mean(axis=-1) + 1e-12)
        diag = {"soft_rms": round(float(srms.mean().item()), 4),
                "token_rms": round(float(trms.mean().item()), 4),
                # where each forward query looked, as a fraction into the prompt:
                # a channel that only reads the last token is a different channel
                "attn_pos": [round(float(v) / max(1, L - 1), 3)
                             for v in weights[0].argmax(-1).tolist()]}
        return tokens, diag

    def inject(self, h, tokens, mode, floor=None, contentless=None):
        self.phi.inject.gate_floor = floor          # None for the trained arms
        if contentless is not None:
            self.phi.inject.contentless = int(contentless)   # already per-question
        try:
            h_inj, sigma = self.phi.inject(h, tokens)     # sigma is pre-floor
        finally:
            self.phi.inject.gate_floor = None
            self.phi.inject.contentless = None
        if mode == "psilm":
            return h_inj, sigma
        return h, sigma                             # zeroed: gate measured, no write


def load_constitution_stack(ckpt_path, const_model_path: Optional[str] = None):
    """(bridges, const_model, meta) rebuilt from bridges.npz + bridges.npz.meta.

    Everything about the module's shape lives in the meta, including the resolved
    write/read dimension lists, so an evaluation never has to re-resolve a
    --write-dims spelling (a random control would come back different if the
    seed were lost) and cannot silently disagree with the checkpoint.
    """
    ckpt = Path(ckpt_path)
    if ckpt.suffix == ".safetensors":
        # The Hugging Face layout written by eval/export_constitution_bridge.py:
        # bridges.safetensors beside a config.json whose "meta" is the npz meta.
        meta = json.loads((ckpt.parent / "config.json").read_text())["meta"]
    else:
        meta = json.loads(Path(str(ckpt) + ".meta").read_text())
    path = const_model_path or meta["const_model"]
    const = ConstitutionModelMLX(path, m_tokens=int(meta["m_tokens"]))
    if const.n_readout != int(meta["m_tokens"]):      # a different suffix tokenization
        raise ValueError(f"{path}: gives {const.n_readout} readout tokens, "
                         f"the checkpoint has {meta['m_tokens']}")
    if const.d_const != int(meta["d_const"]):
        raise ValueError(f"{path}: hidden {const.d_const} != checkpoint's {meta['d_const']}")
    bridges = ConstitutionBridgesMLX(
        d_model=int(meta["d_model"]), d_const=int(meta["d_const"]),
        write_idx=meta["write_dims"], read_idx=meta["read_dims"],
        k_fwd=int(meta["k_fwd"]), m_tokens=int(meta["m_tokens"]),
        gate_bias=float(meta["gate_bias"]), inj_cap=meta["inj_cap"],
        readout_norm=meta["readout_norm"], emb_rms=float(meta["emb_rms"]),
        d_hidden=int(meta.get("d_hidden", 512)))
    load_constitution_bridge_weights(bridges, ckpt)
    bridges.freeze()
    mx.eval(bridges.parameters())
    return bridges, const, meta


def stack_meta(bridges: ConstitutionBridgesMLX, const: ConstitutionModelMLX,
               psi: PsiConstitutionMLX, step: int, extra: Dict[str, Any] = None):
    """The .meta payload: everything load_constitution_stack needs, plus args."""
    meta = {"d_model": bridges.d_model, "d_const": bridges.d_const,
            "k_fwd": bridges.k_fwd, "m_tokens": bridges.m_tokens,
            "write_dims": bridges.write_dims, "read_dims": bridges.read_dims,
            "l_fwd": psi.l_fwd, "l_rev": psi.l_rev, "const_model": const.path,
            # the INITIAL gate bias, not the trained one: it is a constructor
            # argument, and the trained value arrives with the weights
            "gate_bias": float(bridges.inject.gate_bias),
            "inj_cap": bridges.inject.inj_cap,
            "readout_norm": bridges.fwd.readout_norm,
            "emb_rms": bridges.fwd.emb_rms, "step": int(step),
            "d_hidden": bridges.fwd.mlp1.weight.shape[0]}
    if extra:
        meta.update(extra)
    return meta


# ----------------------------------------------------------------------------
# tiny synthetic stack + self-test (CPU only, no weights)
# ----------------------------------------------------------------------------

class FakeTok:
    """Just enough tokenizer for the tiny stack: ids that stay inside a 256-token
    vocabulary and a decode that keeps the test readable."""

    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = None

    def encode(self, s, add_special_tokens=False):
        return [2 + (ord(c) % 200) for c in s[:16]]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def build_tiny_stack(d_model: int = 64, n_layers: int = 4, vocab: int = 256,
                     heads: int = 4, k_fwd: int = 3, m_tokens: int = 4,
                     write_idx=None, read_idx=None, gate_bias: float = 0.0,
                     inj_cap=0.2, l_fwd: int = 1, l_rev: int = 3, seed: int = 0):
    """A random mlx-lm Qwen2 as both backbone and constitution model.

    CPU by default: the tiny tests must never queue work on the shared GPU.
    """
    mx.set_default_device(mx.cpu)
    mx.random.seed(seed)
    from mlx_lm.models.qwen2 import Model, ModelArgs
    def _args():
        return ModelArgs(model_type="qwen2", hidden_size=d_model, num_hidden_layers=n_layers,
                         intermediate_size=2 * d_model, num_attention_heads=heads,
                         rms_norm_eps=1e-6, vocab_size=vocab, num_key_value_heads=heads,
                         max_position_embeddings=512, rope_theta=10000.0,
                         tie_word_embeddings=True)
    model = Model(_args())
    model.freeze()
    tok = FakeTok()
    const = ConstitutionModelMLX.from_loaded(Model(_args()), tok, m_tokens=m_tokens,
                                             path="<tiny>")
    bridges = ConstitutionBridgesMLX(d_model, const.d_const, write_idx=write_idx,
                                     read_idx=read_idx, k_fwd=k_fwd,
                                     m_tokens=const.n_readout, gate_bias=gate_bias,
                                     inj_cap=inj_cap, readout_norm="dim",
                                     emb_rms=const.emb_rms, d_hidden=32)
    # a material injection: at init to_out is 1e-3 and g2 is exactly zero, which
    # is the right training start but makes every "does the channel do anything"
    # assertion vacuous
    bridges.inject.to_out.weight = 0.3 * mx.random.normal(bridges.inject.to_out.weight.shape)
    bridges.inject.g2.weight = 0.02 * mx.random.normal(bridges.inject.g2.weight.shape)
    mx.eval(model.parameters(), const.model.parameters(), bridges.parameters())
    psi = PsiConstitutionMLX(model, tok, const, bridges, l_fwd=l_fwd, l_rev=l_rev)
    meta = stack_meta(bridges, const, psi, 0)
    return ConstStack(model=model, tok=tok, hf_tok=tok, const=const,
                      bridges=bridges, meta=meta), psi


def synthetic_items(n: int, n_prompt: int = 11, n_cont: int = 5, vocab: int = 256,
                    seed: int = 3, key: str = "teacher_ids"):
    """Data in agent A3b's format, from random ids: enough to drive the loops."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = [rng.randrange(2, vocab) for _ in range(n_prompt)]
        t = [rng.randrange(2, vocab) for _ in range(n_cont)]
        b = [rng.randrange(2, vocab) for _ in range(n_cont)]
        rec = {"prompt_ids": p, key: t, "base_ids": b,
               "prompt_text": f"synthetic {i}", "source": f"synthetic:{i}",
               "differs": True}
        if key != "teacher_ids":
            rec["teacher_ids"] = t
        out.append(rec)
    return out


def self_test(verbose: bool = True):
    """Assertions (1)-(5) of the constitution bridge, on the tiny stack."""
    from .bridges import GatedCrossAttentionMLX
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    d, M = 64, 5

    # 1. all dims == the unmasked module, bit for bit (capped and uncapped, and
    #    with a leaky gate floor)
    for cap in (None, 0.05):
        mx.random.seed(11)
        g = GatedCrossAttentionMLX(d, gate_bias=-0.5, inj_cap=cap)
        m = MaskedGatedCrossAttentionMLX(d, write_idx=None, gate_bias=-0.5, inj_cap=cap)
        for name in ("to_q", "to_k", "to_v", "to_out", "g1", "g2"):
            setattr(m, name, getattr(g, name))
        g.to_out.weight = 0.3 * mx.random.normal(g.to_out.weight.shape)
        g.g2.weight = 0.02 * mx.random.normal(g.g2.weight.shape)
        h = 3.0 * mx.random.normal((2, 7, d))
        tk = mx.random.normal((2, M, d))
        for floor in (None, 0.1):
            g.gate_floor = m.gate_floor = floor
            og, sg, rg = g(h, tk, return_ratio=True)
            om, sm, rm = m(h, tk, return_ratio=True)
            mx.eval(og, om, sg, sm, rg, rm)
            assert bool(mx.all(og == om).item()), f"masked != unmasked (cap {cap}, floor {floor})"
            assert bool(mx.all(sg == sm).item()) and bool(mx.all(rg == rm).item())
        g.gate_floor = m.gate_floor = None
    if verbose:
        print("[self-test] masked(all dims) == GatedCrossAttentionMLX bit-for-bit  OK")

    # 2. a 6-dim write touches exactly 6 dims
    widx = [1, 3, 5, 7, 11, 13]
    mx.random.seed(5)
    m6 = MaskedGatedCrossAttentionMLX(d, write_idx=widx, gate_bias=0.0, inj_cap=0.2)
    m6.to_out.weight = 0.3 * mx.random.normal(m6.to_out.weight.shape)
    h = 3.0 * mx.random.normal((2, 7, d))
    out, _ = m6(h, mx.random.normal((2, M, d)))
    delta = out.astype(mx.float32) - h.astype(mx.float32)
    outside = [i for i in range(d) if i not in widx]
    mx.eval(delta)
    assert float(mx.abs(mx.take(delta, mx.array(outside), axis=-1)).max().item()) == 0.0, \
        "the masked injection leaked outside its write dims"
    assert float(mx.abs(mx.take(delta, mx.array(widx), axis=-1)).max().item()) > 0.0, \
        "the masked injection wrote nothing"
    if verbose:
        print(f"[self-test] {len(widx)}-dim write: delta exactly 0 on the other "
              f"{len(outside)} dims  OK")

    # 3./4./5. the coupled model
    stack, psi = build_tiny_stack(write_idx=[2, 4, 8, 16, 32, 48], read_idx=None)
    bridges = stack.bridges
    ids = mx.array([[int(v) for v in mx.random.randint(2, 200, (13,)).tolist()]])
    lb, _, _ = psi.logits(ids, mode="base")
    lz, sz, _ = psi.logits(ids, mode="zeroed")
    lp, _, _ = psi.logits(ids, mode="psilm")
    mx.eval(lb, lz, lp, sz)
    assert float(mx.abs(lb - lz).max().item()) == 0.0, "zeroed != base"
    assert float(mx.abs(lb - lp).max().item()) > 0.0, "psilm == base (dead channel)"
    if verbose:
        print(f"[self-test] zeroed == base bit-for-bit; psilm differs by "
              f"{float(mx.abs(lb - lp).max().item()):.3f}, gate {float(sz.mean().item()):.3f}  OK")

    # 4. every trainable parameter gets a gradient
    items = synthetic_items(2, n_prompt=9, n_cont=4)
    batch = pad_batch(items, 0, "teacher_ids")
    def wrapped(b, bt):
        psi.phi = b
        return psi.loss_fn(bt)
    (loss, aux), grads = nn.value_and_grad(bridges, wrapped)(bridges, batch)
    mx.eval(loss, grads)
    flat = dict(tree_flatten(grads))
    names = {k for k, _ in tree_flatten(bridges.trainable_parameters())}
    assert set(flat) == names, (sorted(set(flat) ^ names))
    dead = [k for k, v in flat.items() if float(mx.abs(v).max().item()) == 0.0]
    assert not dead, f"no gradient reaches {dead}"
    assert math.isfinite(float(loss.item()))
    if verbose:
        print(f"[self-test] loss {float(loss.item()):.3f}, all {len(flat)} trainable "
              f"tensors have a non-zero gradient  OK")

    # the pristine init is a special case worth stating: g2.weight is exactly 0,
    # so the gate's hidden layer has no gradient until g2 moves off zero
    fresh = ConstitutionBridgesMLX(64, stack.const.d_const, write_idx=[2, 4, 8],
                                  k_fwd=3, m_tokens=stack.const.n_readout, d_hidden=32)
    mx.eval(fresh.parameters())
    psi.phi = fresh
    _, g0 = nn.value_and_grad(fresh, wrapped)(fresh, batch)
    mx.eval(g0)
    zero0 = sorted(k for k, v in dict(tree_flatten(g0)).items()
                   if float(mx.abs(v).max().item()) == 0.0)
    assert zero0 == ["inject.g1.bias", "inject.g1.weight"], zero0
    if verbose:
        print("[self-test] at the pristine init only inject.g1 is gradient-free "
              "(g2.weight == 0 by construction)  OK")
    psi.phi = bridges

    # 5. save -> strict load -> identical logits
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bridges.npz"
        bridges.save_weights(str(p))
        again = ConstitutionBridgesMLX(64, stack.const.d_const,
                                       write_idx=bridges.write_dims,
                                       read_idx=bridges.read_dims, k_fwd=bridges.k_fwd,
                                       m_tokens=bridges.m_tokens, d_hidden=32,
                                       readout_norm=bridges.fwd.readout_norm,
                                       emb_rms=bridges.fwd.emb_rms,
                                       inj_cap=bridges.inject.inj_cap)
        load_constitution_bridge_weights(again, p)
        mx.eval(again.parameters())
        psi.phi = again
        lp2, _, _ = psi.logits(ids, mode="psilm")
        mx.eval(lp2)
        assert float(mx.abs(lp - lp2).max().item()) == 0.0, "reloaded bridges changed the logits"
        # and a checkpoint from a different write set must not load quietly
        other = ConstitutionBridgesMLX(64, stack.const.d_const, write_idx=[1, 3, 5, 7, 9, 11],
                                       read_idx=None, k_fwd=bridges.k_fwd,
                                       m_tokens=bridges.m_tokens, d_hidden=32,
                                       readout_norm=bridges.fwd.readout_norm,
                                       emb_rms=bridges.fwd.emb_rms,
                                       inj_cap=bridges.inject.inj_cap)
        try:
            load_constitution_bridge_weights(other, p)
        except ValueError as e:
            assert "different dimension set" in str(e), e
        else:
            raise AssertionError("a foreign write set loaded without complaint")
    psi.phi = bridges
    if verbose:
        print("[self-test] save -> load: identical logits; a foreign write set is "
              "refused  OK")

    # the bench-side coupler must be the same channel as the training path, and
    # must leave gate_floor where it found it
    cp = ConstitutionCoupler(bridges, stack.const)
    s = MlxStream(psi.model, ids)
    s.run(0, psi.l_fwd)
    tk, diag = cp.tokens(s.hidden)
    soft, _ = bridges.fwd(s.hidden, mx.ones(ids.shape, dtype=mx.bool_))
    tk_ref = bridges.rev(stack.const.features(soft))
    mx.eval(tk, tk_ref)
    assert bool(mx.all(tk == tk_ref).item()), "coupler tokens != the training path's"
    h_mid = s.run(psi.l_fwd, psi.l_rev)
    h_psi, sg1 = cp.inject(h_mid, tk, "psilm")
    h_zero, sg2 = cp.inject(h_mid, tk, "zeroed", 0.1)
    mx.eval(h_psi, h_zero, sg1, sg2)
    assert float(mx.abs(h_zero - h_mid).max().item()) == 0.0, "zeroed wrote to the stream"
    assert float(mx.abs(h_psi - h_mid).max().item()) > 0.0
    assert bool(mx.all(sg1 == sg2).item()), "the floor changed the REPORTED sigma"
    assert bridges.inject.gate_floor is None, "the gate floor leaked out of inject()"
    assert set(diag) == {"soft_rms", "token_rms", "attn_pos"} and len(diag["attn_pos"]) == bridges.k_fwd
    if verbose:
        print("[self-test] ConstitutionCoupler: same tokens as the training path, "
              "zeroed writes nothing, the gate floor is restored  OK")

    # the strict variant: read only the value-neuron dims, write only into them
    sub = [2, 4, 8, 16, 32, 48]
    s2, psi2 = build_tiny_stack(write_idx=sub, read_idx=sub, seed=4)
    assert s2.bridges.fwd.d_read == len(sub)
    lb2, _, _ = psi2.logits(ids, mode="base")
    lp2, _, _ = psi2.logits(ids, mode="psilm")
    def w2(b, bt):
        psi2.phi = b
        return psi2.loss_fn(bt)
    (l2, _), g2 = nn.value_and_grad(s2.bridges, w2)(s2.bridges, batch)
    mx.eval(lb2, lp2, l2, g2)
    assert float(mx.abs(lb2 - lp2).max().item()) > 0.0
    dead2 = [k for k, v in dict(tree_flatten(g2)).items()
             if float(mx.abs(v).max().item()) == 0.0]
    assert not dead2, f"no gradient reaches {dead2} in the read-restricted variant"
    if verbose:
        print(f"[self-test] read-restricted forward bridge (d_read={len(sub)}): couples "
              f"and trains  OK")
    return stack, psi


# ----------------------------------------------------------------------------
# batching (shared by the trainer and the evaluator)
# ----------------------------------------------------------------------------

def pad_batch(items, pad_id: int, tgt_key: Optional[str] = "teacher_ids",
              noharm: bool = False):
    """Right-padded prompt (+ continuation) batch in make_noharm_batch's layout:
    labels only on the continuation, prompt_mask over the prompt.

    tgt_key None gives a prompt-only batch (all labels masked) for calibration.
    """
    import numpy as np
    plens = [len(it["prompt_ids"]) for it in items]
    seqs = [list(it["prompt_ids"]) + (list(it[tgt_key]) if tgt_key else [])
            for it in items]
    L = max(len(q) for q in seqs)
    B = len(seqs)
    I = np.full((B, L), pad_id, dtype=np.int32)
    A = np.zeros((B, L), dtype=np.int32)
    LB = np.full((B, L), -100, dtype=np.int32)
    PM = np.zeros((B, L), dtype=bool)
    for i, (q, pl) in enumerate(zip(seqs, plens)):
        I[i, :len(q)] = q
        A[i, :len(q)] = 1
        LB[i, pl:len(q)] = q[pl:]
        PM[i, :pl] = True
    out = {"p_ids": mx.array(I), "p_attn": mx.array(A), "p_labels": mx.array(LB),
           "prompt_mask": mx.array(PM)}
    if noharm:
        out["noharm"] = True
    return out


if __name__ == "__main__":
    self_test()
    print("[self-test] psilm.mlx.constitution: all assertions passed")
