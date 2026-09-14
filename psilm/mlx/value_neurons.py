"""Value neurons: the sparse reward subsystem of a frozen backbone.

Reproduction machinery for "Sparse Reward Subsystem in Large Language Models"
(arXiv:2602.00986) on a PsiLM backbone, shared by eval/vn_collect.py (sample
trajectories, keep the residual stream), eval/vn_probe.py (train the TD value
probe, rank and prune input dimensions) and eval/vn_ablate.py (zero the top-1%
dimensions and watch accuracy fall).

Why this lives here: the constitution bridge injects at ``l_rev``, and the
dimensions it should write into are the ones the backbone itself already uses
to carry "is this continuation going to be right". Identification therefore has
to be per layer and has to end in an index list, which is what these three
scripts produce.

Layer convention, identical to psilm.mlx.staged.MlxStream: **"layer l" is the
residual stream ENTERING block l**, i.e. what ``MlxStream.run(0, l)`` returns
and what ``PsiBridgesMLX.inject`` sees at ``l_rev``. l = 0 is the embedding
output, l = n_layers the final pre-norm hidden state. Capturing and zeroing use
the same index, so an index list from a probe at layer l is directly the set of
dimensions a bridge injecting at ``l_rev = l`` writes into.

Nothing here loads weights at import time; mlx_lm is imported inside the
functions that need it, so a --self-test / --dry-run path stays weight-free.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GAMMA = 1.0 - 1e-5          # the paper's discount
D_PROBE_HIDDEN = 1024       # the paper's probe width


# ----------------------------------------------------------------------------
# decoding with residual-stream capture
# ----------------------------------------------------------------------------

class StreamDecoder:
    """KV-cached decode through ``model.model.layers`` that hands back the
    residual stream at chosen layer boundaries, at EVERY position.

    Same shape of computation as eval/bench_common.StagedDecoder (prefill the
    whole prompt with mask "causal", then one position per step with mask None),
    minus the bridges and plus the capture: the paper's probe is trained on the
    states of a trajectory, which are exactly the positions a decode visits.
    Capture is moved to numpy float16 after every step, so the MLX buffers for a
    320-token rollout never pile up.

    Backbone-agnostic: only ``model.model.layers`` / ``embed_tokens`` / ``norm``
    and an optional ``lm_head`` are touched, and the "causal"/None masks are
    what psilm.mlx.qwen35_loader._LayerShim expects, so a Qwen3.5 tower with
    hybrid linear/full attention works unchanged.
    """

    def __init__(self, model, hf_tok, eos_ids: Optional[Iterable[int]] = None,
                 flush_every: int = 32):
        self.model = model
        self.inner = model.model
        self.hf_tok = hf_tok
        self.n_layers = len(self.inner.layers)
        self.eos_ids = set(int(i) for i in eos_ids) if eos_ids is not None else set()
        # device->host copies are synchronous, so one per layer per step costs
        # more than the decode itself: 62 tok/s against 161 tok/s uncaptured at
        # 10 layers on Qwen2.5-0.5B. The captured states are already materialized
        # (mx.eval on the sampled token forces them), so holding a few dozen of
        # them and copying in one go is free memory-wise and ~2x faster.
        self.flush_every = max(1, int(flush_every))

    # -- pieces --------------------------------------------------------------
    def _blocks(self, h, mask, cache, pend: Dict[int, list]):
        for i, layer in enumerate(self.inner.layers):
            if i in pend:
                pend[i].append(h)
            h = layer(h, mask, cache[i])
        if self.n_layers in pend:
            pend[self.n_layers].append(h)
        return h

    def _logits_last(self, h):
        h = self.inner.norm(h[:, -1:, :])
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(h)[:, -1, :]
        else:
            logits = self.inner.embed_tokens.as_linear(h)[:, -1, :]
        post = getattr(self.model, "logit_postprocess", None)   # Gemma's soft-cap
        return post(logits) if post is not None else logits

    @staticmethod
    def _pick(logits, sampler):
        """Greedy, or an mlx_lm sampler (which takes log-probabilities)."""
        if sampler is None:
            return mx.argmax(logits, axis=-1)
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        return sampler(lp)

    @staticmethod
    def _flush(pend: Dict[int, list], out: Dict[int, list]):
        """Evaluate the pending captures in one graph and move them to numpy."""
        cast = {l: [a.astype(mx.float16) for a in lst] for l, lst in pend.items()}
        flat = [a for lst in cast.values() for a in lst]
        if flat:
            mx.eval(*flat)
        for l, lst in cast.items():
            for a in lst:
                out[l].append(np.array(a)[0])          # (1, T, d) -> (T, d)
            pend[l].clear()

    # -- public --------------------------------------------------------------
    def rollout(self, prompt_ids: Sequence[int], max_new: int = 320,
                capture: Sequence[int] = (), sampler=None) -> Dict:
        """One trajectory.

        Returns gen_ids / text / stopped_eos plus ``states``: {layer -> (T, d)
        float16} with T = len(prompt_ids) + n_gen_states. The rows are, in
        order, every prompt position and then the position of each generated
        token that was fed back in. The token that ENDS the rollout (EOS, or the
        max_new-th) is never fed back, so n_gen_states = len(gen_ids) - 1: state
        s_0 of the paper is row len(prompt_ids) - 1 (the prompt-final position,
        the one that predicts the first generated token) and the last row is the
        final state whose TD target is the reward.
        """
        from mlx_lm.models.cache import make_prompt_cache
        cap = sorted({int(l) for l in capture})
        for l in cap:
            assert 0 <= l <= self.n_layers, f"layer {l} outside [0, {self.n_layers}]"
        pend: Dict[int, list] = {l: [] for l in cap}
        out: Dict[int, list] = {l: [] for l in cap}
        t0 = time.perf_counter()
        cache = make_prompt_cache(self.model)
        h = self.inner.embed_tokens(mx.array([list(prompt_ids)]))
        h = self._blocks(h, "causal" if len(prompt_ids) > 1 else None, cache, pend)
        y = self._pick(self._logits_last(h), sampler)
        mx.eval(y)
        self._flush(pend, out)
        del h
        n_pend = 0
        gen: List[int] = []
        stopped = False
        n_gen_states = 0
        while True:
            tid = int(y.item())
            gen.append(tid)
            if tid in self.eos_ids:
                stopped = True
                break
            if len(gen) >= max_new:
                break
            h = self.inner.embed_tokens(y[None])                 # (1, 1, d)
            h = self._blocks(h, None, cache, pend)
            y = self._pick(self._logits_last(h), sampler)
            mx.eval(y)
            n_pend += 1
            if n_pend >= self.flush_every:
                self._flush(pend, out)
                n_pend = 0
            n_gen_states += 1
            del h
        self._flush(pend, out)
        del cache
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        text_ids = gen[:-1] if stopped else gen
        states = {l: (np.concatenate(out[l], axis=0) if out[l] else
                      np.zeros((0, 0), dtype=np.float16)) for l in cap}
        return {"gen_ids": gen, "text": self.hf_tok.decode(text_ids, skip_special_tokens=True),
                "stopped_eos": stopped, "states": states, "prompt_len": len(prompt_ids),
                "n_gen_states": n_gen_states, "sec": time.perf_counter() - t0}


# ----------------------------------------------------------------------------
# the causal check: zero a set of dimensions of the stream entering block l
# ----------------------------------------------------------------------------

class ZeroDimsShim:
    """Wrap decoder block ``index`` so that the chosen dimensions of its OUTPUT
    -- the residual stream entering block index+1 -- are multiplied by zero at
    every position.

    A plain object rather than an nn.Module, exactly like
    psilm.mlx.qwen35_loader._LayerShim: filing a foreign Module into the
    backbone's parameter tree would re-nest its weights one level deeper, and
    the ablation only ever reads them. ``calls`` / ``positions`` count what the
    shim actually saw, so a caller can assert that its generation path really
    goes through here (eval/vn_ablate.py does).
    """

    def __init__(self, layer, dims: Sequence[int], d_model: int, index: int):
        self.layer = layer
        self.index = index
        self.dims = sorted(int(i) for i in dims)
        m = np.ones((d_model,), dtype=np.float32)
        if self.dims:
            m[np.asarray(self.dims, dtype=np.int64)] = 0.0
        self.keep = mx.array(m)
        self.calls = 0
        self.positions = 0

    def __call__(self, x, mask=None, cache=None):
        out = self.layer(x, mask, cache)
        self.calls += 1
        self.positions += int(out.shape[-2])
        if not self.dims:
            return out
        return out * self.keep.astype(out.dtype)

    def __getattr__(self, name):          # the rest of the block (is_linear, args, ...)
        if name == "layer":
            raise AttributeError(name)
        return getattr(self.layer, name)


def install_zero_shim(model, layer: int, dims: Sequence[int], d_model: Optional[int] = None):
    """Zero ``dims`` of the stream entering block ``layer`` -> wrap block layer-1.

    layer == 0 (the embedding output) has no block to wrap and is refused; the
    candidate injection depths are all mid-stack.
    """
    assert layer >= 1, "layer 0 is the embedding output; nothing to wrap"
    layers = model.model.layers
    assert layer <= len(layers), (layer, len(layers))
    d = d_model if d_model is not None else int(model.args.hidden_size)
    shim = ZeroDimsShim(layers[layer - 1], dims, d, layer - 1)
    layers[layer - 1] = shim
    return shim


def remove_zero_shim(model, shim: ZeroDimsShim):
    model.model.layers[shim.index] = shim.layer


# ----------------------------------------------------------------------------
# the probe
# ----------------------------------------------------------------------------

class ValueProbe(nn.Module):
    """V(s) = sigmoid(w2 . relu(W1 s)) -- the paper's 2-layer MLP value probe
    (hidden 1024, ReLU), predicting the probability that the trajectory ends
    correct. ``column_l1`` is the pruning statistic: the L1 norm of the first
    layer's weight column for each input dimension."""

    def __init__(self, d_in: int, d_hidden: int = D_PROBE_HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, 1)

    def __call__(self, x):
        return mx.sigmoid(self.l2(nn.relu(self.l1(x))))[..., 0]

    def column_l1(self) -> mx.array:
        # nn.Linear.weight is (out, in): the column of input dim j is [:, j]
        return mx.abs(self.l1.weight).sum(axis=0)


def auc_rank(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """AUC(score, label) as the rank statistic (Mann-Whitney U), ties averaged.
    None when one class is missing. No sklearn in this repo's dependencies."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0          # 1-based average rank
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def keep_count(d: int, ratio: float) -> int:
    """How many input dimensions survive a pruning ratio (at least one)."""
    return max(1, int(round(d * (1.0 - ratio))))


def sink_dims(root, layer: int, n_traj: int = 20, factor: float = 50.0, floor: float = 100.0):
    """Dimensions that carry the attention sink: huge at position 0, ordinary
    everywhere else. Zeroing one of them at any layer above the sink breaks
    attention at every position -- on Qwen2.5-0.5B dim 62 sits at |h| ~1,550 at
    position 0 and 0.5 elsewhere, and the random control that drew it scored 0%
    with no item ever stopping (2026-09-12, layer 16). A random control that can
    hit such a dim measures the sink, not the value neurons, so the ablation's
    --random-exclude-sink keeps them out of the pool. Detected from the first
    n_traj collected trajectories: mean |h| at position 0 over factor times the
    RMS over the other prompt positions, and above an absolute floor.
    """
    import json
    from pathlib import Path
    root = Path(root)
    rows = [json.loads(l) for l in (root / "index.jsonl").read_text().splitlines() if l.strip()][:n_traj]
    p0, rest = [], []
    for r in rows:
        z = np.load(root / "traj" / f"{r['idx']}.npz")[f"h{layer}"].astype(np.float32)
        p0.append(z[0])
        rest.append(z[1:r["prompt_len"]])
    a0 = np.abs(np.stack(p0)).mean(axis=0)
    a1 = np.sqrt((np.concatenate(rest) ** 2).mean(axis=0)) + 1e-6
    return sorted(int(i) for i in np.where((a0 > factor * a1) & (a0 > floor))[0])
