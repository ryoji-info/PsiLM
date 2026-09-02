"""Language tower of the Qwen3.5 vision-language backbone, in MlxStream layout.

Target: mlx-community/Qwen3.8-27B-4bit (config ``model_type = "qwen3_5"``,
``architectures = ["Qwen3_5ForConditionalGeneration"]``).  PsiLM only needs
the text tower.  The checkpoint is loaded through mlx_lm's own
``mlx_lm.models.qwen3_5`` implementation -- ``mlx_vlm`` is neither installed
nor needed: mlx_lm's ``Model`` contains no vision module and its ``sanitize``
drops every ``vision_tower.*`` tensor before ``load_weights`` -- and exposed
with EXACTLY the attribute layout ``psilm.mlx.staged.MlxStream`` drives::

    tower.model.layers[i](x, mask=..., cache=None)   # per-layer call
    tower.model.embed_tokens(ids)  (+ .as_linear when tied)
    tower.model.norm
    tower.lm_head                                     # untied here
    tower.args.hidden_size
    tower(ids)                                        # one-shot reference logits
    tower.freeze()

Architecture facts (config.json + safetensors headers; no weights touched):

* mlx_lm nesting: ``Model.language_model`` (TextModel)
    ``.model`` (Qwen3_5TextModel): ``embed_tokens``, ``layers[64]``, ``norm``
    ``.lm_head`` (``tie_word_embeddings=False`` -> a real QuantizedLinear)
    ``.args`` (TextModelArgs): ``hidden_size=5120``, ``num_hidden_layers=64`` ...
  Checkpoint keys are already ``language_model.model.*`` /
  ``language_model.lm_head.*`` (plus ``vision_tower.*``, dropped).
* hidden 5120, 64 layers, vocab 248320, 4-bit affine / group 64 for every
  Linear AND the embedding AND lm_head; bf16 scales -> hidden dtype bf16.
* Hybrid stack, ``layer_types`` = 3x linear_attention + 1x full_attention
  repeated: full attention at ``i % 4 == 3`` (16 layers), Gated DeltaNet
  linear attention elsewhere (48 layers).  mlx_lm sets
  ``is_linear = (i + 1) % full_attention_interval != 0``; we cross-check
  that against the config's ``layer_types``.
* Full attention: 24 heads x head_dim 256, 4 KV heads, gated output
  (``q_proj`` emits 2*heads*head_dim: query + sigmoid gate), q/k RMSNorm,
  partial RoPE on 64 of 256 dims, theta 1e7.  ``mrope_section`` /
  ``mrope_interleaved`` are ignored by mlx_lm's text path (plain 1-D
  ``nn.RoPE``), which is exact for pure-text input because all three M-RoPE
  position streams coincide with the token index.
* Linear attention (GatedDeltaNet): 16 key heads x 128, 48 value heads x
  128, causal depthwise conv1d (kernel 4), recurrent state
  ``(B, 48, 128, 128)`` fp32 per layer.

Layer-call signature differences vs the plain mlx_lm transformers MlxStream
was written for (Qwen2.5 / Qwen3 / Qwen3-MoE):

1. MASK.  ``DecoderLayer.__call__(x, mask, cache)`` has the same signature,
   but the two layer kinds want DIFFERENT masks.  Full-attention layers take
   the additive ``(B,1,L,L)`` mask MlxStream builds (straight into
   ``mx.fast.scaled_dot_product_attention``).  GatedDeltaNet layers take a
   ``(B, S)`` boolean token-validity mask (or None): it is applied as
   ``mx.where(mask[..., None], qkv, 0)`` and indexed per ``(b, t)`` inside the
   recurrence kernel, so the 4-D additive mask would break the broadcast.
   ``LayerShim`` derives the ``(B, S)`` mask from the LAST QUERY ROW of the
   additive mask (a causal row sees every key, so its unblocked entries are
   exactly the valid keys) and broadcasts it to the batch size (the Metal
   kernel indexes ``mask[b*T + t]``, so a ``(1, S)`` mask would read out of
   bounds for ``B > 1``).
2. CACHE.  ``cache=None`` is supported by both kinds (rope offset 0, zero
   conv state, zero recurrent state) -- what MlxStream passes.
3. GRADIENTS.  GatedDeltaNet selects its recurrence with
   ``use_kernel = not self.training``.  In eval mode (what ``mlx_lm.load``
   sets) it runs a custom ``mx.fast.metal_kernel`` which has NO VJP
   (``fast::CustomKernel`` never overrides ``Primitive::vjp``), so a backward
   pass through any linear-attention layer raises.  PsiLM needs gradients
   through every layer AFTER the injection point ``l_rev`` and none before.
   ``LanguageTower.set_differentiable(from_layer)`` flips exactly the
   GatedDeltaNet modules of layers ``>= from_layer`` to training mode (the
   ops path: a Python loop over T of an ``mx.compile``'d step; same maths,
   fp rounding only), leaving earlier layers on the fast kernel.  The ops
   path retains ~2 recurrent states per timestep for backward
   (2 * 48*128*128 * 4 B = 6.3 MB per batch row per token per layer), so
   those layers are also wrapped in ``mx.checkpoint`` by default: peak memory
   is then ONE layer's worth instead of all post-injection layers'.
4. PADDING.  Right padding (what ``psilm.stage2.qa.make_batch`` emits) is
   exact for both kinds.  Left padding is not position-exact for the
   full-attention layers (RoPE positions are not shifted; mlx_lm does that
   through ``cache.left_padding``, which MlxStream never uses) -- the same
   limitation as the existing 8B path.

CPU self-test (tiny synthetic model, no checkpoint weights)::

    python -m psilm.mlx.vlm_loader --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

DEFAULT_REPO = "mlx-community/Qwen3.8-27B-4bit"

# Additive-mask entries at or below this are "blocked" (staged.padded_causal_mask
# uses -1e9, or -65504 for float16); unblocked entries are exactly 0.
_BLOCKED_BELOW = -1e4

GB = 1e9


# --------------------------------------------------------------------------- #
# checkpoint discovery / config (no weights)
# --------------------------------------------------------------------------- #

def snapshot_dir(repo_id: str = DEFAULT_REPO) -> Path:
    """Local snapshot directory of an HF repo id (or a local path). Offline only."""
    p = Path(repo_id).expanduser()
    if p.exists():
        return p
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        pass
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = Path(HF_HUB_CACHE)
    except Exception:
        cache = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        snap = repo_dir / "snapshots" / ref.read_text().strip()
        if snap.exists():
            return snap
    snaps = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").exists() else []
    if snaps:
        return snaps[-1]
    raise FileNotFoundError(f"{repo_id}: not in the local HF cache ({cache})")


def hf_tokenizer_path(repo_id: str = DEFAULT_REPO) -> str:
    """Path for ``AutoTokenizer.from_pretrained`` -- the QA builder must use the
    backbone's OWN tokenizer (vocab 248320 here; Qwen/Qwen3-8B is 151936)."""
    return str(snapshot_dir(repo_id))


def read_config(repo_id: str = DEFAULT_REPO) -> dict:
    return json.loads((snapshot_dir(repo_id) / "config.json").read_text())


def text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def layer_types_of(cfg: dict) -> List[str]:
    tc = text_config(cfg)
    n = tc["num_hidden_layers"]
    if tc.get("layer_types"):
        return list(tc["layer_types"])
    interval = tc.get("full_attention_interval")
    if not interval:                      # plain transformer
        return ["full_attention"] * n
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(n)]


def default_coupling(n_layers: int) -> Tuple[int, int]:
    """(l_fwd, l_rev) exactly as PsiLMMLX derives them (10/24 and 15/24 depth)."""
    return round(n_layers * 10 / 24), round(n_layers * 15 / 24)


def describe(cfg: dict) -> dict:
    tc = text_config(cfg)
    lt = layer_types_of(cfg)
    hd = tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
    rp = tc.get("rope_parameters") or {}
    prf = rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0))
    q = cfg.get("quantization") or tc.get("quantization") or {}
    return {
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc.get("intermediate_size"),
        "num_layers": tc["num_hidden_layers"],
        "n_linear": sum(t == "linear_attention" for t in lt),
        "n_full": sum(t == "full_attention" for t in lt),
        "full_attention_indices": [i for i, t in enumerate(lt) if t == "full_attention"],
        "vocab_size": tc["vocab_size"],
        "tie_word_embeddings": tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)),
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc.get("num_key_value_heads"),
        "head_dim": hd,
        "rope_dims": int(hd * prf),
        "rope_theta": rp.get("rope_theta", tc.get("rope_theta")),
        "linear_attention": {
            "num_k_heads": tc.get("linear_num_key_heads"),
            "k_dim": tc.get("linear_key_head_dim"),
            "num_v_heads": tc.get("linear_num_value_heads"),
            "v_dim": tc.get("linear_value_head_dim"),
            "conv_kernel": tc.get("linear_conv_kernel_dim"),
        } if "linear_num_value_heads" in tc else None,
        "quantization": q,
        "eos_token_id": cfg.get("eos_token_id"),
        "default_coupling": default_coupling(tc["num_hidden_layers"]),
        "layer_types": lt,
    }


def checkpoint_bytes(repo_id: str = DEFAULT_REPO) -> dict:
    """Byte totals from the safetensors HEADERS only (nothing is loaded)."""
    snap = snapshot_dir(repo_id)
    files = sorted(snap.glob("model*.safetensors"))
    lang = vis = 0
    keys: List[str] = []
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            if k.startswith(("vision_tower", "model.visual")):
                vis += b
            else:
                lang += b
                keys.append(k)
    return {"language_tower": lang, "vision_tower": vis, "total": lang + vis,
            "n_files": len(files), "language_keys": sorted(keys)}


def analytic_weight_bytes(cfg: dict) -> int:
    """Language-tower bytes from the config alone (cross-checks the headers)."""
    tc = text_config(cfg)
    d = describe(cfg)
    q = d["quantization"]
    bits, gs = q.get("bits", 16), q.get("group_size", 64)
    H, I, V, n = tc["hidden_size"], tc["intermediate_size"], tc["vocab_size"], tc["num_hidden_layers"]
    hd, nh = d["head_dim"], tc["num_attention_heads"]
    nkv = tc.get("num_key_value_heads", nh)
    n_lin, n_full = d["n_linear"], d["n_full"]
    la = d["linear_attention"] or {}
    kd = (la.get("num_k_heads") or 0) * (la.get("k_dim") or 0)
    vd = (la.get("num_v_heads") or 0) * (la.get("v_dim") or 0)
    nv = la.get("num_v_heads") or 0
    conv_dim = 2 * kd + vd
    lin = H * conv_dim + H * vd + 2 * H * nv + vd * H            # in_proj_qkv, z, a+b, out_proj
    attn = H * (2 * nh * hd) + 2 * H * (nkv * hd) + (nh * hd) * H  # gated q, k, v, o
    mlp = 3 * H * I
    quantizable = n_lin * lin + n_full * attn + n * mlp + V * H * (1 if d["tie_word_embeddings"] else 2)
    other = (n_lin * (conv_dim * (la.get("conv_kernel") or 0) + 2 * nv + (la.get("v_dim") or 0))
             + n_full * 2 * hd + n * 2 * H + H)
    per_weight_bits = bits + (2 * 16 / gs if bits < 16 else 0)   # bf16 scales + biases per group
    return int(quantizable * per_weight_bits / 8 + other * 2)


# --------------------------------------------------------------------------- #
# mask adaptation
# --------------------------------------------------------------------------- #

def ssm_mask_from_additive(mask, batch_size: int):
    """(B,1,L,L) additive / boolean attention mask -> (B, S) bool token mask.

    None or "causal" -> None (no padding).  Accepts 2-D masks too (bool as
    is, integer 0/1 attention masks, float additive masks)."""
    if mask is None or isinstance(mask, str):
        return None
    if mask.ndim == 4:
        row = mask[:, 0, -1, :]
    elif mask.ndim == 3:
        row = mask[:, -1, :]
    elif mask.ndim == 2:
        row = mask
    else:
        raise ValueError(f"unsupported mask rank {mask.ndim}")
    if row.dtype == mx.bool_:
        ok = row
    elif mx.issubdtype(row.dtype, mx.integer):
        ok = row != 0
    else:
        ok = row > _BLOCKED_BELOW
    return mx.broadcast_to(ok, (batch_size, ok.shape[-1]))


class LayerShim:
    """Per-layer adapter with the mlx_lm layer signature ``(x, mask, cache)``.

    Routes the additive attention mask to full-attention layers unchanged and
    a derived ``(B, S)`` token mask to Gated DeltaNet layers; optionally wraps
    the call in ``mx.checkpoint`` (only meaningful for layers that are on the
    backward path).  Attribute access falls through to the wrapped layer
    (``shim.mlp``, ``shim.self_attn``, ``shim.linear_attn`` ...)."""

    def __init__(self, layer, index: int, is_linear: Optional[bool] = None,
                 checkpoint: bool = False):
        self.layer = layer
        self.index = index
        self.is_linear = bool(getattr(layer, "is_linear", False)) if is_linear is None else bool(is_linear)
        self.checkpoint = checkpoint

    def __call__(self, x, mask=None, cache=None):
        if self.is_linear:
            mask = ssm_mask_from_additive(mask, x.shape[0])
        if self.checkpoint and cache is None:
            layer, m = self.layer, mask
            return mx.checkpoint(lambda h: layer(h, mask=m, cache=None))(x)
        return self.layer(x, mask=mask, cache=cache)

    @property
    def sequence_mixer(self):
        return self.layer.linear_attn if self.is_linear else self.layer.self_attn

    def __getattr__(self, name):
        if name == "layer":
            raise AttributeError(name)
        return getattr(self.layer, name)

    def __repr__(self):
        kind = "linear_attention" if self.is_linear else "full_attention"
        return f"LayerShim({self.index}, {kind}, checkpoint={self.checkpoint})"


class _InnerView:
    """``tower.model``: live views of embed_tokens / norm, shimmed layers."""

    def __init__(self, inner, shims: List[LayerShim]):
        self._inner = inner
        self.layers = shims

    @property
    def embed_tokens(self):
        return self._inner.embed_tokens

    @property
    def norm(self):
        return self._inner.norm

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class LanguageTower:
    """MlxStream-shaped view of an mlx_lm text model (VLM tower or plain LM)."""

    def __init__(self, text_model, *, full_model=None, config: Optional[dict] = None):
        self.text_model = text_model
        self.full_model = full_model
        self.config = config or {}
        inner = text_model.model
        self._shims = [LayerShim(l, i) for i, l in enumerate(inner.layers)]
        self.layer_types = ["linear_attention" if s.is_linear else "full_attention" for s in self._shims]
        if self.config:
            expect = layer_types_of(self.config)
            if len(expect) == len(self.layer_types) and expect != self.layer_types:
                bad = [i for i, (a, b) in enumerate(zip(expect, self.layer_types)) if a != b]
                raise ValueError(f"layer type mismatch vs config.layer_types at {bad[:8]}")
        self.model = _InnerView(inner, self._shims)
        self._diff_from: Optional[int] = None
        self._checkpoint = True

    # ---- MlxStream contract -------------------------------------------------
    @property
    def args(self):
        return self.text_model.args

    @property
    def lm_head(self):
        try:
            return self.text_model.lm_head
        except AttributeError:
            raise AttributeError("lm_head")          # tied: MlxStream uses embed_tokens.as_linear

    def __call__(self, inputs, cache=None, input_embeddings=None):
        """One-shot reference forward (mlx_lm's own path: "causal" + no SSM mask)."""
        return self.text_model(inputs, cache=cache, input_embeddings=input_embeddings)

    # ---- nn.Module-ish delegation ------------------------------------------
    def freeze(self, *a, **k):
        return self.text_model.freeze(*a, **k)

    def unfreeze(self, *a, **k):
        return self.text_model.unfreeze(*a, **k)

    def parameters(self):
        return self.text_model.parameters()

    def trainable_parameters(self):
        return self.text_model.trainable_parameters()

    def eval(self):
        self.text_model.eval()
        self._apply_modes()
        return self

    def train(self, mode: bool = True):
        self.text_model.train(mode)
        self._apply_modes()
        return self

    def __getattr__(self, name):
        if name in ("text_model", "_shims"):
            raise AttributeError(name)
        return getattr(self.text_model, name)

    # ---- structure ------------------------------------------------------------
    @property
    def layers(self) -> List[LayerShim]:
        return self._shims

    @property
    def raw_layers(self):
        return self.text_model.model.layers

    @property
    def n_layers(self) -> int:
        return len(self._shims)

    @property
    def hidden_size(self) -> int:
        return self.text_model.args.hidden_size

    def linear_layer_indices(self) -> List[int]:
        return [s.index for s in self._shims if s.is_linear]

    def full_attention_indices(self) -> List[int]:
        return [s.index for s in self._shims if not s.is_linear]

    # ---- gradient regime ------------------------------------------------------
    @property
    def differentiable_from(self) -> Optional[int]:
        return self._diff_from

    def set_differentiable(self, from_layer: Optional[int], checkpoint: bool = True):
        """Make layers ``>= from_layer`` safe for backward: their Gated DeltaNet
        modules use the ops recurrence (differentiable) instead of the Metal
        kernel (no VJP), and are gradient-checkpointed when ``checkpoint``.
        ``None`` restores the all-kernel inference regime."""
        self._diff_from = from_layer
        self._checkpoint = checkpoint
        self._apply_modes()
        return self

    def _apply_modes(self):
        for s in self._shims:
            diff = self._diff_from is not None and s.index >= self._diff_from
            s.checkpoint = bool(diff and self._checkpoint)
            if s.is_linear:
                s.layer.linear_attn.train(diff)      # training=True -> use_kernel=False

    def regime(self) -> dict:
        return {
            "differentiable_from": self._diff_from,
            "checkpoint": self._checkpoint,
            "ops_path_layers": [s.index for s in self._shims
                                if s.is_linear and s.layer.linear_attn.training],
            "kernel_path_layers": [s.index for s in self._shims
                                   if s.is_linear and not s.layer.linear_attn.training],
        }

    def __repr__(self):
        d = self.regime()
        return (f"LanguageTower(layers={self.n_layers}, hidden={self.hidden_size}, "
                f"linear={len(self.linear_layer_indices())}, full={len(self.full_attention_indices())}, "
                f"differentiable_from={d['differentiable_from']}, checkpoint={d['checkpoint']})")


def wrap_language_tower(model, config: Optional[dict] = None) -> LanguageTower:
    """``qwen3_5.Model`` (has ``.language_model``) or any mlx_lm text model."""
    text = getattr(model, "language_model", None) or model
    return LanguageTower(text, full_model=model if text is not model else None, config=config)


# --------------------------------------------------------------------------- #
# loading (touches weights -- never call while the GPU is busy)
# --------------------------------------------------------------------------- #

def load_language_tower(repo_id: str = DEFAULT_REPO, *, lazy: bool = False,
                        differentiable_from: Optional[int] = None,
                        checkpoint: bool = True, tokenizer_config: Optional[dict] = None):
    """-> (LanguageTower, TokenizerWrapper).  Same machinery as ``mlx_lm.load``
    (``load_model`` + ``load_tokenizer``), resolved offline from the HF cache.
    Vision-tower tensors are dropped by ``Model.sanitize`` and never evaluated."""
    from mlx_lm.utils import load_model, load_tokenizer

    path = snapshot_dir(repo_id)
    cfg = read_config(path)
    model, loaded_cfg = load_model(path, lazy=lazy)
    tok = load_tokenizer(path, tokenizer_config, eos_token_ids=loaded_cfg.get("eos_token_id"))
    tower = wrap_language_tower(model, cfg)
    tower.freeze()
    tower.eval()
    if differentiable_from is not None:
        tower.set_differentiable(differentiable_from, checkpoint=checkpoint)
    return tower, tok


# --------------------------------------------------------------------------- #
# memory budget (pure arithmetic)
# --------------------------------------------------------------------------- #

def device_memory() -> dict:
    info: Dict[str, Any] = {}
    for getter in (getattr(mx, "device_info", None), getattr(getattr(mx, "metal", None), "device_info", None)):
        if getter is None:
            continue
        try:
            info = dict(getter())
            break
        except Exception:
            continue
    return {"device_name": info.get("device_name"),
            "memory_size": info.get("memory_size"),
            "max_recommended_working_set": info.get("max_recommended_working_set_size")}


def system_free_bytes() -> Optional[int]:
    """macOS: free + inactive + speculative pages (what other processes leave)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    page = 16384
    tot = 0
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            if line.startswith(key):
                tot += int(line.split(":")[1].strip().rstrip("."))
    return tot * page if tot else None


def memory_estimate(cfg: dict, batch: int = 1, seq_len: int = 128, l_rev: Optional[int] = None,
                    weights_bytes: Optional[int] = None, checkpoint: bool = True,
                    act_bytes: int = 2, overhead: int = int(0.75 * GB)) -> dict:
    """Training-step budget for PsiLM coupling at ``l_rev`` (bytes).

    Components: 4-bit weights; activations autograd retains in the
    post-injection layers (matmul inputs, bf16, ~6*hidden + 3*intermediate per
    token per layer); logits (bf16 out + f32 cast + f32 grad); Gated DeltaNet
    BPTT states (2 fp32 states of ``Hv*Dv*Dk`` per row per token per linear
    layer -- ONE layer at a time with checkpointing, all post-injection linear
    layers without); fixed overhead (runtime, tokenizer, bridges, FNO)."""
    tc = text_config(cfg)
    d = describe(cfg)
    n = d["num_layers"]
    lt = d["layer_types"]
    l_rev = default_coupling(n)[1] if l_rev is None else l_rev
    post = list(range(l_rev, n))
    n_lin_post = sum(lt[i] == "linear_attention" for i in post)
    tokens = batch * seq_len
    H, I, V = tc["hidden_size"], tc["intermediate_size"], tc["vocab_size"]
    weights = weights_bytes if weights_bytes is not None else analytic_weight_bytes(cfg)
    act = tokens * (6 * H + 3 * I) * act_bytes * len(post)
    logits = tokens * V * (act_bytes + 4 + 4)
    la = d["linear_attention"] or {}
    state = 2 * (la.get("num_v_heads") or 0) * (la.get("v_dim") or 0) * (la.get("k_dim") or 0) * 4
    bptt_layer = tokens * state
    bptt = bptt_layer * (1 if checkpoint else max(n_lin_post, 1))
    fwd = batch * seq_len * H * act_bytes * 4
    total = weights + act + logits + bptt + fwd + overhead
    dm = device_memory()
    ws = dm.get("max_recommended_working_set")
    return {
        "batch": batch, "seq_len": seq_len, "l_rev": l_rev,
        "post_injection_layers": len(post), "post_injection_linear_layers": n_lin_post,
        "checkpoint": checkpoint,
        "weights": weights, "activations": act, "logits": logits,
        "deltanet_bptt": bptt, "deltanet_bptt_per_layer": bptt_layer,
        "forward_transient": fwd, "overhead": overhead, "total": total,
        "inference_total": weights + fwd + overhead,
        "device": dm, "fits_working_set": (ws is None) or (total <= ws),
        "headroom_vs_working_set": (ws - total) if ws else None,
    }


def format_estimate(est: dict) -> str:
    dm = est["device"]
    ws = dm.get("max_recommended_working_set")
    lines = [
        f"memory budget  batch={est['batch']} seq_len={est['seq_len']} l_rev={est['l_rev']} "
        f"(post-injection layers {est['post_injection_layers']}, of which linear "
        f"{est['post_injection_linear_layers']}; checkpoint={est['checkpoint']})",
        f"  weights (4-bit language tower) {est['weights'] / GB:7.2f} GB",
        f"  post-injection activations    {est['activations'] / GB:7.2f} GB",
        f"  logits (V x {est['seq_len']} x {est['batch']})       {est['logits'] / GB:7.2f} GB",
        f"  DeltaNet BPTT states          {est['deltanet_bptt'] / GB:7.2f} GB "
        f"({est['deltanet_bptt_per_layer'] / GB:.2f} GB per layer)",
        f"  forward transient + overhead  {(est['forward_transient'] + est['overhead']) / GB:7.2f} GB",
        f"  TRAINING STEP TOTAL           {est['total'] / GB:7.2f} GB",
        f"  inference only                {est['inference_total'] / GB:7.2f} GB",
    ]
    if ws:
        lines.append(f"  device {dm.get('device_name')}: RAM {dm['memory_size'] / GB:.1f} GB, "
                     f"Metal recommended working set {ws / GB:.2f} GB -> "
                     f"{'FITS' if est['fits_working_set'] else 'DOES NOT FIT'} "
                     f"(headroom {est['headroom_vs_working_set'] / GB:+.2f} GB)")
    free = system_free_bytes()
    if free is not None:
        lines.append(f"  currently free+inactive on this machine: {free / GB:.1f} GB")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CPU self-test on a tiny synthetic qwen3_5 model (no checkpoint weights)
# --------------------------------------------------------------------------- #

def make_tiny_model(n_layers: int = 8, hidden: int = 64, vocab: int = 97, tie: bool = False,
                    quantize: bool = True, seed: int = 0):
    """Tiny ``mlx_lm.models.qwen3_5.Model`` with the real module classes."""
    from mlx_lm.models import qwen3_5 as q

    interval = 4
    tc = dict(
        model_type="qwen3_5_text", hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        rms_norm_eps=1e-6, vocab_size=vocab, max_position_embeddings=4096,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=tie,
        attention_bias=False, full_attention_interval=interval,
        layer_types=["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                     for i in range(n_layers)],
        rope_parameters={"rope_type": "default", "mrope_section": [11, 11, 10],
                         "mrope_interleaved": True, "partial_rotary_factor": 0.25,
                         "rope_theta": 10000.0},
    )
    cfg = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
           "text_config": tc, "tie_word_embeddings": tie,
           "quantization": {"group_size": 64, "bits": 4, "mode": "affine"} if quantize else {}}
    mx.random.seed(seed)
    model = q.Model(q.ModelArgs.from_dict(cfg))
    if quantize:
        nn.quantize(model, group_size=64, bits=4,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized"))
    model.eval()
    mx.eval(model.parameters())
    return model, cfg


def selftest_cpu(verbose: bool = True, repo_id: str = DEFAULT_REPO) -> dict:
    """Runs entirely on the CPU stream with a tiny model.  Checks: attribute
    layout (tied and untied), staged-vs-one-shot parity, right-padded batch
    parity, the SSM mask derivation (left-padded single linear layer),
    gradient flow through post-injection layers with and without
    checkpointing, and -- if the 27B snapshot is present -- that the
    checkpoint's language keys equal the module tree's parameter names and
    that the analytic weight-byte count matches the safetensors headers."""
    from mlx.utils import tree_flatten
    from .staged import MlxStream, padded_causal_mask

    res: Dict[str, Any] = {}
    say = print if verbose else (lambda *a, **k: None)
    with mx.stream(mx.cpu):
        assert mx.default_device() == mx.cpu
        model, cfg = make_tiny_model(n_layers=8)
        tower = wrap_language_tower(model, cfg)
        n = tower.n_layers
        res["layout"] = {
            "layers": n, "hidden": tower.args.hidden_size,
            "has_lm_head": hasattr(tower, "lm_head"),
            "embed_is_quantized": hasattr(tower.model.embed_tokens, "scales"),
            "linear": tower.linear_layer_indices(), "full": tower.full_attention_indices(),
        }
        assert res["layout"]["has_lm_head"] and res["layout"]["hidden"] == 64
        assert res["layout"]["full"] == [3, 7]
        tied, _ = make_tiny_model(n_layers=4, tie=True)
        ttower = wrap_language_tower(tied)
        assert not hasattr(ttower, "lm_head") and hasattr(ttower.model.embed_tokens, "as_linear")
        ids_t = mx.array([[1, 2, 3, 4]])
        st = MlxStream(ttower, ids_t)
        st.run(0, ttower.n_layers)
        res["tied_parity"] = mx.abs(ttower(ids_t) - st.finish()).max().item()
        say(f"[selftest] layout ok: {res['layout']}  tied-parity {res['tied_parity']:.2e}")

        # 1) staged vs one-shot, single sequence, in chunks
        L = 12
        ids = mx.random.randint(0, 97, (1, L))
        ref = tower(ids)
        st = MlxStream(tower, ids)
        st.run(0, 3); st.run(3, 5); st.run(5, n)
        out = st.finish()
        res["parity_single"] = mx.abs(ref - out).max().item()
        say(f"[selftest] staged vs one-shot parity: {res['parity_single']:.2e}")
        assert res["parity_single"] < 1e-4

        # 2) right-padded batch parity against unpadded rows
        ids_b = mx.random.randint(0, 97, (1, 9))
        ref_b = tower(ids_b)
        pad = mx.concatenate([ids_b, mx.zeros((1, L - 9), dtype=ids_b.dtype)], axis=1)
        batch = mx.concatenate([ids, pad], axis=0)
        attn = mx.array([[1] * L, [1] * 9 + [0] * (L - 9)], dtype=mx.int32)
        st = MlxStream(tower, batch, attn)
        st.run(0, n)
        outb = st.finish()
        res["parity_padded_row0"] = mx.abs(outb[0:1] - ref).max().item()
        res["parity_padded_row1"] = mx.abs(outb[1:2, :9] - ref_b).max().item()
        say(f"[selftest] right-padded batch parity: row0 {res['parity_padded_row0']:.2e} "
            f"row1 {res['parity_padded_row1']:.2e}")
        assert max(res["parity_padded_row0"], res["parity_padded_row1"]) < 1e-4

        # 3) SSM mask derivation: left-padded linear layer == unpadded on real positions
        shim0 = tower.model.layers[0]
        assert shim0.is_linear
        x = mx.random.normal((1, L, 64))
        x_pad = mx.concatenate([mx.random.normal((1, 3, 64)), x], axis=1)
        attn_l = mx.array([[0, 0, 0] + [1] * L], dtype=mx.int32)
        m4 = padded_causal_mask(attn_l, x.dtype)
        m2 = ssm_mask_from_additive(m4, 1)
        assert m2.shape == (1, L + 3) and m2.dtype == mx.bool_
        assert (m2[0, :3] == False).all().item() and m2[0, 3:].all().item()  # noqa: E712
        y_pad = shim0(x_pad, mask=m4)[:, 3:]
        y = shim0(x, mask=None)
        res["ssm_mask_left_pad"] = mx.abs(y_pad - y).max().item()
        say(f"[selftest] SSM mask (left-padded linear layer vs unpadded): {res['ssm_mask_left_pad']:.2e}")
        assert res["ssm_mask_left_pad"] < 1e-4
        # batch broadcast of a (1,S) row to B rows
        assert ssm_mask_from_additive(m4, 3).shape == (3, L + 3)

        # 4) gradients through post-injection layers, ops path, with/without checkpoint
        l_rev = 5

        def loss_fn(delta, ckpt):
            tower.set_differentiable(l_rev, checkpoint=ckpt)
            s = MlxStream(tower, ids)
            s.run(0, l_rev)
            s.hidden = s.hidden + delta
            s.run(l_rev, n)
            lg = s.finish()
            return (lg.astype(mx.float32) ** 2).mean()

        delta = mx.zeros((1, L, 64))
        g_ck = mx.grad(lambda d: loss_fn(d, True))(delta)
        g_pl = mx.grad(lambda d: loss_fn(d, False))(delta)
        mx.eval(g_ck, g_pl)
        res["grad_norm"] = mx.sqrt((g_pl ** 2).sum()).item()
        res["grad_checkpoint_diff"] = mx.abs(g_ck - g_pl).max().item()
        reg = tower.regime()
        res["regime"] = reg
        assert reg["ops_path_layers"] == [i for i in tower.linear_layer_indices() if i >= l_rev]
        assert reg["kernel_path_layers"] == [i for i in tower.linear_layer_indices() if i < l_rev]
        assert res["grad_norm"] > 0 and res["grad_checkpoint_diff"] < 1e-5
        say(f"[selftest] grad through layers {l_rev}..{n}: |g|={res['grad_norm']:.3e}, "
            f"checkpoint vs plain {res['grad_checkpoint_diff']:.2e}; regime {reg}")
        # parity unchanged in the differentiable regime (ops path on CPU anyway)
        s = MlxStream(tower, ids); s.run(0, n)
        res["parity_differentiable"] = mx.abs(ref - s.finish()).max().item()
        assert res["parity_differentiable"] < 1e-4
        tower.eval()
        assert tower.regime()["ops_path_layers"] == [i for i in tower.linear_layer_indices() if i >= l_rev]
        tower.set_differentiable(None)
        assert tower.regime()["ops_path_layers"] == []

        # 5) checkpoint key layout vs module tree (headers only), analytic bytes
        try:
            real_cfg = read_config(repo_id)
            cb = checkpoint_bytes(repo_id)
        except FileNotFoundError:
            res["key_layout"] = "snapshot not present"
        else:
            d = describe(real_cfg)
            big, _ = make_tiny_model(n_layers=d["num_layers"], tie=d["tie_word_embeddings"])
            names = sorted(k for k, _ in tree_flatten(big.parameters()))
            ck = set(cb["language_keys"])
            missing = sorted(set(names) - ck)
            unexpected = sorted(ck - set(names))
            ana = analytic_weight_bytes(real_cfg)
            res["key_layout"] = {"module_params": len(names), "checkpoint_language_keys": len(ck),
                                 "missing_in_checkpoint": missing[:5], "unexpected_in_checkpoint": unexpected[:5],
                                 "header_bytes": cb["language_tower"], "vision_bytes": cb["vision_tower"],
                                 "analytic_bytes": ana, "analytic_rel_err": ana / cb["language_tower"] - 1}
            say(f"[selftest] key layout: {len(names)} module params vs {len(ck)} checkpoint keys; "
                f"missing {len(missing)} unexpected {len(unexpected)}; header {cb['language_tower'] / GB:.3f} GB "
                f"vs analytic {ana / GB:.3f} GB ({res['key_layout']['analytic_rel_err']:+.2%})")
            assert not missing and not unexpected
            assert abs(res["key_layout"]["analytic_rel_err"]) < 0.05
    res["ok"] = True
    say("[selftest] OK")
    return res


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="tiny-model CPU self-test (no weights)")
    ap.add_argument("--describe", action="store_true", help="print config summary + memory budget")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--l-rev", type=int, default=None)
    args = ap.parse_args()
    if args.describe or not args.selftest:
        cfg = read_config(args.repo)
        d = describe(cfg)
        print(json.dumps({k: v for k, v in d.items() if k != "layer_types"}, indent=1))
        cb = checkpoint_bytes(args.repo)
        print(f"checkpoint: language {cb['language_tower'] / GB:.3f} GB, vision {cb['vision_tower'] / GB:.3f} GB "
              f"({cb['n_files']} shards); analytic {analytic_weight_bytes(cfg) / GB:.3f} GB")
        for ck in (True, False):
            print(format_estimate(memory_estimate(cfg, args.batch, args.seq_len, args.l_rev,
                                                  weights_bytes=cb["language_tower"], checkpoint=ck)))
    if args.selftest:
        t0 = time.time()
        selftest_cpu(repo_id=args.repo)
        print(f"selftest {time.time() - t0:.1f}s")


if __name__ == "__main__":
    _main()
