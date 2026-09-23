"""Ternary Bonsai 2 packs (prism-ml/Ternary-Bonsai-2-*-mlx-2bit) as a PsiLM backbone.

The pack stores its language model in a rotated basis: every packed projection's
weights carry a blockwise Hadamard rotation folded in offline, and the runtime
must apply the matching rotation to the activations (and its inverse to the
embedding lookup). Ordinary MLX loaders skip that and return wrong output rather
than an error, so the pack bundles a `runtime/` whose `Packed` module does it.

PsiLM needs the language model only. This builds mlx-lm's Qwen3.5 model, installs
the pack's `Packed` modules at the paths its config lists, and loads the
`language_model.*` tensors; the vision tower's 0.92 GB are never read (MLX loads
safetensors lazily). The result goes into the same Qwen35Tower as the Qwen3.5 9B,
so the staged forward, the bridges' read and write hooks and the grad window
carry over unchanged.

Checked on Ternary-Bonsai-2-27B (revision 3f926b4) against the vendor's own
`vision_artifact.load_vl_model`: last-position logits agree to 3.2e-5 with the
same top five, the chat app's staged prefill (psilm_chat.engine.prefill, one
unpadded prompt) matches the stock forward to 4.2e-5, and the gradient of a
next-token loss reaches the hidden state at layer 48 of 64 through the
differentiable SSM scan, finite. The training path (MlxStream on right-padded
batches, the ops scan above the write layer, the recomputed backward) is checked
on the GPU by results/bonsai/precheck.py before any bridge is trained.
"""
import json
import sys
from pathlib import Path

BONSAI_MODEL_TYPES = {"prism_hadamard_qwen35"}


def is_bonsai_pack(path) -> bool:
    try:
        cfg = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return cfg.get("model_type") in BONSAI_MODEL_TYPES


def _packed_class(pack: Path):
    rt = str(pack / "runtime")
    if rt not in sys.path:
        sys.path.insert(0, rt)
    from runtime import Packed          # noqa: E402  (the pack's own module)
    return Packed


def load_text_model(pack):
    """mlx-lm's qwen3_5 Model holding the pack's packed language model, and the pack config."""
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    pack = Path(pack)
    cfg = json.loads((pack / "config.json").read_text())
    if cfg.get("model_type") not in BONSAI_MODEL_TYPES or cfg.get("base_model_type", "qwen3_5") != "qwen3_5":
        raise ValueError(f"{pack}: not a Hadamard Qwen3.5 pack (model_type {cfg.get('model_type')})")
    if cfg.get("gdn_activation_layout", "grouped") != "grouped":
        raise ValueError(f"{pack}: GDN layout {cfg['gdn_activation_layout']!r}; only 'grouped' is supported")
    q = cfg.get("quantization") or {}
    if (q.get("bits"), q.get("group_size"), q.get("mode")) != (2, 128, "affine"):
        raise ValueError(f"{pack}: quantization {q}; the runtime handles 2-bit / g128 / affine only")
    Packed = _packed_class(pack)

    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": dict(cfg["text_config"])}))
    lm = model.language_model
    weights = mx.load(str(pack / "model.safetensors"))
    prefix = "language_model."
    for record in cfg["modules"]:
        path = record["path"]
        if record["dtype"] != "float16":
            raise ValueError(f"{path}: activation dtype {record['dtype']}")
        block = record["block"]
        if block and block not in (512, 1024, 2048, 4096):
            raise ValueError(f"{path}: block size {block}")
        key = prefix + path
        arrays = [weights[key + "." + s] for s in ("weight", "scales", "biases")]
        signs = weights.get(key + ".signs")
        if block and signs is None:
            raise ValueError(f"{path}: missing sign vector")
        parts = path.split(".")
        parent = lm
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], Packed(arrays, block, signs, record["embedding"], mx.float16))
    model.load_weights([(k, v) for k, v in weights.items() if k.startswith(prefix)], strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, cfg


def eos_ids(pack) -> list:
    pack = Path(pack)
    out = set()
    gen = pack / "generation_config.json"
    if gen.exists():
        e = json.loads(gen.read_text()).get("eos_token_id")
        out |= set(e if isinstance(e, list) else [e]) - {None}
    tc = json.loads((pack / "config.json").read_text()).get("text_config", {})
    if tc.get("eos_token_id") is not None:
        out.add(tc["eos_token_id"])
    return sorted(int(i) for i in out)


def load_bonsai(pack):
    """(tower, stock model, mlx-lm tokenizer) in load_backbone_any's shape."""
    from mlx_lm.utils import load_tokenizer
    from .qwen35_loader import Qwen35Tower

    model, _ = load_text_model(pack)
    tok = load_tokenizer(Path(pack), eos_token_ids=eos_ids(pack))
    return Qwen35Tower(model), model, tok
