#!/usr/bin/env python3
"""Gradient-checkpointed staged forward vs the taped one: are the gradients equal?

MlxStream.checkpoint_from recomputes layers >= L in the backward pass instead
of keeping their intermediates. It exists for Qwen3.5, whose pure-ops SSM scan
keeps its whole recurrence alive and puts injection depths of 24 and below
over this machine's memory (results/qwen35/probes.txt). This checks, on the
4-bit 0.5B, that the transform changes nothing the bridges would see: the
gradient of a scalar built from the finished logits with respect to an array
added to the residual stream mid-stack -- the exact path the trainer's
backward takes -- with checkpointing off, from the injection point, and from
layer 0.

  python eval/checkpoint_selftest.py            # prints the three gradients' agreement
  python eval/checkpoint_selftest.py --device cpu --hybrid-only

A second check (--hybrid, on by default) builds a tiny random Qwen3.5-shaped
hybrid stack (GatedDeltaNet + full attention, the Bonsai/Qwen3.5 layer mix) and
asks the question the gradient check cannot: does recomputing actually save
memory? It trains a delta at depth L through the pure-ops scan above L, taped
and recomputed, for backward windows of 4, 8 and 12 layers, and requires (a)
gradients equal to the taped ones and (b) a recomputed peak that stays nearly
flat as the window grows while the taped one grows with it. mx.checkpoint
failed (b) on the GPU (the 9B constitution runs peaked as if fully taped); see
psilm/mlx/staged.py recompute_in_backward.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402

from psilm.mlx.staged import MlxStream  # noqa: E402


def main(model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
    model, tok = mlx_lm.load(model_id)
    ids = mx.array([tok.encode("The temperature at x = 0.37 is")])
    attn = mx.ones(ids.shape, dtype=mx.int32)
    n = len(model.model.layers)
    l_rev = n // 2

    def loss(delta, cp):
        s = MlxStream(model, ids, attn)
        s.checkpoint_from = cp
        s.run(0, l_rev)
        s.hidden = s.hidden + delta.astype(s.hidden.dtype)
        s.run(l_rev, n)
        lg = s.finish().astype(mx.float32)
        return mx.mean(mx.log(mx.softmax(lg, axis=-1)[..., 0] + 1e-9))

    mx.random.seed(0)
    d = (mx.random.normal(shape=(1, ids.shape[1], model.args.hidden_size)) * 0.01).astype(mx.float32)
    out = {}
    for cp in (None, l_rev, 0):
        v, g = mx.value_and_grad(lambda x: loss(x, cp))(d)
        mx.eval(v, g)
        out[cp] = (float(v), g)
        print(f"checkpoint_from={str(cp):>4}: loss {float(v):.8f}  |grad| {float(mx.linalg.norm(g)):.6e}")
    v0, g0 = out[None]
    worst = 0.0
    for cp in (l_rev, 0):
        v, g = out[cp]
        rel = float(mx.linalg.norm(g - g0) / mx.linalg.norm(g0))
        worst = max(worst, rel)
        print(f"vs taped, checkpoint_from={cp}: |dloss| {abs(v - v0):.3e}  relative grad diff {rel:.3e}")
    print(f"{model_id}: {n} layers, injection at {l_rev}; worst relative gradient difference {worst:.3e}")
    return worst


def hybrid_memory_check(T=256, B=2):
    """Tiny random hybrid stack: gradient equality and peak memory, taped vs recomputed."""
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    from psilm.mlx.qwen35_loader import Qwen35Tower

    D, NL, V = 64, 16, 256
    tc = {"model_type": "qwen3_5_text", "hidden_size": D, "intermediate_size": 128,
          "num_hidden_layers": NL, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": V, "linear_num_value_heads": 4,
          "linear_num_key_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4, "tie_word_embeddings": False,
          "full_attention_interval": 4, "attn_output_gate": True,
          "rope_parameters": {"mrope_interleaved": True, "mrope_section": [1, 1, 0],
                              "partial_rotary_factor": 0.25, "rope_theta": 10000,
                              "rope_type": "default"}}
    mx.random.seed(0)
    stock = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": tc}))
    stock.eval()
    mx.eval(stock.parameters())
    tower = Qwen35Tower(stock)
    ids = mx.random.randint(2, V, (B, T))
    attn = mx.ones((B, T), dtype=mx.int32)
    attn[1, T - T // 4:] = 0                          # a right-padded row, as in training
    d = mx.random.normal((B, T, D)) * 0.01

    def loss(delta, lo, cp):
        s = MlxStream(tower, ids, attn)
        s.checkpoint_from = cp
        s.run(0, lo)
        s.hidden = s.hidden + delta.astype(s.hidden.dtype)
        s.run(lo, NL)
        lg = s.finish().astype(mx.float32)
        return mx.mean(mx.logsumexp(lg, axis=-1) - lg[..., 0])

    rows, worst = [], 0.0
    for lo in (12, 8, 4):
        tower.set_grad_window(lo)
        res = {}
        for cp in (None, lo):
            f = mx.value_and_grad(lambda x: loss(x, lo, cp))
            mx.eval(f(d))                               # warm-up: compile, allocate
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            v, g = f(d)
            mx.eval(v, g)
            res[cp] = (float(v), g, (mx.get_peak_memory() - base) / 2**20)
        rel = float(mx.linalg.norm(res[lo][1] - res[None][1]) / mx.linalg.norm(res[None][1]))
        worst = max(worst, rel)
        rows.append((NL - lo, res[None][2], res[lo][2]))
        print(f"hybrid window {NL - lo:2d} layers: taped peak {res[None][2]:8.1f} MiB  "
              f"recomputed {res[lo][2]:8.1f} MiB  relative grad diff {rel:.3e}")
    taped_growth = rows[-1][1] - rows[0][1]
    recomputed_growth = rows[-1][2] - rows[0][2]
    ok = worst < 1e-6 and recomputed_growth < 0.25 * taped_growth
    print(f"hybrid: taped grows {taped_growth:.1f} MiB from 4 to 12 window layers, recomputed "
          f"{recomputed_growth:.1f} MiB; gradients {'equal' if worst < 1e-6 else 'DIFFER'}; "
          f"{'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--hybrid-only", action="store_true",
                    help="skip the 0.5B gradient check (no download, no model load)")
    ap.add_argument("--no-hybrid", action="store_true")
    a = ap.parse_args()
    if a.device == "cpu":
        mx.set_default_device(mx.cpu)
    ok = True
    if not a.hybrid_only:
        ok = main() < 1e-6
    if not a.no_hybrid:
        ok = hybrid_memory_check() and ok
    sys.exit(0 if ok else 1)
