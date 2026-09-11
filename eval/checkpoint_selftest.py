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


if __name__ == "__main__":
    sys.exit(0 if main() < 1e-6 else 1)
