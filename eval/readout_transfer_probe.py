#!/usr/bin/env python3
"""Does the span readout with mode-shared heads transfer where the pooled one does not?

The 12B multi-mode run costs ~35 GPU-hours, so the coverage hypothesis is
tested first on a 0.5B backbone, cheaply and teacher-forced: train ONLY the
readout (phase A -- no injection, no generation) on the single-mode training
family, then read the held-out families and score what the physics model would
answer from the parameters each readout produces. That number -- acc@0.05 of
FNO(predicted params) against the true value -- is the ceiling any channel
could deliver, so it predicts the rollout without running one.

  python eval/readout_transfer_probe.py --steps 800 --batch 8

Both readouts see the same data, steps and seed.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.fno import convert_from_torch  # noqa: E402
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from psilm.mlx.multimode import PsiLMMLXMulti, build_ic_multi_mlx, make_bridges_multi  # noqa: E402
from psilm.mlx.multimode_span import (  # noqa: E402
    PsiLMMLXMultiSpan, batch_extras, make_bridges_multi_span)
from psilm.mlx.staged import MlxStream  # noqa: E402
from psilm.stage2.qa2 import QA2Builder, make_batch as torch_make_batch  # noqa: E402

FAMILIES = ("val_iid", "val_combo", "val_amp")
TOL = 0.05


def with_second_term(item, rng, amp_max=0.0):
    """As with_zero_term, but the distractor carries a small NONZERO amplitude
    drawn from U(0.02, amp_max), with the answer recomputed by the spectral
    solver. A distractor that is always exactly 0.0 teaches 'a second term
    contributes nothing', which is the opposite of what the combination family
    asks; a small nonzero one teaches 'read whatever is written there' while
    leaving the held-out family's joint amplitude range (0.3-0.7 with 0.5-1.0)
    unseen."""
    present = {m for m, _, _ in item["modes"]}
    missing = [m for m in (1, 2) if m not in present]
    if not missing:
        return item
    from psilm.physics.burgers import initial_condition_multi, solve
    from psilm.stage2.qa import fourier_interp
    a = round(rng.uniform(0.02, amp_max), 2)
    modes = sorted(item["modes"] + [[missing[0], a, round(rng.uniform(0, 6.28), 2)]],
                   key=lambda t: t[0])
    field = solve(initial_condition_multi([tuple(m) for m in modes]))
    return {**item, "modes": modes, "u": round(float(fourier_interp(field, item["x0"])), 4)}


def with_zero_term(item, rng):
    """The same physics, written as two terms: the absent mode is added at
    amplitude 0.0. u(x,0) is unchanged (a zero-amplitude sinusoid contributes
    nothing), so the IC distribution and every answer stay exactly as they
    were -- what changes is that the prompt now HAS two terms, which is the
    context a single-mode training set never shows the readout. The held-out
    combination family, where both amplitudes are in their ranges, is still
    held out."""
    present = {m for m, _, _ in item["modes"]}
    missing = [m for m in (1, 2) if m not in present]
    if not missing:
        return item
    modes = item["modes"] + [[missing[0], 0.0, round(rng.uniform(0, 6.28), 2)]]
    return {**item, "modes": sorted(modes, key=lambda t: t[0])}


def to_batch(builder, items, span):
    tb = torch_make_batch(builder, items, "cpu")
    b = {"p_ids": mx.array(tb["p_ids"].numpy()),
         "p_labels": mx.array(tb["p_labels"].numpy()),
         "p_attn": mx.array(tb["p_attn"].numpy().astype(np.int32)),
         "prompt_mask": mx.array(tb["prompt_mask"].numpy()),
         "params": mx.array(tb["params"].numpy()),
         "param_mask": mx.array(tb["param_mask"].numpy()),
         "u_true": mx.array(tb["u_true"].numpy()),
         "x0": mx.array(tb["x0"].numpy()),
         "x0_bins": mx.array((tb["x0"].numpy() * 100).round().astype(np.int32).clip(0, 99)),
         "x0_span": mx.array(tb["x0_span"].numpy().astype(np.int32))}
    if span:
        b.update(batch_extras(builder, items))
        b["x0_span"] = b["spans"]
    return b


def fourier_interp_batch(field, x0):
    """Value of each periodic 128-point field at its x0, by Fourier series."""
    f = np.fft.rfft(np.array(field), axis=-1) / field.shape[-1]
    k = np.arange(f.shape[-1])[None, :]
    ang = 2 * math.pi * k * np.array(x0)[:, None]
    w = np.where(k > 0, 2.0, 1.0)
    return (w * (f.real * np.cos(ang) - f.imag * np.sin(ang))).sum(axis=-1)


def read_family(psi, builder, items, span, batch=8):
    """Teacher-forced readout quality on one family: the answer the physics
    model would give from the predicted parameters, plus per-quantity error so
    a deficit can be attributed to amplitude, phase or the x0 pointer."""
    a_err, p_err, x_err, hit, n = [], [], [], 0, 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b = to_batch(builder, chunk, span)
        s = MlxStream(psi.model, b["p_ids"], b["p_attn"])
        s.run(0, psi.l_fwd)
        params_hat, x0_hat, _, _ = psi.phi.fwd(s.hidden, b["prompt_mask"], b["x0_span"])
        u = psi.fno(build_ic_multi_mlx(params_hat))
        mx.eval(u)
        pred = fourier_interp_batch(u, [it["x0"] for it in chunk])
        true = np.array([it["u"] for it in chunk])
        hit += int((np.abs(pred - true) <= TOL).sum())
        n += len(chunk)
        p = np.array(params_hat)
        xh = np.array(x0_hat)
        for j, it in enumerate(chunk):
            d = abs(xh[j] - it["x0"])
            x_err.append(min(d, 1 - d))
            for m, a, phi in it["modes"]:
                a_err.append(abs(float(p[j, 3 * m - 3]) - a))
                phi_hat = math.atan2(float(p[j, 3 * m - 2]), float(p[j, 3 * m - 1]))
                dp = abs((phi_hat - phi + math.pi) % (2 * math.pi) - math.pi)
                p_err.append(dp)
    return {"acc": round(hit / n, 3), "amp_mae": round(float(np.mean(a_err)), 4),
            "phase_mae": round(float(np.mean(p_err)), 4),
            "x0_mae": round(float(np.mean(x_err)), 5)}


def run(kind, args, model, tok, hf_tok, fno, train_items, evals):
    span = kind == "span"
    bridges = (make_bridges_multi_span if span else make_bridges_multi)(
        model.args.hidden_size, gate_bias=0.0, inj_cap=0.2, channel="value",
        readout_norm=args.readout_norm)
    psi = (PsiLMMLXMultiSpan if span else PsiLMMLXMulti)(
        model, tok, fno, bridges, l_fwd=args.l_fwd, lam_x0=1.0)
    psi.readout_only = True
    psi.detach_x0 = True
    builder = QA2Builder(hf_tok)
    rng = random.Random(args.seed)

    if args.readout_norm == "dim":
        cb = to_batch(builder, rng.sample(train_items, 32), span)
        s = MlxStream(model, cb["p_ids"], cb["p_attn"])
        s.run(0, psi.l_fwd)
        bridges.fwd.calibrate_readout(s.hidden, cb["p_attn"].astype(mx.bool_))

    opt = optim.AdamW(learning_rate=args.lr, bias_correction=True)
    state = {}

    def lf(p):
        bridges.update(p)
        return psi.loss_fn(state["batch"])[0]

    t0 = time.time()
    for step in range(args.steps):
        sample = rng.sample(train_items, args.batch)
        if args.aug_zero_frac:
            aug = (with_zero_term if args.aug_amp_max <= 0 else
                   lambda it, r: with_second_term(it, r, args.aug_amp_max))
            sample = [aug(it, rng) if rng.random() < args.aug_zero_frac else it
                      for it in sample]
        state["batch"] = to_batch(builder, sample, span)
        loss, grads = mx.value_and_grad(lf)(bridges.trainable_parameters())
        opt.update(bridges, grads)
        mx.eval(bridges.parameters(), opt.state, loss)
        if (step + 1) % 200 == 0:
            print(f"  [{kind}] step {step+1}/{args.steps} loss {float(loss):.3f} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)
    n_train = sum(v.size for _, v in tree_flatten(bridges.trainable_parameters()))
    return {"params": int(n_train),
            **{fam: read_family(psi, builder, items, span) for fam, items in evals.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--readout-norm", default="dim", choices=["rms", "dim"])
    ap.add_argument("--l-fwd", type=int, default=None,
                    help="readout layer; earlier layers mix less context across terms, "
                         "which is what the combination family is sensitive to")
    ap.add_argument("--readouts", default="pooled,span")
    ap.add_argument("--aug-amp-max", type=float, default=0.0,
                    help="distractor amplitude ceiling: 0 writes the absent mode at exactly "
                         "0.0, >0 draws U(0.02, this) and recomputes the answer with the "
                         "solver")
    ap.add_argument("--aug-zero-frac", type=float, default=0.0,
                    help="fraction of training prompts rewritten with the absent mode at "
                         "amplitude 0.0: same physics, same answers, but the readout sees "
                         "two-term prompts. Covers the CONTEXT of a combination without "
                         "covering its amplitudes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/readout_transfer/probe.json")
    args = ap.parse_args()

    model, _, tok = load_backbone_any(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    fno = convert_from_torch("results/stage2b/fno.pt")
    train_items = json.loads(Path("data/stage2b_qa_train.json").read_text())
    evals = {f: json.loads(Path(f"data/stage2b_qa_{f}.json").read_text())[: args.n_eval]
             for f in FAMILIES}

    out = {"model": args.model, "steps": args.steps, "batch": args.batch,
           "readout_norm": args.readout_norm, "n_eval": args.n_eval,
           "l_fwd": args.l_fwd, "aug_zero_frac": args.aug_zero_frac,
           "aug_amp_max": args.aug_amp_max, "readouts": {}}
    for kind in args.readouts.split(","):
        print(f"== {kind} readout", flush=True)
        out["readouts"][kind] = run(kind, args, model, tok, hf_tok, fno, train_items, evals)
        print(f"   {json.dumps(out['readouts'][kind])}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\nteacher-forced readout -> physics answer, acc@0.05 (amplitude MAE):")
    for kind, r in out["readouts"].items():
        print(f"  {kind:7s} " + "   ".join(
            f"{f.replace('val_', ''):6s} {r[f]['acc']:.3f} (a {r[f]['amp_mae']:.3f} "
            f"phi {r[f]['phase_mae']:.3f} x0 {r[f]['x0_mae']:.4f})"
            for f in FAMILIES))


if __name__ == "__main__":
    main()
