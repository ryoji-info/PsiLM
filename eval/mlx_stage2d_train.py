"""Train the Stage-2d (2D Fisher-KPP, DPOT-Tiny) bridges over MLX. Hybrid:
language side MLX (4-bit-capable), physics side torch (MPS). Chunked,
resumable; mirror of eval/mlx_stage2_train.py.

Usage (smoke, 0.5B):
  python eval/mlx_stage2d_train.py --steps 6 --readout-only 3 --batch 2 --fresh \
      --channel value --readout-norm dim --calib-n 4 --tag _mlx2d

Gemma 4 12B (the 1D recipe, results/stage2_gemma12b/bridges.npz.meta):
  python eval/mlx_stage2d_train.py --model mlx-community/gemma-4-12B-it-4bit \
      --hf-tokenizer mlx-community/gemma-4-12B-it-4bit --tag _gemma12b_2d \
      --steps 500 --batch 4 --lr 1e-4 --readout-only 2000 --clip module \
      --channel value --readout-norm dim --inj-cap 0.2 --gate-bias 0.0 \
      --eval-n 48 --fresh            (then the same command without --fresh, per chunk;
                                      add --noharm-data data/noharm_gemma_all.json
                                      once phase B is under way)
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.optimizers import clip_grad_norm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.bridges2d import PsiBridges2DMLX  # noqa: E402
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from psilm.mlx.model2d import PsiLM2DMLX  # noqa: E402
from psilm.mlx.physics2d import TorchPhysics2D  # noqa: E402
from psilm.mlx.staged import MlxStream  # noqa: E402
from psilm.stage2d.qa2d import QUANTITIES, QA2DBuilder, make_batch as torch_make_batch  # noqa: E402

TRAIN = "data/stage2d_qa_train.json"
VAL = "data/stage2d_qa_val.json"


def to_mlx_batch(tb):
    metas = tb["metas"]
    return {
        "p_ids": mx.array(tb["p_ids"].numpy()),
        "p_labels": mx.array(tb["p_labels"].numpy()),
        "p_attn": mx.array(tb["p_attn"].numpy().astype(np.int32)),
        "prompt_mask": mx.array(tb["prompt_mask"].numpy()),
        "spans": mx.array(tb["spans"].numpy().astype(np.int32)),          # (B, 6, 2)
        "qbins": mx.array(tb["qbins"].numpy().astype(np.int32)),          # (B, 6)
        "params": mx.array(np.array([[m["a"], m["cx"], m["cy"], m["w"]] for m in metas],
                                    dtype=np.float32)),
        "x0": mx.array(np.array([m["x0"] for m in metas], dtype=np.float32)),
        "y0": mx.array(np.array([m["y0"] for m in metas], dtype=np.float32)),
        "u_true": mx.array(tb["u_true"].numpy()),
    }


def parse_answer(text):
    """The trained reply is 'u at ({x0}, {y0}) equals {u}.'; only the number after
    'equals' counts, so a reply that stops at the coordinate echo is a miss."""
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def make_noharm_batch(items, pad_id):
    """Right-padded (prompt + backbone continuation); labels only on the
    continuation; every quantity span = the whole prompt (the guard-rail's
    non-physics convention)."""
    seqs = [it["prompt_ids"] + it["target_ids"] for it in items]
    plens = [len(it["prompt_ids"]) for it in items]
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
    spans = np.array([[[0, pl]] * len(QUANTITIES) for pl in plens], dtype=np.int32)
    return {"p_ids": mx.array(I), "p_attn": mx.array(A), "p_labels": mx.array(LB),
            "prompt_mask": mx.array(PM), "spans": mx.array(spans), "noharm": True}


def rollout_eval(psi, builder, items, n=12):
    correct, errs = 0, []
    for item in items[:n]:
        pred = parse_answer(psi.generate(builder, item))
        ok = pred is not None and abs(pred - item["u"]) <= 0.05
        correct += ok
        if pred is not None:
            errs.append(abs(pred - item["u"]))
    return correct / n, (sum(errs) / len(errs) if errs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF tokenizer for the QA builder (matches the base model)")
    ap.add_argument("--tag", default="_mlx2d")
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    ap.add_argument("--l-rev", type=int, default=None)
    ap.add_argument("--lam-cls", type=float, default=1.0,
                    help="weight of the sum of the six readout cross-entropies")
    ap.add_argument("--detach-x0", action="store_true",
                    help="accepted for parity; the 2D lookup coordinates are always detached")
    ap.add_argument("--readout-only", type=int, default=0,
                    help="phase A: first N global steps train the six readouts through "
                         "layers 0..l_fwd only (no physics, no injection, no answer loss)")
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--readout-norm", default="rms", choices=["rms", "dim"])
    ap.add_argument("--calib-n", type=int, default=32)
    ap.add_argument("--inj-cap", type=float, default=None)
    ap.add_argument("--channel", default="value", choices=["value"],
                    help="physics->language channel: the looked-up u(x0, y0) as value tokens")
    ap.add_argument("--phys-device", default="mps", help="torch device for DPOT (mps|cpu)")
    ap.add_argument("--noharm-data", default=None)
    ap.add_argument("--noharm-every", type=int, default=2)
    ap.add_argument("--noharm-gate-only", type=int, default=1)
    ap.add_argument("--lam-gate", type=float, default=1.0)
    ap.add_argument("--reinit-channel", action="store_true")
    ap.add_argument("--clip", default="global", choices=["global", "module"])
    args = ap.parse_args()

    ckpt = Path(f"results/stage2d{args.tag}/bridges.npz")
    log = Path(f"results/stage2d{args.tag}/train_log.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    model, stock, tok = load_backbone_any(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    phys = TorchPhysics2D(device=args.phys_device)
    bridges = PsiBridges2DMLX(d_model=model.args.hidden_size, gate_bias=args.gate_bias,
                              inj_cap=args.inj_cap, channel=args.channel,
                              readout_norm=args.readout_norm)
    opt = optim.AdamW(learning_rate=args.lr, bias_correction=True)
    opt_path = ckpt.parent / "opt.npz"

    global_step = 0
    if ckpt.exists() and not args.fresh:
        ref = bridges.fwd.h2[0].weight
        bridges.load_weights(str(ckpt), strict=not args.reinit_channel)
        assert float(mx.abs(bridges.fwd.h2[0].weight - ref).max()) > 0, "readout did not load"
        meta = json.loads(Path(str(ckpt) + ".meta").read_text())
        global_step = meta["step"]
        if args.reinit_channel:
            bridges.reinit_channel(gate_bias=args.gate_bias, inj_cap=args.inj_cap)
            mx.eval(bridges.parameters())
            print(f"resumed at step {global_step}: readouts kept, channel re-initialized "
                  f"(gate_bias {args.gate_bias}, inj_cap {args.inj_cap}); optimizer fresh")
        elif opt_path.exists():
            opt.state = tree_unflatten(list(mx.load(str(opt_path)).items()))
            opt.state["learning_rate"] = mx.array(args.lr)
            mx.eval(opt.state)
            print(f"resumed at step {global_step} (optimizer state restored, "
                  f"opt step {int(opt.state['step'].item())}, lr {args.lr})")
        else:
            print(f"resumed at step {global_step} (optimizer state fresh)")
        prev = meta.get("args", {})
        for k in ("channel", "inj_cap", "lam_cls", "clip", "l_rev", "gate_bias", "readout_norm"):
            if k in prev and prev[k] != getattr(args, k):
                print(f"[WARN] --{k.replace('_', '-')}={getattr(args, k)} differs from the checkpoint's {prev[k]}")

    psi = PsiLM2DMLX(model, tok, phys, bridges, l_rev=args.l_rev, lam_cls=args.lam_cls)
    builder = QA2DBuilder(hf_tok)
    train_items = json.loads(Path(TRAIN).read_text())
    if args.readout_norm == "dim" and (args.fresh or not ckpt.exists()):
        cbatch = to_mlx_batch(torch_make_batch(builder, random.Random(7).sample(train_items, args.calib_n), "cpu"))
        cs = MlxStream(model, cbatch["p_ids"], cbatch["p_attn"])
        cs.run(0, psi.l_fwd)
        bridges.fwd.calibrate_readout(cs.hidden, cbatch["prompt_mask"])
        print(f"readout calibrated on {args.calib_n} prompts at layer {psi.l_fwd}: "
              f"sigma max {float(bridges.fwd.dim_sigma.max()):.1f} median "
              f"{float(mx.sort(bridges.fwd.dim_sigma)[bridges.fwd.dim_sigma.shape[0]//2]):.3f}")
    noharm_items = json.loads(Path(args.noharm_data).read_text()) if args.noharm_data else None
    if noharm_items:
        assert args.noharm_every >= 2, "--noharm-every 1 would starve the physics arm"
        print(f"no-harm arm: {len(noharm_items)} prompts, every {args.noharm_every}th step, "
              f"gate-only updates: {bool(args.noharm_gate_only)}, lam_gate {args.lam_gate}")
    psi.lam_gate = args.lam_gate
    val_items = json.loads(Path(VAL).read_text())
    n_params = sum(v.size for _, v in tree_flatten(bridges.parameters()))
    print(f"bridges: {n_params/1e6:.2f}M | backbone: {args.model} | physics: DPOT-Tiny on "
          f"{phys.device} | coupling {psi.l_fwd}/{psi.l_rev} of {psi.n_layers}")

    def wrapped(bridges_, batch):
        psi.phi = bridges_
        return psi.loss_fn(batch)

    loss_and_grad = nn.value_and_grad(bridges, wrapped)

    t0 = time.time()
    run = {"B": {}, "N": {}}
    def _acc(phase, **kv):
        d = run[phase]
        for k, v in kv.items():
            d[k] = d.get(k, 0.0) + v
        d["_n"] = d.get("_n", 0) + 1
    for i in range(args.steps):
        rng = random.Random(21_000_000 + global_step)
        psi.readout_only = global_step < args.readout_only
        if noharm_items and not psi.readout_only and (global_step % args.noharm_every == args.noharm_every - 1):
            pad = hf_tok.pad_token_id or hf_tok.eos_token_id
            batch = make_noharm_batch(rng.sample(noharm_items, args.batch), pad)
        else:
            batch = to_mlx_batch(torch_make_batch(builder, rng.sample(train_items, args.batch), "cpu"))
        (loss, aux), grads = loss_and_grad(bridges, batch)
        if batch.get("noharm") and args.noharm_gate_only:
            keep = {"g1": grads["inject"]["g1"], "g2": grads["inject"]["g2"]}
            grads = tree_map(lambda g: mx.zeros_like(g), grads)
            grads["inject"]["g1"], grads["inject"]["g2"] = keep["g1"], keep["g2"]
        if args.clip == "module":
            grads = {k: clip_grad_norm(g, 1.0)[0] for k, g in grads.items()}
        else:
            grads, _ = clip_grad_norm(grads, 1.0)
        opt.update(bridges, grads)
        mx.eval(bridges.parameters(), opt.state)
        global_step += 1
        ph = "A" if psi.readout_only else ("N" if batch.get("noharm") else "B")
        if ph == "B":
            _acc("B", loss_ans=aux["loss_ans"].item(), gate_ans=aux["gate_ans"].item(),
                 inj_ratio=aux["inj_ratio_ans"].item(), u_err=aux["u_err"].item())
        elif ph == "N":
            _acc("N", ce=aux["loss_ans"].item(), gate_ans=aux["gate_ans"].item(),
                 gate_all=aux["gate"].item(), inj_ratio=aux["inj_ratio_ans"].item())
        if global_step % 25 == 0 or global_step == args.readout_only or i == args.steps - 1:
            rec = {"step": global_step, "loss": round(loss.item(), 4),
                   "loss_ans": round(aux["loss_ans"].item(), 4),
                   "loss_cls": round(aux["loss_cls"].item(), 4),
                   "u_err": round(aux["u_err"].item(), 5),
                   "u_err_oracle": round(aux["u_err_oracle"].item(), 5),
                   "pos_err": round(aux["pos_err"].item(), 4),
                   "gate": round(aux["gate"].item(), 4),
                   "gate_ans": round(aux["gate_ans"].item(), 4),
                   "inj_ratio_ans": round(aux["inj_ratio_ans"].item(), 4),
                   "exact": {q: round(float(v), 3) for q, v in zip(QUANTITIES, np.array(aux["exact"]))},
                   "phase": ph,
                   "sec_per_step": round((time.time() - t0) / (i + 1), 2)}
            if "ces" in aux:
                rec["ces"] = {q: round(float(v), 3) for q, v in zip(QUANTITIES, np.array(aux["ces"]))}
            for pk, d in run.items():
                n = d.get("_n", 0)
                if n:
                    for k, v in d.items():
                        if k != "_n":
                            rec[f"{pk}_{k}"] = round(v / n, 4)
                    rec[f"{pk}_n"] = n
            run = {"B": {}, "N": {}}
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

    if global_step <= args.readout_only:
        acc, mae = None, None
    else:
        psi.readout_only = False
        acc, mae = rollout_eval(psi, builder, val_items, n=args.eval_n)
    bridges.save_weights(str(ckpt))
    mx.savez(str(opt_path), **dict(tree_flatten(opt.state)))
    Path(str(ckpt) + ".meta").write_text(json.dumps({
        "step": global_step, "model": args.model, "l_rev": psi.l_rev, "l_fwd": psi.l_fwd,
        "task": "stage2d", "quantities": list(QUANTITIES), "args": vars(args)}))
    with log.open("a") as f:
        f.write(json.dumps({"step": global_step, "eval_acc": acc, "eval_mae": mae}) + "\n")
    print(f"CHUNK DONE step={global_step} acc@0.05={acc if acc is None else round(acc, 3)} mae={mae}")


if __name__ == "__main__":
    main()
