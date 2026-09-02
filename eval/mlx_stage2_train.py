"""Train PsiLM bridges over MLX (4-bit-capable backbones). Chunked, resumable.

Usage:
  python eval/mlx_stage2_train.py --steps 1000 --fresh \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --tag _mlx0.5b4
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
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.optimizers import clip_grad_norm  # noqa: E402
import mlx_lm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.bridges import PsiBridgesMLX  # noqa: E402
from psilm.mlx.fno import convert_from_torch  # noqa: E402
from psilm.mlx.model import PsiLMMLX  # noqa: E402
from psilm.stage2.qa import QABuilder, make_batch as torch_make_batch  # noqa: E402


def to_mlx_batch(tb):
    return {
        "p_ids": mx.array(tb["p_ids"].numpy()),
        "p_labels": mx.array(tb["p_labels"].numpy()),
        "p_attn": mx.array(tb["p_attn"].numpy().astype(np.int32)),
        "prompt_mask": mx.array(tb["prompt_mask"].numpy()),
        "params": mx.array(tb["params"].numpy()),
        "u_true": mx.array(tb["u_true"].numpy()),
        "x0": mx.array(tb["x0"].numpy()),
        "x0_bins": mx.array((tb["x0"].numpy() * 100).round().astype(np.int32).clip(0, 99)),
        "x0_span": mx.array(tb["x0_span"].numpy().astype(np.int32)),
    }


def parse_value(text):
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def parse_answer(text):
    """The trained reply is 'u at x = {x0} equals {u}.'; only the number after
    'equals' counts, so a reply that stops at the x0 echo scores as a miss."""
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    ap.add_argument("--l-rev", type=int, default=None)
    ap.add_argument("--lam-x0", type=float, default=0.3)
    ap.add_argument("--detach-x0", action="store_true",
                    help="stop_gradient on the pointer fed to the reverse bridge")
    ap.add_argument("--readout-only", type=int, default=0,
                    help="phase A: first N global steps train fwd + lookup on the true x0 "
                         "through layers 0..l_fwd only (no injection, no answer loss)")
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--inj-cap", type=float, default=None,
                    help="cap the injection RMS at this fraction of the stream RMS (pre-gate)")
    ap.add_argument("--reinit-channel", action="store_true",
                    help="on resume: keep the trained readouts/lookup, re-initialize the reverse "
                         "token heads and the gated injection (fresh optimizer)")
    ap.add_argument("--clip", default="global", choices=["global", "module"],
                    help="module: clip each bridge (fwd/rev/inject) at 1.0 separately, "
                         "so a large injection gradient cannot starve the readouts "
                         "(the regime the 8B readout probe converged in)")
    args = ap.parse_args()

    ckpt = Path(f"results/stage2{args.tag}/bridges.npz")
    log = Path(f"results/stage2{args.tag}/train_log.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    model, tok = mlx_lm.load(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    fno = convert_from_torch("results/stage2/fno.pt")
    d_model = model.model.embed_tokens.weight.shape[-1] if not hasattr(
        model.model.embed_tokens, "scales") else model.args.hidden_size
    bridges = PsiBridgesMLX(d_model=model.args.hidden_size, gate_bias=args.gate_bias,
                            inj_cap=args.inj_cap)
    # bias correction matters: MLX defaults to none, so a fresh AdamW takes
    # 3-6x steps for its first ~15 updates. State is persisted across chunks
    # (the torch trainer always did; the 8B v5 gate closed at chunk boundaries).
    opt = optim.AdamW(learning_rate=args.lr, bias_correction=True)
    opt_path = ckpt.parent / "opt.npz"

    global_step = 0
    if ckpt.exists() and not args.fresh:
        bridges.load_weights(str(ckpt))
        meta = json.loads(Path(str(ckpt) + ".meta").read_text())
        global_step = meta["step"]
        if args.reinit_channel:
            bridges.reinit_channel(gate_bias=args.gate_bias, inj_cap=args.inj_cap)
            mx.eval(bridges.parameters())
            print(f"resumed at step {global_step}: readouts/lookup kept, channel re-initialized "
                  f"(gate_bias {args.gate_bias}, inj_cap {args.inj_cap}); optimizer fresh")
        elif opt_path.exists():
            opt.state = tree_unflatten(list(mx.load(str(opt_path)).items()))
            mx.eval(opt.state)
            print(f"resumed at step {global_step} (optimizer state restored, "
                  f"opt step {int(opt.state['step'].item())})")
        else:
            print(f"resumed at step {global_step} (optimizer state fresh)")

    psi = PsiLMMLX(model, tok, fno, bridges, l_rev=args.l_rev, lam_x0=args.lam_x0)
    psi.detach_x0 = args.detach_x0
    builder = QABuilder(hf_tok)
    train_items = json.loads(Path("data/stage2_qa_train.json").read_text())
    val_items = json.loads(Path("data/stage2_qa_val.json").read_text())
    n_params = sum(v.size for _, v in tree_flatten(bridges.parameters()))
    print(f"bridges: {n_params/1e6:.2f}M | backbone: {args.model} | "
          f"coupling {psi.l_fwd}/{psi.l_rev} of {psi.n_layers}")

    def wrapped(bridges_, batch):
        psi.phi = bridges_
        return psi.loss_fn(batch)

    loss_and_grad = nn.value_and_grad(bridges, wrapped)

    t0 = time.time()
    for i in range(args.steps):
        rng = random.Random(21_000_000 + global_step)
        batch = to_mlx_batch(torch_make_batch(builder, rng.sample(train_items, args.batch), "cpu"))
        psi.readout_only = global_step < args.readout_only
        (loss, aux), grads = loss_and_grad(bridges, batch)
        if args.clip == "module":
            grads = {k: clip_grad_norm(g, 1.0)[0] for k, g in grads.items()}
        else:
            grads, _ = clip_grad_norm(grads, 1.0)
        opt.update(bridges, grads)
        mx.eval(bridges.parameters(), opt.state)
        global_step += 1
        if global_step % 25 == 0:
            rec = {"step": global_step,
                   "loss_ans": round(aux[0].item(), 4),
                   "loss_param": round(aux[1].item(), 5),
                   "loss_x0": round(aux[2].item(), 4),
                   "loss_u": round(aux[3].item(), 5),
                   "x0_err": round(aux[4].item(), 4),
                   "gate": round(aux[5].item(), 4),
                   "loss_attn": round(aux[6].item(), 4),
                   "gate_ans": round(aux[7].item(), 4),
                   "inj_ratio_ans": round(aux[8].item(), 4),
                   "x0_exact": round(aux[9].item(), 4),
                   "phase": "A" if psi.readout_only else "B",
                   "sec_per_step": round((time.time() - t0) / (i + 1), 2)}
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

    if global_step <= args.readout_only:
        acc, mae = None, None       # still in phase A: no channel to evaluate
    else:
        psi.readout_only = False
        acc, mae = rollout_eval(psi, builder, val_items, n=args.eval_n)
    bridges.save_weights(str(ckpt))
    mx.savez(str(opt_path), **dict(tree_flatten(opt.state)))
    Path(str(ckpt) + ".meta").write_text(json.dumps({
        "step": global_step, "model": args.model, "l_rev": psi.l_rev, "l_fwd": psi.l_fwd,
        "args": vars(args)}))
    with log.open("a") as f:
        f.write(json.dumps({"step": global_step, "eval_acc": acc, "eval_mae": mae}) + "\n")
    print(f"CHUNK DONE step={global_step} acc@0.05={acc if acc is None else round(acc, 3)} mae={mae}")


if __name__ == "__main__":
    main()
