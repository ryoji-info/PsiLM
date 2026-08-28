"""Train the Bicameral interface (Stage 1). Resumable, chunked.

Usage:
  python eval/stage1_train.py --steps 1000            # one chunk, auto-resume
  python eval/stage1_train.py --steps 1000 --fresh    # start over

Both Qwen streams stay frozen; only the interface trains. Data is generated
procedurally (multiplication, log-uniform operands), seeded per global step so
resumed runs continue with fresh examples. Each chunk ends with a rollout
evaluation (tool recall + answer accuracy) and a checkpoint.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.bicameral.data import ExampleBuilder, make_batch, sample_operands  # noqa: E402
from psilm.bicameral.interface import BicameralInterface  # noqa: E402
from psilm.bicameral.model import Bicameral  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = Path("results/stage1/interface.pt")
LOG = Path("results/stage1/train_log.jsonl")


def rollout_eval(bic, builder, n=8, seed=12345, lo=10**3, hi=10**5):
    rng = random.Random(seed)
    tool_ok = acc = 0
    samples = []
    for _ in range(n):
        a, b = sample_operands(rng, lo, hi)
        answer, aux_text = bic.generate(builder, a, b)
        called = f"calc({a}*{b})" in aux_text
        correct = str(a * b) in answer
        tool_ok += called
        acc += correct
        samples.append({"a": a, "b": b, "answer": answer[-80:], "aux": aux_text[:80],
                        "tool_ok": called, "correct": correct})
    return tool_ok / n, acc / n, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--hi", type=int, default=10**5)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fp32", action="store_true", help="fp32 backbone (default fp16)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float32 if args.fp32 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(args.device).eval()
    phi = BicameralInterface(d_model=896, kind="pull_standard", f_hidden=1024, f_layers=2).to(args.device)
    opt = torch.optim.AdamW(phi.parameters(), lr=args.lr)

    global_step = 0
    if CKPT.exists() and not args.fresh:
        state = torch.load(CKPT, map_location=args.device, weights_only=False)
        phi.load_state_dict(state["phi"])
        opt.load_state_dict(state["opt"])
        global_step = state["step"]
        print(f"resumed from {CKPT} at step {global_step}")

    bic = Bicameral(model, tok, phi)
    builder = ExampleBuilder(tok)
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i in range(args.steps):
        rng = random.Random(1_000_000 + global_step)  # per-step seed: resume-safe
        batch = make_batch(builder, rng, args.batch, args.device, lo=args.lo, hi=args.hi)
        out = bic.train_forward(batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        global_step += 1

        if global_step % 25 == 0:
            rec = {
                "step": global_step,
                "samples": global_step * args.batch,
                "loss_p": round(out["loss_p"].item(), 4),
                "loss_a": round(out["loss_a"].item(), 4),
                "gate_fwd": round(out["gate_fwd"].item(), 4),
                "gate_rev": round(out["gate_rev"].item(), 4),
                "pert_fwd": round(out["pert_fwd"].item(), 4),
                "pert_rev": round(out["pert_rev"].item(), 4),
                "sec_per_step": round((time.time() - t0) / (i + 1), 2),
            }
            with LOG.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if global_step % 200 == 0 and args.device == "mps":
            torch.mps.empty_cache()

    # end-of-chunk rollout evaluation + checkpoint
    tool_recall, acc, samples = rollout_eval(bic, builder, n=8, lo=args.lo, hi=args.hi)
    torch.save({"phi": phi.state_dict(), "opt": opt.state_dict(), "step": global_step}, CKPT)
    rec = {"step": global_step, "eval_tool_recall": tool_recall, "eval_acc": acc}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        for s in samples:
            f.write(json.dumps({"step": global_step, "sample": s}) + "\n")
    print(f"CHUNK DONE step={global_step} tool_recall={tool_recall:.2f} acc={acc:.2f}")
    for s in samples[:4]:
        print("  ", s)


if __name__ == "__main__":
    main()
