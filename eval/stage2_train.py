"""Train the PsiLM Stage-2 bridges. Chunked and resumable (Stage-1 pattern).

Usage:
  python eval/stage2_train.py --steps 1000            # one chunk, auto-resume
  python eval/stage2_train.py --steps 1000 --fresh
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.physics.fno import FNO1d  # noqa: E402
from psilm.stage2.bridges import PsiBridges  # noqa: E402
from psilm.stage2.model import PsiLM  # noqa: E402
from psilm.stage2.qa import QABuilder, make_batch  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = Path("results/stage2/bridges.pt")
LOG = Path("results/stage2/train_log.jsonl")


def parse_value(text):
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def rollout_eval(psi, builder, items, n=12):
    correct = 0
    errs = []
    samples = []
    for item in items[:n]:
        out = psi.generate(builder, item)
        pred = parse_value(out)
        ok = pred is not None and abs(pred - item["u"]) <= 0.05
        correct += ok
        if pred is not None:
            errs.append(abs(pred - item["u"]))
        samples.append({"item": item, "out": out[-60:], "ok": bool(ok)})
    mae = sum(errs) / len(errs) if errs else None
    return correct / n, mae, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(args.device).eval()
    fno = FNO1d().to(args.device)
    fno.load_state_dict(torch.load("results/stage2/fno.pt", map_location=args.device))
    fno.eval()
    bridges = PsiBridges().to(args.device)
    opt = torch.optim.AdamW(bridges.parameters(), lr=args.lr)

    global_step = 0
    if CKPT.exists() and not args.fresh:
        state = torch.load(CKPT, map_location=args.device, weights_only=False)
        bridges.load_state_dict(state["bridges"])
        opt.load_state_dict(state["opt"])
        global_step = state["step"]
        print(f"resumed at step {global_step}")

    psi = PsiLM(model, tok, fno, bridges)
    builder = QABuilder(tok)
    train_items = json.loads(Path("data/stage2_qa_train.json").read_text())
    val_items = json.loads(Path("data/stage2_qa_val.json").read_text())
    print(f"bridges: {bridges.n_params()/1e6:.2f}M params | train items: {len(train_items)}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i in range(args.steps):
        rng = random.Random(5_000_000 + global_step)
        batch = make_batch(builder, rng.sample(train_items, args.batch), args.device)
        out = psi.train_forward(batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(bridges.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        global_step += 1
        if global_step % 25 == 0:
            rec = {"step": global_step,
                   "loss_ans": round(out["loss_ans"].item(), 4),
                   "loss_param": round(out["loss_param"].item(), 5),
                   "gate": round(out["gate"].item(), 4),
                   "sec_per_step": round((time.time() - t0) / (i + 1), 2)}
            with LOG.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if global_step % 200 == 0 and args.device == "mps":
            torch.mps.empty_cache()

    acc, mae, samples = rollout_eval(psi, builder, val_items)
    torch.save({"bridges": bridges.state_dict(), "opt": opt.state_dict(),
                "step": global_step}, CKPT)
    rec = {"step": global_step, "eval_acc": acc, "eval_mae": mae}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        for s in samples:
            f.write(json.dumps({"step": global_step, "sample": s}) + "\n")
    print(f"CHUNK DONE step={global_step} acc@0.05={acc:.2f} mae={mae}")


if __name__ == "__main__":
    main()
