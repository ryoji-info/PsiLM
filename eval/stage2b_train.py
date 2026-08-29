"""Train Stage-2b bridges (multi-mode ICs). Chunked and resumable."""

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
from psilm.stage2.bridges import PsiBridges, build_ic_multi  # noqa: E402
from psilm.stage2.model import PsiLM  # noqa: E402
from psilm.stage2.qa2 import QA2Builder, make_batch  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = Path("results/stage2b/bridges.pt")      # overridden by --v2
LOG = Path("results/stage2b/train_log.jsonl")


def parse_value(text):
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def rollout_eval(psi, builder, items, n=12):
    correct, errs = 0, []
    for item in items[:n]:
        out = psi.generate(builder, item)
        pred = parse_value(out)
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
    ap.add_argument("--device", default="mps")
    ap.add_argument("--v2", action="store_true", help="per-mode binned forward bridge")
    args = ap.parse_args()

    global CKPT, LOG
    if args.v2:
        CKPT = Path("results/stage2b_v2/bridges.pt")
        LOG = Path("results/stage2b_v2/train_log.jsonl")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(args.device).eval()
    fno = FNO1d().to(args.device)
    fno.load_state_dict(torch.load("results/stage2b/fno.pt", map_location=args.device))
    fno.eval()
    bridges = PsiBridges(n_params=6, fwd_kind="per_mode" if args.v2 else "pooled").to(args.device)
    opt = torch.optim.AdamW(bridges.parameters(), lr=args.lr)

    global_step = 0
    if CKPT.exists() and not args.fresh:
        state = torch.load(CKPT, map_location=args.device, weights_only=False)
        bridges.load_state_dict(state["bridges"])
        opt.load_state_dict(state["opt"])
        global_step = state["step"]
        print(f"resumed at step {global_step}")

    psi = PsiLM(model, tok, fno, bridges, ic_fn=build_ic_multi)
    builder = QA2Builder(tok)
    train_items = json.loads(Path("data/stage2b_qa_train.json").read_text())
    val_items = json.loads(Path("data/stage2b_qa_val_iid.json").read_text())
    print(f"bridges: {bridges.n_params()/1e6:.2f}M | train items: {len(train_items)}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i in range(args.steps):
        rng = random.Random(9_000_000 + global_step)
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
                   "loss_x0": round(out["loss_x0"].item(), 4),
                   "loss_u": round(out["loss_u"].item(), 5),
                   "x0_err": round(out["x0_err"].item(), 4),
                   "gate": round(out["gate"].item(), 4),
                   "sec_per_step": round((time.time() - t0) / (i + 1), 2)}
            with LOG.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if global_step % 200 == 0 and args.device == "mps":
            torch.mps.empty_cache()

    acc, mae = rollout_eval(psi, builder, val_items)
    torch.save({"bridges": bridges.state_dict(), "opt": opt.state_dict(),
                "step": global_step}, CKPT)
    with LOG.open("a") as f:
        f.write(json.dumps({"step": global_step, "eval_acc": acc, "eval_mae": mae}) + "\n")
    print(f"CHUNK DONE step={global_step} acc@0.05={acc:.2f} mae={mae}")


if __name__ == "__main__":
    main()
