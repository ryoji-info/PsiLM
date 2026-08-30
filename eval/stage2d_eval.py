"""Final Stage-2d evaluation, four arms on held-out 2D Fisher-KPP QA."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.physics.dpot_wrapper import DPOTPhysics  # noqa: E402
from psilm.stage2d.bridges2d import PsiBridges2D  # noqa: E402
from psilm.stage2d.model2d import PsiLM2D  # noqa: E402
from psilm.stage2d.qa2d import QUESTION, QA2DBuilder  # noqa: E402
from psilm.stage2.qa import SYSTEM  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 0.05


def parse_value(text):
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def chat_generate(model, tok, device, user, max_new=80):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(out, list):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    ids = torch.tensor([out], device=device)
    gen = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(args.device).eval()
    phys = DPOTPhysics(device=args.device)
    phys.net.load_state_dict(torch.load("results/stage2d/dpot_ft.pt", map_location=args.device))
    phys.eval()
    bridges = PsiBridges2D().to(args.device)
    state = torch.load("results/stage2d/bridges.pt", map_location=args.device, weights_only=False)
    bridges.load_state_dict(state["bridges"])
    psi = PsiLM2D(model, tok, phys, bridges)
    builder = QA2DBuilder(tok)

    items = json.loads(Path("data/stage2d_qa_val.json").read_text())[: args.n]
    agg = {k: [0, []] for k in ("baseline", "oracle", "psilm", "zero")}
    rows = []
    for i, item in enumerate(items):
        q = QUESTION.format(**{k: item[k] for k in ("a", "cx", "cy", "w", "x0", "y0")})
        true = item["u"]
        nudge = "\nReply with only the number."
        preds = {
            "baseline": parse_value(chat_generate(model, tok, args.device, q + nudge)),
            "oracle": parse_value(chat_generate(
                model, tok, args.device,
                q + f"\n\nA trusted solver reports: u({item['x0']}, {item['y0']}) = {true:.2f}." + nudge)),
            "psilm": parse_value(psi.generate(builder, item)),
            "zero": 0.0,
        }
        row = {"item": item}
        for k, p in preds.items():
            ok = p is not None and abs(p - true) <= TOL
            agg[k][0] += ok
            if p is not None:
                agg[k][1].append(abs(p - true))
            row[k] = {"pred": p, "ok": bool(ok)}
        rows.append(row)
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)

    n = len(items)
    summary = {"n": n, "step": state["step"], "tolerance": TOL}
    for k, (c, errs) in agg.items():
        summary[k] = {"acc": round(c / n, 4),
                      "mae": round(sum(errs) / len(errs), 4) if errs else None}
    Path("results/stage2d/final_eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
