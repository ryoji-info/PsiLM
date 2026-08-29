"""Final Stage-2 evaluation, three arms on held-out QA:

  baseline : frozen LLM alone (plain chat generation)
  oracle   : frozen LLM + the true value stated in text (tool-loop ceiling)
  psilm    : the coupled system (latent bridges, no text at the interface)

Also reports the degenerate always-0.00 strategy for calibration.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.physics.fno import FNO1d  # noqa: E402
from psilm.stage2.bridges import PsiBridges  # noqa: E402
from psilm.stage2.model import PsiLM  # noqa: E402
from psilm.stage2.qa import QUESTION, SYSTEM, QABuilder  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 0.05


def parse_value(text):
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def chat_generate(model, tok, device, user, max_new=40):
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


def score(pred, true):
    return pred is not None and abs(pred - true) <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ckpt", default="results/stage2/bridges.pt")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(args.device).eval()
    fno = FNO1d().to(args.device)
    fno.load_state_dict(torch.load("results/stage2/fno.pt", map_location=args.device))
    fno.eval()
    bridges = PsiBridges().to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    bridges.load_state_dict(state["bridges"])
    psi = PsiLM(model, tok, fno, bridges)
    builder = QABuilder(tok)

    items = json.loads(Path("data/stage2_qa_val.json").read_text())[: args.n]
    agg = {k: [0, []] for k in ("baseline", "oracle", "psilm", "zero")}
    rows = []
    for i, item in enumerate(items):
        q = QUESTION.format(a=item["a"], phi=item["phi"], x0=item["x0"])
        true = item["u"]
        preds = {}
        preds["baseline"] = parse_value(chat_generate(model, tok, args.device, q))
        oracle_q = q + f"\n\nA trusted solver reports: u({item['x0']}) = {true:.2f}."
        preds["oracle"] = parse_value(chat_generate(model, tok, args.device, oracle_q))
        preds["psilm"] = parse_value(psi.generate(builder, item))
        preds["zero"] = 0.0
        row = {"item": item}
        for k, p in preds.items():
            ok = score(p, true)
            agg[k][0] += ok
            if p is not None:
                agg[k][1].append(abs(p - true))
            row[k] = {"pred": p, "ok": bool(ok)}
        rows.append(row)
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    n = len(items)
    summary = {"n": n, "step": state["step"], "tolerance": TOL}
    for k, (c, errs) in agg.items():
        summary[k] = {"acc": round(c / n, 4),
                      "mae": round(sum(errs) / len(errs), 4) if errs else None}
    Path("results/stage2/final_eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
