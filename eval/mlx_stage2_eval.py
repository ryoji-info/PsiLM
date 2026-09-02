"""Final Stage-2 evaluation on MLX, four arms on held-out QA (port of eval/stage2_eval.py):

  baseline : frozen LLM alone (plain chat generation, "Answer: <number>" protocol)
  oracle   : frozen LLM + the true value stated in text (tool-loop ceiling)
  psilm    : the coupled system (latent bridges, no text at the interface)
  zero     : degenerate always-0.00 strategy (calibration)

Usage:
  python eval/mlx_stage2_eval.py --model mlx-community/Qwen3-8B-4bit \
      --hf-tokenizer Qwen/Qwen3-8B --tag _mlx8b4 --l-rev 27 --n 60
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx_lm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.bridges import PsiBridgesMLX  # noqa: E402
from psilm.mlx.fno import convert_from_torch  # noqa: E402
from psilm.mlx.model import PsiLMMLX  # noqa: E402
from psilm.stage2.qa import QUESTION, SYSTEM, QABuilder  # noqa: E402

TOL = 0.05
NUDGE = "\nEnd your reply with a line of the form \"Answer: <number>\"."


def parse_value(text):
    m = re.findall(r"Answer:\s*\$?\\?\(?\s*(-?\d+\.?\d*)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def chat_generate(model, hf_tok, user, max_new=768):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ids = hf_tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                     enable_thinking=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return mlx_lm.generate(model, hf_tok, prompt=list(ids), max_tokens=max_new, verbose=False)


def parse_answer(text):
    """PsiLM arm: the trained reply is 'u at x = {x0} equals {u}.' -- score only
    the number after 'equals' (a reply that stops at the x0 echo is a miss)."""
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def score(pred, true):
    return pred is not None and abs(pred - true) <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--l-rev", type=int, default=None)
    ap.add_argument("--arms", default="baseline,oracle,psilm")
    ap.add_argument("--max-new", type=int, default=768,
                    help="token budget for the text arms; Qwen3-8B writes a long derivation "
                         "before its Answer line (160 truncated every item)")
    ap.add_argument("--out", default="final_eval.json")
    args = ap.parse_args()
    arms = args.arms.split(",")

    model, tok = mlx_lm.load(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    fno = convert_from_torch("results/stage2/fno.pt")
    bridges = PsiBridgesMLX(d_model=model.args.hidden_size)
    ckpt = Path(f"results/stage2{args.tag}/bridges.npz")
    bridges.load_weights(str(ckpt))
    meta = json.loads(Path(str(ckpt) + ".meta").read_text())
    psi = PsiLMMLX(model, tok, fno, bridges, l_rev=args.l_rev)
    builder = QABuilder(hf_tok)
    print(f"{args.model} | bridges step {meta['step']} | couple {psi.l_fwd}/{psi.l_rev} of {psi.n_layers}",
          flush=True)

    items = json.loads(Path("data/stage2_qa_val.json").read_text())[: args.n]
    agg = {k: [0, []] for k in arms + ["zero"]}
    rows = []
    t0 = time.time()
    for i, item in enumerate(items):
        q = QUESTION.format(a=item["a"], phi=item["phi"], x0=item["x0"])
        true = item["u"]
        preds, texts = {}, {}
        if "baseline" in arms:
            texts["baseline"] = chat_generate(model, hf_tok, q + NUDGE, args.max_new)
            preds["baseline"] = parse_value(texts["baseline"])
        if "oracle" in arms:
            oq = q + f"\n\nA trusted solver reports: u({item['x0']}) = {true:.2f}." + NUDGE
            texts["oracle"] = chat_generate(model, hf_tok, oq, args.max_new)
            preds["oracle"] = parse_value(texts["oracle"])
        if "psilm" in arms:
            texts["psilm"] = psi.generate(builder, item)
            preds["psilm"] = parse_answer(texts["psilm"])
        preds["zero"] = 0.0
        row = {"item": item, "text": {k: v[-200:] for k, v in texts.items()},
               "has_answer_line": {k: ("Answer:" in v) for k, v in texts.items()}}
        for k, p in preds.items():
            ok = score(p, true)
            agg[k][0] += ok
            if p is not None:
                agg[k][1].append(abs(p - true))
            row[k] = {"pred": p, "ok": bool(ok)}
        rows.append(row)
        if (i + 1) % 10 == 0:
            running = {k: round(v[0] / (i + 1), 3) for k, v in agg.items()}
            print(f"  {i+1}/{len(items)} {running} ({(time.time()-t0)/(i+1):.1f}s/item)", flush=True)

    n = len(items)
    summary = {"n": n, "step": meta["step"], "model": args.model, "tolerance": TOL,
               "couple": [psi.l_fwd, psi.l_rev, psi.n_layers]}
    for k, (c, errs) in agg.items():
        summary[k] = {"acc": round(c / n, 4),
                      "mae": round(sum(errs) / len(errs), 4) if errs else None}
        if k in texts:
            summary[k]["answer_line_rate"] = round(
                sum(r["has_answer_line"][k] for r in rows) / n, 4)
    Path(f"results/stage2{args.tag}/{args.out}").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("FINAL " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
