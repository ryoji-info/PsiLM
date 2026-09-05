"""Final Stage-2d evaluation on MLX (hybrid: MLX language side, torch DPOT),
four arms on held-out 2D Fisher-KPP QA (port of eval/stage2d_eval.py with
eval/mlx_stage2_eval.py's protocol):

  baseline : frozen LLM alone (plain chat generation, "Answer: <number>" nudge)
  oracle   : frozen LLM + the true value stated in text (tool-loop ceiling)
  psilm    : the coupled system (latent bridges, no text at the interface);
             the trained reply is 'u at ({x0}, {y0}) equals {u}.' and only the
             number after 'equals' is scored
  zero     : degenerate always-0.00 strategy (calibration)

Usage:
  python eval/mlx_stage2d_eval.py --tag _mlx2d --n 60
  python eval/mlx_stage2d_eval.py --model mlx-community/gemma-4-12B-it-4bit \
      --hf-tokenizer mlx-community/gemma-4-12B-it-4bit --tag _gemma12b_2d --n 60
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

from psilm.mlx.bridges2d import PsiBridges2DMLX  # noqa: E402
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from psilm.mlx.model2d import PsiLM2DMLX  # noqa: E402
from psilm.mlx.physics2d import TorchPhysics2D  # noqa: E402
from psilm.stage2.qa import SYSTEM  # noqa: E402
from psilm.stage2d.qa2d import QUANTITIES, QUESTION, QA2DBuilder  # noqa: E402

TOL = 0.05
NUDGE = "\nEnd your reply with a line of the form \"Answer: <number>\"."


def parse_value(text):
    m = re.findall(r"Answer:\s*\$?\\?\(?\s*(-?\d+\.?\d*)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def parse_answer(text):
    """PsiLM arm: strict -- only the number after 'equals'."""
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def chat_generate(model, hf_tok, user, max_new=768, gen_tok=None):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ids = hf_tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                     enable_thinking=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return mlx_lm.generate(model, gen_tok or hf_tok, prompt=list(ids), max_tokens=max_new, verbose=False)


def score(pred, true):
    return pred is not None and abs(pred - true) <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tag", default="_mlx2d")
    ap.add_argument("--l-rev", type=int, default=None)
    ap.add_argument("--arms", default="baseline,oracle,psilm")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--phys-device", default="mps")
    ap.add_argument("--out", default="final_eval.json")
    args = ap.parse_args()
    arms = args.arms.split(",")

    model, stock, tok = load_backbone_any(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    ckpt = Path(f"results/stage2d{args.tag}/bridges.npz")
    meta = json.loads(Path(str(ckpt) + ".meta").read_text())
    margs = meta.get("args", {})
    assert meta.get("quantities", list(QUANTITIES)) == list(QUANTITIES), "checkpoint quantity order differs"
    bridges = PsiBridges2DMLX(d_model=model.args.hidden_size,
                              gate_bias=margs.get("gate_bias", -2.0),
                              inj_cap=margs.get("inj_cap"), channel=margs.get("channel", "value"),
                              readout_norm=margs.get("readout_norm", "rms"))
    bridges.load_weights(str(ckpt))
    phys = TorchPhysics2D(device=args.phys_device)
    l_rev = args.l_rev if args.l_rev is not None else meta.get("l_rev")
    psi = PsiLM2DMLX(model, tok, phys, bridges, l_fwd=meta.get("l_fwd"), l_rev=l_rev)
    builder = QA2DBuilder(hf_tok)
    print(f"{args.model} | bridges step {meta['step']} | couple {psi.l_fwd}/{psi.l_rev} of "
          f"{psi.n_layers} | DPOT on {phys.device}", flush=True)

    items = json.loads(Path("data/stage2d_qa_val.json").read_text())[: args.n]
    agg = {k: [0, []] for k in arms + ["zero"]}
    rows = []
    t0 = time.time()
    for i, item in enumerate(items):
        q = QUESTION.format(**{k: item[k] for k in QUANTITIES})
        true = item["u"]
        preds, texts = {}, {}
        if "baseline" in arms:
            texts["baseline"] = chat_generate(stock, hf_tok, q + NUDGE, args.max_new, gen_tok=tok)
            preds["baseline"] = parse_value(texts["baseline"])
        if "oracle" in arms:
            oq = q + f"\n\nA trusted solver reports: u({item['x0']}, {item['y0']}) = {true:.2f}." + NUDGE
            texts["oracle"] = chat_generate(stock, hf_tok, oq, args.max_new, gen_tok=tok)
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
    Path(f"results/stage2d{args.tag}/{args.out}").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("FINAL " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
