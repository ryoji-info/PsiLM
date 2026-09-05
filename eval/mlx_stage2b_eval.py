"""Stage-2b final evaluation on MLX: four arms across three held-out families
(port of eval/stage2b_eval.py onto the stack of eval/mlx_stage2_eval.py).

Families (n=48 each by default): data/stage2b_qa_val_{iid,combo,amp}.json --
in-distribution, the held-out mode combination {1,2}, amplitude extrapolation.

Arms:
  baseline : frozen LLM alone (plain chat generation, "Answer: <number>" nudge)
  oracle   : frozen LLM + the true value stated in text (tool-loop ceiling)
  psilm    : the coupled system (latent bridges; strict 'equals' parsing)
  zero     : degenerate always-0.00 strategy (calibration)

Text arms run mlx_lm.generate on the STOCK model with the mlx tokenizer
(knows every stop id, e.g. Gemma's <turn|>) exactly as eval/mlx_stage2_eval.py.
Bridge construction args (gate_bias, inj_cap, channel, readout_norm) are read
from the checkpoint meta. Output: results/stage2b{tag}/final_eval.json with the
same 'families' layout as the torch script, plus per-item rows.

Usage:
  python eval/mlx_stage2b_eval.py --tag _mlx2b --n 48
  python eval/mlx_stage2b_eval.py --model mlx-community/gemma-4-12B-it-4bit \\
      --hf-tokenizer mlx-community/gemma-4-12B-it-4bit --tag _gemma12b_2b --n 48
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

from psilm.mlx.fno import convert_from_torch  # noqa: E402
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from psilm.mlx.multimode import PsiLMMLXMulti, make_bridges_multi  # noqa: E402
from psilm.stage2.qa import SYSTEM  # noqa: E402
from psilm.stage2.qa2 import QUESTION, QA2Builder, ic_text  # noqa: E402

TOL = 0.05
NUDGE = "\nEnd your reply with a line of the form \"Answer: <number>\"."
FAMILIES = ("val_iid", "val_combo", "val_amp")
FNO_PATH = "results/stage2b/fno.pt"


def parse_value(text):
    m = re.findall(r"Answer:\s*\$?\\?\(?\s*(-?\d+\.?\d*)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def chat_generate(model, hf_tok, user, max_new=768, gen_tok=None):
    """gen_tok: the mlx-lm tokenizer wrapper (knows all of a backbone's stop ids,
    e.g. Gemma's <eos>/<turn|>); hf_tok only builds the chat prompt."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ids = hf_tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                     enable_thinking=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return mlx_lm.generate(model, gen_tok or hf_tok, prompt=list(ids), max_tokens=max_new, verbose=False)


def parse_answer(text):
    """PsiLM arm: the trained reply is 'u at x = {x0} equals {u}.' -- score only
    the number after 'equals' (a reply that stops at the x0 echo is a miss)."""
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def score(pred, true):
    return pred is not None and abs(pred - true) <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="items per family")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tag", default="_mlx2b")
    ap.add_argument("--l-rev", type=int, default=None)
    ap.add_argument("--arms", default="baseline,oracle,psilm")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--max-new", type=int, default=768,
                    help="token budget for the text arms (large backbones write a long "
                         "derivation before their Answer line)")
    ap.add_argument("--out", default="final_eval.json")
    args = ap.parse_args()
    arms = args.arms.split(",")
    families = args.families.split(",")

    model, stock, tok = load_backbone_any(args.model)     # tower for PsiLM, stock for generate
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    fno = convert_from_torch(FNO_PATH)
    ckpt = Path(f"results/stage2b{args.tag}/bridges.npz")
    meta = json.loads(Path(str(ckpt) + ".meta").read_text())
    margs = meta.get("args", {})
    bridges = make_bridges_multi(model.args.hidden_size,
                                 gate_bias=margs.get("gate_bias", -2.0),
                                 inj_cap=margs.get("inj_cap"), channel=margs.get("channel", "field"),
                                 readout_norm=margs.get("readout_norm", "rms"))
    bridges.load_weights(str(ckpt))
    l_rev = args.l_rev if args.l_rev is not None else meta.get("l_rev")
    psi = PsiLMMLXMulti(model, tok, fno, bridges, l_rev=l_rev)
    builder = QA2Builder(hf_tok)
    print(f"{args.model} | bridges step {meta['step']} | couple {psi.l_fwd}/{psi.l_rev} of {psi.n_layers}",
          flush=True)

    all_summaries, all_rows = {}, {}
    t0 = time.time()
    for family in families:
        items = json.loads(Path(f"data/stage2b_qa_{family}.json").read_text())[: args.n]
        agg = {k: [0, []] for k in arms + ["zero"]}
        rows = []
        for i, item in enumerate(items):
            q = QUESTION.format(ic=ic_text(item["modes"]), x0=item["x0"])
            true = item["u"]
            preds, texts = {}, {}
            if "baseline" in arms:
                texts["baseline"] = chat_generate(stock, hf_tok, q + NUDGE, args.max_new, gen_tok=tok)
                preds["baseline"] = parse_value(texts["baseline"])
            if "oracle" in arms:
                oq = q + f"\n\nA trusted solver reports: u({item['x0']}) = {true:.2f}." + NUDGE
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
            if (i + 1) % 16 == 0 or i + 1 == len(items):
                running = {k: round(v[0] / (i + 1), 3) for k, v in agg.items()}
                print(f"  {family} {i+1}/{len(items)} {running} "
                      f"({(time.time()-t0)/max(1, sum(len(r) for r in all_rows.values()) + i + 1):.1f}s/item)",
                      flush=True)
        n = len(items)
        summ = {}
        for k, (c, errs) in agg.items():
            summ[k] = {"acc": round(c / n, 4),
                       "mae": round(sum(errs) / len(errs), 4) if errs else None}
            if k in texts:
                summ[k]["answer_line_rate"] = round(sum(r["has_answer_line"][k] for r in rows) / n, 4)
        all_summaries[family] = summ
        all_rows[family] = rows
        print(family, json.dumps(summ), flush=True)

    out = {"n_per_family": args.n, "step": meta["step"], "tolerance": TOL,
           "model": args.model, "couple": [psi.l_fwd, psi.l_rev, psi.n_layers],
           "families": all_summaries, "rows": all_rows}
    Path(f"results/stage2b{args.tag}/{args.out}").write_text(json.dumps(out, indent=1))
    print("FINAL " + json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
