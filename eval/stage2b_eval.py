"""Stage-2b/2c final evaluation: four arms across three test families.

Families: val_iid (in-distribution), val_combo (held-out mode combination
{1,2,3}), val_amp (amplitude extrapolation). Arms: baseline / oracle / psilm
/ zero, as in stage2_eval.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.physics.fno import FNO1d  # noqa: E402
from psilm.stage2.bridges import PsiBridges, build_ic_multi  # noqa: E402
from psilm.stage2.loop_model import PsiLMLoop  # noqa: E402
from psilm.stage2.model import PsiLM  # noqa: E402
from psilm.stage2.qa2 import QUESTION, QA2Builder, ic_text  # noqa: E402
from psilm.stage2.qa import SYSTEM  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 0.05


def parse_value(text):
    m = re.findall(r"Answer:\s*\$?\\?\(?\s*(-?\d+\.?\d*)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def chat_generate(model, tok, device, user, max_new=160):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
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
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--ckpt", default="results/stage2b/bridges.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--loop", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(args.device).eval()
    fno = FNO1d().to(args.device)
    fno.load_state_dict(torch.load("results/stage2b/fno.pt", map_location=args.device))
    fno.eval()
    bridges = PsiBridges(n_params=6, fwd_kind="per_mode" if args.v2 else "pooled").to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    bridges.load_state_dict(state["bridges"])
    if args.loop:
        psi = PsiLMLoop(model, tok, fno, bridges, ic_fn=build_ic_multi,
                        n_passes=args.loop, shared=True)
    else:
        psi = PsiLM(model, tok, fno, bridges, ic_fn=build_ic_multi)
    builder = QA2Builder(tok)

    all_summaries = {}
    for family in ("val_iid", "val_combo", "val_amp"):
        items = json.loads(Path(f"data/stage2b_qa_{family}.json").read_text())[: args.n]
        agg = {k: [0, []] for k in ("baseline", "oracle", "psilm", "zero")}
        for i, item in enumerate(items):
            q = QUESTION.format(ic=ic_text(item["modes"]), x0=item["x0"])
            true = item["u"]
            nudge = "\nEnd your reply with a line of the form \"Answer: <number>\"."
            preds = {
                "baseline": parse_value(chat_generate(model, tok, args.device, q + nudge)),
                "oracle": parse_value(chat_generate(
                    model, tok, args.device,
                    q + f"\n\nA trusted solver reports: u({item['x0']}) = {true:.2f}." + nudge)),
                "psilm": parse_value(psi.generate(builder, item)),
                "zero": 0.0,
            }
            for k, p in preds.items():
                ok = p is not None and abs(p - true) <= TOL
                agg[k][0] += ok
                if p is not None:
                    agg[k][1].append(abs(p - true))
            if (i + 1) % 16 == 0:
                print(f"  {family} {i+1}/{len(items)}", flush=True)
        n = len(items)
        all_summaries[family] = {
            k: {"acc": round(c / n, 4),
                "mae": round(sum(e) / len(e), 4) if e else None}
            for k, (c, e) in agg.items()
        }
        print(family, json.dumps(all_summaries[family]))

    out = {"n_per_family": args.n, "step": state["step"], "tolerance": TOL,
           "families": all_summaries}
    tag = "_v2" if args.v2 else (f"_loop{args.loop}" if args.loop else "")
    Path(f"results/stage2b{tag}/final_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
