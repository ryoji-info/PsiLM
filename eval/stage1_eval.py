"""Final Stage-1 evaluation: coupled Bicameral vs primary-alone baseline.

Held-out multiplication problems (seed disjoint from training and from the
chunk evals). Reports exact-match accuracy, exact tool recall, and per-digit
fidelity of the answer (edit-distance based), for:
  - baseline: the frozen primary model alone (plain chat generation)
  - coupled:  the Bicameral system (hidden-state channel + calculator)

Usage: python eval/stage1_eval.py [--n 40] [--ckpt results/stage1/interface.pt]
"""

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.bicameral.data import SYSTEM, ExampleBuilder, sample_operands  # noqa: E402
from psilm.bicameral.interface import BicameralInterface  # noqa: E402
from psilm.bicameral.model import Bicameral  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def digit_similarity(pred: str, true: str) -> float:
    """Similarity of the predicted result digits to the true product."""
    nums = re.findall(r"\d{4,}", pred)
    if not nums:
        return 0.0
    best = max(nums, key=lambda n: difflib.SequenceMatcher(None, n, true).ratio())
    return difflib.SequenceMatcher(None, best, true).ratio()


def baseline_answer(model, tok, device, a, b, max_new=48):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"What is {a} * {b}?"},
    ]
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
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--lo", type=int, default=10**3)
    ap.add_argument("--hi", type=int, default=10**5)
    ap.add_argument("--ckpt", default="results/stage1/interface.pt")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(args.device).eval()
    phi = BicameralInterface(896, "pull_standard", 1024, 2).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    phi.load_state_dict(state["phi"])
    bic = Bicameral(model, tok, phi)
    builder = ExampleBuilder(tok)

    rng = random.Random(args.seed)
    problems = [sample_operands(rng, args.lo, args.hi) for _ in range(args.n)]

    rows, agg = [], {"base_acc": 0, "coup_acc": 0, "tool": 0,
                     "base_sim": 0.0, "coup_sim": 0.0}
    for i, (a, b) in enumerate(problems):
        true = str(a * b)
        base = baseline_answer(model, tok, args.device, a, b)
        answer, aux = bic.generate(builder, a, b)
        row = {
            "a": a, "b": b, "true": true,
            "base_correct": true in base,
            "coup_correct": true in answer,
            "tool_exact": f"calc({a}*{b})" in aux,
            "base_sim": round(digit_similarity(base, true), 3),
            "coup_sim": round(digit_similarity(answer, true), 3),
            "base": base[-90:], "coupled": answer[-90:], "aux": aux[:90],
        }
        rows.append(row)
        agg["base_acc"] += row["base_correct"]
        agg["coup_acc"] += row["coup_correct"]
        agg["tool"] += row["tool_exact"]
        agg["base_sim"] += row["base_sim"]
        agg["coup_sim"] += row["coup_sim"]
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)

    n = len(problems)
    summary = {
        "n": n, "step": state["step"], "band": [args.lo, args.hi],
        "baseline_acc": round(agg["base_acc"] / n, 4),
        "coupled_acc": round(agg["coup_acc"] / n, 4),
        "tool_recall_exact": round(agg["tool"] / n, 4),
        "baseline_digit_sim": round(agg["base_sim"] / n, 4),
        "coupled_digit_sim": round(agg["coup_sim"] / n, 4),
    }
    out = Path("results/stage1/final_eval.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
