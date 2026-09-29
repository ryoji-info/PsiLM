#!/usr/bin/env python3
"""Does the excerpt in context change WHETHER the backbone refuses? Paired, per split.

The data builder (eval/build_constitution_data.py) stores, for every prompt, the
backbone's own reply and the reply of the teacher (the same backbone with the
constitution excerpt in its context), and their keyword refusal RATES per split
(data/constitution_<name>_stats.json). A rate that does not move can hide
decisions that do, and a bridge trained on one split is evaluated on another, so
this counts the decisions prompt by prompt:

  python eval/teacher_refusal_paired.py --name bonsai27b --tokenizer <backbone dir>

Writes results/constitution/teacher_refusal_<name>.json. No model is loaded: the
stored token ids are decoded and read by the keyword test of eval/bench_common.py,
and the rates are checked against the ones the builder stored.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import is_refusal                      # noqa: E402
from const_refusal_mcnemar import mcnemar_exact          # noqa: E402

SPLITS = (("train", "train"), ("val", "val"), ("test", "test"), ("helpful_test", "helpful_test"))


def paired(flags_base, flags_teacher):
    to = sum(t and not b for b, t in zip(flags_base, flags_teacher))
    frm = sum(b and not t for b, t in zip(flags_base, flags_teacher))
    n = len(flags_base)
    return {"n": n, "base": sum(flags_base), "teacher": sum(flags_teacher),
            "rate_base": round(sum(flags_base) / n, 4), "rate_teacher": round(sum(flags_teacher) / n, 4),
            "to_refusal": to, "from_refusal": frm, "discordant": to + frm,
            "p": round(mcnemar_exact(frm, to), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", required=True, help="the backbone's directory or Hub id (its tokenizer only)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    stats = json.loads((Path(a.data_dir) / f"constitution_{a.name}_stats.json").read_text())
    out = {"name": a.name, "what": "keyword refusal of the teacher's stored reply against the backbone's own, prompt "
                                   "by prompt; exact McNemar", "keywords": stats["refusal_keywords"], "splits": {}}
    for split, key in SPLITS:
        items = json.loads((Path(a.data_dir) / f"constitution_{split}_{a.name}.json").read_text())
        fb = [is_refusal(tok.decode(it["base_ids"], skip_special_tokens=True)) for it in items]
        ft = [is_refusal(tok.decode(it["teacher_ids"], skip_special_tokens=True)) for it in items]
        p = paired(fb, ft)
        want = stats["splits"][key]
        if (p["n"], p["rate_base"], p["rate_teacher"]) != (
                want["n"], want["refusal_rate_base"], want["refusal_rate_teacher"]):
            raise SystemExit(f"{split}: {p} is not what the builder stored ({want})")
        out["splits"][split] = p
        print(f"{split}: base {p['base']} teacher {p['teacher']} of {p['n']}; to a refusal {p['to_refusal']}, "
              f"from one {p['from_refusal']}, p {p['p']}")
    dest = Path(a.out or f"results/constitution/teacher_refusal_{a.name}.json")
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
