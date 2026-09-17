#!/usr/bin/env python3
"""Rescore the MMLU rows of every 9B guard-rail run with the repaired parser.

parse_letter had no bare-letter branch. The 9B backbone answers 18 of 100 MMLU
items as "B" or "(C)" then EOS -- n_gen 2, stopped_eos, not truncation -- and
every one scored as wrong, in every arm, in every run. Base MMLU was reported as
64; it is 75. A coupled arm that changed the FORMAT of a few of those answers
(bare letter -> "Answer: B") read as an accuracy gain, and one such gain reached
p = 0.031 and a bolded table cell. The KL columns are teacher-forced and never
touched the parser, so they are unchanged.

This regenerates each run's tracked summary from its rows with the repaired
parser, marks it rescored, and leaves the raw run file alone. It prints the
recorded and rescored accuracies side by side so the correction is visible.

  python3 eval/rescore_mmlu.py            # all results/bench/*qwen35*_guardrail.json
"""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.bench_common import score, summarize, format_table   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/bench/*qwen35*_guardrail.json")
    a = ap.parse_args()
    print(f"{'run':30s} {'arm':8s} {'recorded':>9s} {'rescored':>9s} {'changed rows':>13s}")
    for f in sorted(glob.glob(a.glob)):
        d = json.loads(Path(f).read_text())
        rows = d.get("rows") or []
        if not any(r.get("dataset") == "mmlu" for r in rows):
            continue
        changed = {}
        for r in rows:
            if r.get("dataset") != "mmlu":
                continue
            pred, ok = score(r["protocol"], r.get("text") or "", r["gold"])
            if (pred, bool(ok)) != (r.get("pred"), bool(r.get("ok"))):
                changed[r["arm"]] = changed.get(r["arm"], 0) + 1
            r["pred"], r["ok"] = pred, ok
        arms = d["arms"]; datasets = d["datasets"]
        thresh = d["config"].get("gate_open_thresh", 0.1)
        s = summarize(rows, datasets, arms, thresh)
        old = d["summary"]
        for arm in arms:
            rec = old.get("mmlu", {}).get(arm) or {}
            new = s["summary"].get("mmlu", {}).get(arm) or {}
            k = next((k for k in ('acc', 'accuracy', 'mean_ok') if k in new), None)
            print(f"{d['tag']:30s} {arm:8s} {str(rec.get(k)):>9s} {str(new.get(k)):>9s} "
                  f"{changed.get(arm, 0):>13d}")
        d["summary"], d["gate_table"] = s["summary"], s["gate_table"]
        d["table_text"] = format_table(s["summary"], arms)
        d["rescored_mmlu_parse"] = {"note": "MMLU pred/ok rescored with the bare-letter "
                                            "branch of parse_letter (eval/rescore_mmlu.py); "
                                            "KL columns unaffected",
                                    "rows_changed": changed}
        out = Path(f).with_name(Path(f).name.replace("_guardrail.json", "_guardrail_summary.json"))
        d.pop("rows", None)
        out.write_text(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
