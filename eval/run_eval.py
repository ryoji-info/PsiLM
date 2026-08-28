"""Run the Stage-0 evaluation: LLM-alone vs LLM+simulator on the physics QA set.

Usage:
  python eval/run_eval.py [--model mlx-community/Qwen2.5-0.5B-Instruct-4bit]
                          [--qa data/qa_stage0.json] [--limit N] [--arms alone,tool]
Writes results/stage0_<model-shortname>.json and prints a per-scene table.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.arms import answer_alone, answer_with_tool  # noqa: E402
from psilm.llm import DEFAULT_MODEL, LLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--qa", default="data/qa_stage0.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="alone,tool")
    args = ap.parse_args()

    items = json.loads(Path(args.qa).read_text())
    if args.limit:
        items = items[: args.limit]
    arms = [a.strip() for a in args.arms.split(",")]

    print(f"model: {args.model}\nitems: {len(items)}  arms: {arms}\n")
    llm = LLM(args.model)

    records = []
    t0 = time.time()
    for i, item in enumerate(items):
        rec = {"id": item["id"], "scene": item["scene"], "answer": item["answer"]}
        if "alone" in arms:
            rec["alone"] = answer_alone(llm, item)
        if "tool" in arms:
            rec["tool"] = answer_with_tool(llm, item)
        records.append(rec)
        done = i + 1
        if done % 10 == 0 or done == len(items):
            print(f"  {done}/{len(items)}  ({time.time() - t0:.0f}s)")

    summary = summarize(records, arms)
    short = args.model.rsplit("/", 1)[-1]
    out = Path("results") / f"stage0_{short}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "n": len(records), "summary": summary, "records": records},
        indent=1,
    ))
    print(f"\nwrote {out}\n")
    print_table(summary, arms)


def summarize(records, arms):
    per_scene = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    total = defaultdict(lambda: [0, 0])
    tool_ok = [0, 0]
    for rec in records:
        for arm in arms:
            correct = rec[arm].get("choice") == rec["answer"]
            per_scene[rec["scene"]][arm][0] += int(correct)
            per_scene[rec["scene"]][arm][1] += 1
            total[arm][0] += int(correct)
            total[arm][1] += 1
        if "tool" in arms:
            tool_ok[0] += int(rec["tool"].get("tool_ok", False))
            tool_ok[1] += 1
    out = {
        "total": {arm: {"correct": c, "n": n, "acc": round(c / n, 4)} for arm, (c, n) in total.items()},
        "per_scene": {
            scene: {arm: {"correct": c, "n": n, "acc": round(c / n, 4)} for arm, (c, n) in d.items()}
            for scene, d in per_scene.items()
        },
    }
    if "tool" in arms:
        out["tool_call_success"] = {"ok": tool_ok[0], "n": tool_ok[1], "rate": round(tool_ok[0] / tool_ok[1], 4)}
    return out


def print_table(summary, arms):
    scenes = sorted(summary["per_scene"])
    header = f"{'scene':<20}" + "".join(f"{arm:>10}" for arm in arms)
    print(header)
    print("-" * len(header))
    for scene in scenes:
        row = f"{scene:<20}"
        for arm in arms:
            s = summary["per_scene"][scene][arm]
            row += f"{s['correct']:>6}/{s['n']:<3}"
        print(row)
    row = f"{'TOTAL':<20}"
    for arm in arms:
        s = summary["total"][arm]
        row += f"{s['acc'] * 100:>8.1f}% "
    print("-" * len(header))
    print(row)
    if "tool_call_success" in summary:
        t = summary["tool_call_success"]
        print(f"tool-call success: {t['ok']}/{t['n']} ({t['rate'] * 100:.0f}%)")


if __name__ == "__main__":
    main()
