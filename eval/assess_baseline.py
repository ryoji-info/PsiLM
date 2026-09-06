#!/usr/bin/env python3
"""Assess a text arm's score in a stage-2 evaluation file.

A 0% baseline can mean two very different things: the backbone answered and was
wrong (a real result), or the protocol never got a number out of it (an
artifact). This reads one final_eval*.json and separates them:

  * did the arm produce a parseable number, and how often was it forced;
  * how far off it was (MAE, near-miss counts at 2x and 4x the tolerance);
  * whether it guessed one constant, and what the best constant would have
    scored on the same items -- the honest floor a text-only arm competes with.

  python eval/assess_baseline.py results/stage2_gemma12b/final_eval_baseline_forced.json
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

TOL = 0.05


def stats(xs):
    if not xs:
        return "n/a"
    return (f"mean {statistics.mean(xs):+.3f}  sd "
            f"{(statistics.stdev(xs) if len(xs) > 1 else 0.0):.3f}  "
            f"min {min(xs):+.3f}  max {max(xs):+.3f}")


def best_constant(trues, tol=TOL):
    """(accuracy, value) of the best always-the-same-number strategy."""
    best = (0.0, 0.0)
    lo, hi = min(trues), max(trues)
    c = lo
    while c <= hi:
        acc = sum(abs(t - c) <= tol for t in trues) / len(trues)
        if acc > best[0]:
            best = (acc, c)
        c += 0.01
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    doc = json.loads(Path(args.path).read_text())
    summary, rows = doc["summary"], doc["rows"]
    arm = args.arm
    if arm not in summary:
        raise SystemExit(f"{args.path}: no '{arm}' arm (has {list(summary)})")

    trues = [r["item"]["u"] for r in rows]
    preds = [r[arm]["pred"] for r in rows]
    got = [p for p in preds if p is not None]
    errs = [abs(p - t) for p, t in zip(preds, trues) if p is not None]
    hits = sum(r[arm]["ok"] for r in rows)
    near2 = sum(e <= 2 * args.tol for e in errs)
    near4 = sum(e <= 4 * args.tol for e in errs)
    counts = collections.Counter(round(p, 2) for p in got).most_common(5)
    bc_acc, bc_val = best_constant(trues, args.tol)

    print(f"{args.path}  arm={arm}  n={len(rows)}")
    if "protocol" in summary:
        print(f"  protocol            {summary['protocol']}")
    print(f"  accuracy            {hits}/{len(rows)} = {hits/len(rows):.3f} "
          f"(tolerance {args.tol})")
    print(f"  parseable answers   {len(got)}/{len(rows)}"
          f"   answer line {summary[arm].get('answer_line_rate')}"
          f"   forced {summary[arm].get('forced_rate')}")
    print(f"  |error|             mean {statistics.mean(errs):.3f}" if errs else
          "  |error|             n/a")
    print(f"  within 2x tol       {near2}/{len(rows)}      within 4x tol {near4}/{len(rows)}")
    print(f"  predictions         {stats(got)}")
    print(f"  truths              {stats(trues)}")
    print(f"  most repeated pred  {counts}")
    print(f"  best constant       {bc_acc:.3f} at u = {bc_val:+.2f}"
          f"   (always-0.00 arm: {summary.get('zero', {}).get('acc')})")

    # A zero is a real result when the arm answered and missed; it is a protocol
    # artifact when the arm never produced a number to score.
    if len(got) < 0.9 * len(rows):
        verdict = ("ARTIFACT SUSPECT: the arm did not produce a number on "
                   f"{len(rows)-len(got)} of {len(rows)} items")
    elif hits == 0 and near4 == 0:
        verdict = ("GENUINE: every item answered, none within 4x the tolerance "
                   "-- the backbone is not solving the PDE, it is guessing")
    elif hits == 0:
        verdict = (f"GENUINE but close on {near4} item(s) at 4x tolerance -- "
                   "report the near-miss count alongside the zero")
    else:
        verdict = "non-zero score; nothing to explain away"
    print(f"  verdict             {verdict}")


if __name__ == "__main__":
    main()
