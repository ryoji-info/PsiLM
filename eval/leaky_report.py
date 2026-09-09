#!/usr/bin/env python3
"""Dose-response table for the leaky-gate sweep.

Reads a guard-rail report whose arms include leaky<eps> and prints, per epsilon:
accuracy on every dataset with its paired McNemar p against the trained
selective gate, the per-token KL to the base model, and the gate's pre-floor
sigma (which should NOT move -- the floor changes what is injected, not what the
gate decides, so a moving sigma column means the instrument is broken).

  python eval/leaky_report.py results/bench/leaky_8b_guardrail.json
"""
import argparse
import json
import math
from pathlib import Path


def eps_of(arm):
    """Sort key: the floor, with the content-control arms just after their
    matched leaky arm rather than crashing on the prefix."""
    if arm in ("base", "psilm", "zeroed"):
        return 0.0
    for prefix, tie in (("leaky", 0.0), ("shuffled", 0.001)):
        if arm.startswith(prefix):
            return float(arm[len(prefix):]) + tie
    raise ValueError(f"unrecognized arm {arm!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default="psilm", help="arm the McNemar tests compare against")
    ap.add_argument("--rows", nargs="*", default=[],
                    help="rows.jsonl files of the same runs. The per-run summaries carry paired "
                         "tests only between arms of their OWN run, so a merged table has no "
                         "p-values for the follow-up arms; given the rows, this recomputes every "
                         "pairing across runs (valid because they share the task cache and seed)")
    ap.add_argument("--merge", nargs="*", default=[],
                    help="further reports (the upper-eps, top-eps and shuffled sweeps) to fold "
                         "into the same table; each must have run on the same task cache and "
                         "seed as the first")
    args = ap.parse_args()

    doc = json.loads(Path(args.path).read_text())
    summary, arms = doc["summary"], doc["arms"]
    for path in args.merge:
        other = json.loads(Path(path).read_text())
        if other.get("model") != doc.get("model") or other.get("ckpt_step") != doc.get("ckpt_step"):
            raise SystemExit(f"refusing to merge {path}: different backbone or bridges checkpoint")
        for ds, blk in other["summary"].items():
            if ds not in summary:
                continue
            summary[ds]["arms"].update({a: v for a, v in blk["arms"].items() if a not in summary[ds]["arms"]})
            summary[ds]["paired"].update(blk.get("paired", {}))
        arms = arms + [a for a in other["arms"] if a not in arms]
    if args.rows:                       # recompute pairing across runs from per-item results
        rows = [json.loads(l) for f in args.rows for l in Path(f).read_text().splitlines() if l.strip()]
        by = {}
        for r in rows:
            by.setdefault((r["dataset"], r["qid"]), {})[r["arm"]] = r["ok"]
        def mcnemar(b, c):
            n = b + c
            if n == 0:
                return 1.0
            return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
        for ds in doc["datasets"]:
            qs = [k for k in by if k[0] == ds and args.ref in by[k]]
            for a in {a for k in qs for a in by[k]} - {args.ref}:
                pres = [k for k in qs if a in by[k]]
                b = sum(by[k][args.ref] and not by[k][a] for k in pres)
                c = sum(by[k][a] and not by[k][args.ref] for k in pres)
                summary[ds].setdefault("paired", {})[f"{args.ref}_vs_{a}"] = {
                    "n": len(pres), "a_only": b, "b_only": c, "mcnemar_p": round(mcnemar(b, c), 4)}

    datasets = list(doc["datasets"])
    coupled = sorted([a for a in arms if a != "base"], key=eps_of)

    print(f"{doc.get('model', '?')}  n={summary[datasets[0]]['n']} per dataset  "
          f"(bridges step {doc.get('ckpt_step')})\n")
    head = f"{'arm':11s} " + " ".join(f"{d:>12s}" for d in datasets) + "   physics MAE"
    print(head)
    print("-" * len(head))
    for arm in ["base"] + coupled:
        cells = []
        for d in datasets:
            a = summary[d]["arms"].get(arm, {})
            acc = a.get("acc")
            if acc is None:
                cells.append("-")
                continue
            p = summary[d]["paired"].get(f"{args.ref}_vs_{arm}", {}).get("mcnemar_p")
            cells.append(f"{acc:.2f}" + (f" p{p:<4.2f}" if p is not None and arm != args.ref else ""))
        mae = summary.get("physics", {}).get("arms", {}).get(arm, {}).get("mae")
        print(f"{arm:11s} " + " ".join(f"{c:>12s}" for c in cells)
              + f"   {mae if mae is not None else '-'}")

    print(f"\n{'arm':11s} " + " ".join(f"{'KL:' + d:>12s}" for d in datasets))
    for arm in coupled:
        row = []
        for d in datasets:
            k = summary[d]["arms"].get(arm, {}).get("kl_to_base")
            row.append(f"{k['mean']:.5f}" if k else "-")
        print(f"{arm:11s} " + " ".join(f"{c:>12s}" for c in row))

    print(f"\n{'arm':11s} " + " ".join(f"{'sigma:' + d:>14s}" for d in datasets)
          + "     (PRE-floor: what the gate decided, not what was applied)")
    for arm in coupled:
        row = []
        for d in datasets:
            sg = summary[d]["arms"].get(arm, {}).get("sigma")
            row.append(f"{sg['mean']:.5f}" if sg else "-")
        print(f"{arm:11s} " + " ".join(f"{c:>14s}" for c in row))


if __name__ == "__main__":
    main()
