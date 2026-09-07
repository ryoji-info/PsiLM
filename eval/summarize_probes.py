#!/usr/bin/env python3
"""Aggregate the readout-transfer probes into one table.

Groups every results/readout_transfer/*.json by (readout, layer, augmentation)
and reports mean and range per family, because the combination family's
seed-to-seed spread turned out to be as large as every effect measured on it.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

FAMILIES = ("val_iid", "val_combo", "val_amp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/readout_transfer")
    args = ap.parse_args()

    groups = defaultdict(list)
    for f in sorted(Path(args.dir).glob("*.json")):
        doc = json.loads(f.read_text())
        for kind, r in doc["readouts"].items():
            aug = ""
            if doc.get("aug_zero_frac"):
                aug = (f", distractor {'0.0' if not doc.get('aug_amp_max') else 'U(0.02,' + str(doc['aug_amp_max']) + ')'}"
                       f" x{doc['aug_zero_frac']}")
            key = (kind, doc.get("l_fwd") or "default", aug, doc.get("steps"))
            groups[key].append({f: r[f]["acc"] for f in FAMILIES})

    print(f"{'readout':8s} {'layer':>7s} {'steps':>6s} {'n':>2s}  "
          + "  ".join(f"{f.replace('val_', ''):>18s}" for f in FAMILIES) + "   augmentation")
    for (kind, layer, aug, steps), runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cells = []
        for fam in FAMILIES:
            xs = [r[fam] for r in runs]
            m = statistics.mean(xs)
            cells.append(f"{m:.3f}" + (f" [{min(xs):.2f}-{max(xs):.2f}]" if len(xs) > 1 else " " * 12))
        print(f"{kind:8s} {str(layer):>7s} {str(steps):>6s} {len(runs):>2d}  "
              + "  ".join(f"{c:>18s}" for c in cells) + f"   {aug or '-'}")


if __name__ == "__main__":
    main()
