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
from pathlib import Path


def eps_of(arm):
    return 0.0 if arm in ("base", "psilm", "zeroed") else float(arm[len("leaky"):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default="psilm", help="arm the McNemar tests compare against")
    args = ap.parse_args()

    doc = json.loads(Path(args.path).read_text())
    summary, arms = doc["summary"], doc["arms"]
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
          + "     (pre-floor; flat down each column = the floor left the gate alone)")
    for arm in coupled:
        row = []
        for d in datasets:
            sg = summary[d]["arms"].get(arm, {}).get("sigma")
            row.append(f"{sg['mean']:.5f}" if sg else "-")
        print(f"{arm:11s} " + " ".join(f"{c:>14s}" for c in row))


if __name__ == "__main__":
    main()
