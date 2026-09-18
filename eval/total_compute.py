#!/usr/bin/env python3
"""Total measured GPU time, summed from the run logs, for the compute statement.

Three sources, none of them wall-clock spans (which would count waits):
  training    sum over runs of steps x sec_per_step from results/*/train_log.jsonl
  evals       the last "k/N items, Ns" line of every eval_*.log (50-item evals)
  guard-rails sec_per_q x n for every arm x dataset in *_guardrail_summary.json
Excluded because they are not logged at a rate: value-neuron state collection and
probe training, data building (teacher generation), the 0.5B ablations, and the
constitution partner's fine-tune. Say so wherever the total is quoted.

  python3 eval/total_compute.py [--out results/compute_summary.json]
"""
import argparse, glob, json, os, re
from pathlib import Path


def training(root):
    per = {}
    for f in sorted(glob.glob(f"{root}/results/*/train_log.jsonl")):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        last, t = 0, 0.0
        for r in rows:
            if "sec_per_step" in r and "step" in r:
                t += (r["step"] - last) * r["sec_per_step"]; last = r["step"]
        if t:
            per[Path(f).parent.name] = round(t / 3600, 2)
    return per


def evals(root):
    per = {}
    for f in sorted(glob.glob(f"{root}/results/*/eval_*.log")):
        m = re.findall(r"(\d+)/(\d+) items, (\d+)s", open(f, errors="ignore").read())
        if m:
            k, n, s = map(int, m[-1])
            per[str(Path(f).relative_to(root))] = round(int(s) * n / max(k, 1) / 3600, 3)  # scale to the full run
    return per


def guardrails(root):
    per = {}
    for f in sorted(glob.glob(f"{root}/results/bench/*_guardrail_summary.json")):
        d = json.load(open(f)); t = 0.0
        for ds, v in d.get("summary", {}).items():
            for arm, a in v.get("arms", {}).items():
                if a.get("sec_per_q") and a.get("n"):
                    t += a["sec_per_q"] * a["n"]
        if t:
            per[Path(f).name.replace("_guardrail_summary.json", "")] = round(t / 3600, 2)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/compute_summary.json")
    a = ap.parse_args()
    tr, ev, gr = training(a.root), evals(a.root), guardrails(a.root)
    tot = {"training_h": round(sum(tr.values()), 1), "evals_h": round(sum(ev.values()), 1),
           "guardrails_h": round(sum(gr.values()), 1)}
    tot["total_h"] = round(sum(tot.values()), 1)
    out = {"totals": tot, "training": tr, "evals": ev, "guardrails": gr,
           "excluded": "value-neuron collection and probes, data building, 0.5B ablations, the partner fine-tune"}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"training {tot['training_h']} h over {len(tr)} runs | evals {tot['evals_h']} h over {len(ev)} | "
          f"guard-rails {tot['guardrails_h']} h over {len(gr)} | total {tot['total_h']} h -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
