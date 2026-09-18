#!/usr/bin/env python3
"""Total measured GPU time, summed from the run logs, for the compute statement.

Three sources, none of them wall-clock spans (which would count waits):
  training    sum over runs of steps x sec_per_step from results/*/train_log.jsonl
  evals       the last "k/N items, Ns" line of every eval_*.log (50-item evals)
  guard-rails sec_per_q x n for every arm x dataset in *_guardrail_summary.json
Excluded because they are not logged at a rate: value-neuron state collection and
probe training, data building (teacher generation), the 0.5B ablations, and the
constitution partner's fine-tune. Say so wherever the total is quoted.

  python3 eval/total_compute.py [--out results/compute_summary.json] [--exclude run,run]

The totals cover every run in the checkout, physics paper and constitution paper
alike. The "psilm2" block is the subset the PsiLM-2 paper quotes: training runs
named stage2c_* (the constitution bridges at both scales) and stage2_qwen35 (the
Qwen3.5 physics bridge), the 50-item evaluations under those directories, and the
const_* and dual_qwen35_* guard-rails; --exclude names runs still in progress so
that a snapshot counts only what it reports.
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


PSILM2_TRAINING = ("stage2c_", "stage2_qwen35")
PSILM2_GUARDRAILS = ("const_", "dual_qwen35_")


def scoped(tr, ev, gr, exclude):
    """The PsiLM-2 paper's runs, by name prefix, minus --exclude."""
    def keep_run(name):
        return name.startswith(PSILM2_TRAINING) and name not in exclude
    tr2 = {k: v for k, v in tr.items() if keep_run(k)}
    ev2 = {k: v for k, v in ev.items() if keep_run(Path(k).parts[1] if len(Path(k).parts) > 1 else "")}
    gr2 = {k: v for k, v in gr.items() if k.startswith(PSILM2_GUARDRAILS)
           and not any(k.startswith("const_" + x[len("stage2c_"):]) for x in exclude if x.startswith("stage2c_"))}
    tot = {"training_h": round(sum(tr2.values()), 1), "evals_h": round(sum(ev2.values()), 1),
           "guardrails_h": round(sum(gr2.values()), 1)}
    tot["total_h"] = round(sum(tot.values()), 1)
    return {"scope": "training runs stage2c_* and stage2_qwen35, their eval_*.log files, guard-rails const_* and dual_qwen35_*",
            "excluded_runs": sorted(exclude), "totals": tot,
            "training": tr2, "evals": ev2, "guardrails": gr2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/compute_summary.json")
    ap.add_argument("--exclude", default="", help="comma-separated run names left out of the psilm2 block (runs still in progress)")
    a = ap.parse_args()
    tr, ev, gr = training(a.root), evals(a.root), guardrails(a.root)
    tot = {"training_h": round(sum(tr.values()), 1), "evals_h": round(sum(ev.values()), 1),
           "guardrails_h": round(sum(gr.values()), 1)}
    tot["total_h"] = round(sum(tot.values()), 1)
    exclude = {x for x in a.exclude.split(",") if x}
    out = {"totals": tot, "training": tr, "evals": ev, "guardrails": gr,
           "psilm2": scoped(tr, ev, gr, exclude),
           "excluded": "value-neuron collection and probes, data building, 0.5B ablations, the partner fine-tune"}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"training {tot['training_h']} h over {len(tr)} runs | evals {tot['evals_h']} h over {len(ev)} | "
          f"guard-rails {tot['guardrails_h']} h over {len(gr)} | total {tot['total_h']} h -> {a.out}")
    p = out["psilm2"]["totals"]
    print(f"psilm2 scope: training {p['training_h']} h over {len(out['psilm2']['training'])} runs | "
          f"evals {p['evals_h']} h | guard-rails {p['guardrails_h']} h over {len(out['psilm2']['guardrails'])} | "
          f"total {p['total_h']} h (excluded: {', '.join(sorted(exclude)) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
