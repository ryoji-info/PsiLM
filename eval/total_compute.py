#!/usr/bin/env python3
"""Total measured GPU time, summed from the run logs, for the compute statement.

Four sources, none of them wall-clock spans (which would count waits):
  training    sum over runs of steps x sec_per_step from results/*/train_log.jsonl
  evals       the last "k/N items, Ns" line of every eval_*.log (50-item evals)
  guard-rails sec_per_q x n for every arm x dataset in *_guardrail_summary.json
  rescores    the KL-read rescoring (bench_guardrail.py --kl-rescore): the "sec" of
              each row of *_klpool_guardrail.rows.jsonl; rows written before that
              field existed are costed at the mean of the file's timed rows, or from
              the run log's last "[rescore k/N] ... X min" line when none is timed
Excluded because they are not logged at a rate: value-neuron state collection and
probe training, data building (teacher generation), the 0.5B ablations, and the
constitution partner's fine-tune. Say so wherever the total is quoted.

A guard-rail arm whose rows were BORROWED from a recorded run (eval/bench_seed_rows.py)
cost nothing here: where a rows file is present, only the rows this run generated are
counted.

The "psilm2_followup" block is the measurements made after the campaign closed
(2026-09-28/29): the partner-free bridge and the two replicates, the stored-token
arms, the teacher ceilings (the "sec" of their rows) and the content controls (not
timed in their own logs: the span between a run's CEILING and CONTROLS lines in the
chain's log, the mean of those for an arm the chain did not run). Its runs are left
out of the "psilm2" block, which stays the campaign's.

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


def generated(rows_file):
    """Seconds of the rows a run generated itself, or None if none is borrowed."""
    rows = [json.loads(l) for l in open(rows_file) if l.strip()]
    if not any("borrowed_from" in r for r in rows):
        return None
    return sum(r.get("sec", 0.0) for r in rows if "borrowed_from" not in r)


def guardrails(root):
    per = {}
    for f in sorted(glob.glob(f"{root}/results/bench/*_guardrail_summary.json")):
        d = json.load(open(f)); t = 0.0
        for ds, v in d.get("summary", {}).items():
            for arm, a in v.get("arms", {}).items():
                if a.get("sec_per_q") and a.get("n"):
                    t += a["sec_per_q"] * a["n"]
        rows = f.replace("_guardrail_summary.json", "_guardrail.rows.jsonl")
        own = generated(rows) if os.path.exists(rows) else None
        if own is not None:
            t = own
        if t:
            per[Path(f).name.replace("_guardrail_summary.json", "")] = round(t / 3600, 2)
    return per


def rescores(root):
    per = {}
    for f in sorted(glob.glob(f"{root}/results/bench/*_klpool_guardrail.rows.jsonl")):
        tag = Path(f).name.replace("_klpool_guardrail.rows.jsonl", "")
        rows = [json.loads(l) for l in open(f) if l.strip()]
        timed = [r["sec"] for r in rows if "sec" in r]
        n_untimed = len(rows) - len(timed)
        t = sum(timed)
        if n_untimed:
            if timed:
                t += n_untimed * t / len(timed)
            else:
                log = Path(f"{root}/results/bench/{tag}_klpool_run.log")
                m = re.findall(r"\[rescore (\d+)/(\d+)\].*?([\d.]+) min", log.read_text(errors="ignore")) \
                    if log.exists() else []
                if m:
                    k, _, mins = m[-1]
                    t += float(mins) * 60 * n_untimed / max(int(k), 1)
        if t:
            per[tag] = round(t / 3600, 2)
    return per


FOLLOWUP_RUNS = ("stage2c_qwen35_nopartner", "stage2c_qwen35_all_r1", "stage2c_qwen35_nopartner_r1")
FOLLOWUP_GUARDRAILS = ("const_qwen35_all_rt400_fixed", "const_qwen35_all_rt400_repro", "const_qwen35_nopartner",
                       "const_qwen35_nopartner_rt400", "const_qwen35_all_r1_rt400",
                       "const_qwen35_nopartner_r1_rt400")


def ceilings(root):
    per = {}
    for f in sorted(glob.glob(f"{root}/results/stage2c_qwen35_*/teacher_ceiling.rows.jsonl")):
        t = sum(json.loads(l).get("sec", 0.0) for l in open(f) if l.strip())
        if t:
            per[Path(f).parent.name] = round(t / 3600, 3)
    return per


def controls(root):
    """Hours of eval/constitution_compress.py per 9B arm, from the chain's log."""
    from datetime import datetime
    log = Path(f"{root}/results/qwen35/arms_controls.log")
    at, per = {}, {}
    for ln in (log.read_text().splitlines() if log.exists() else []):
        m = re.match(r"(CEILING|CONTROLS) (\S+) .*(\d{4}-\d\d-\d\d \d\d:\d\d)$", ln)
        if m:
            at[(m.group(1), m.group(2))] = datetime.strptime(m.group(3), "%Y-%m-%d %H:%M")
    for (kind, arm), t in at.items():
        if kind == "CONTROLS" and ("CEILING", arm) in at:
            per[f"stage2c_qwen35_{arm}"] = round((t - at[("CEILING", arm)]).total_seconds() / 3600, 3)
    mean = sum(per.values()) / len(per) if per else 0.0
    for f in sorted(glob.glob(f"{root}/results/stage2c_qwen35_*/compress.json")):
        per.setdefault(Path(f).parent.name, round(mean, 3))
    return per


def followup(tr, ev, gr, root):
    tr2 = {k: v for k, v in tr.items() if k.startswith(FOLLOWUP_RUNS)}      # the long-batch smokes too
    ev2 = {k: v for k, v in ev.items() if len(Path(k).parts) > 1 and Path(k).parts[1].startswith(FOLLOWUP_RUNS)}
    gr2 = {k: v for k, v in gr.items() if k in FOLLOWUP_GUARDRAILS}
    for tag in FOLLOWUP_GUARDRAILS:                      # a run with rows and no summary of its own
        rows = f"{root}/results/bench/{tag}_guardrail.rows.jsonl"
        if tag not in gr2 and os.path.exists(rows):
            t = generated(rows)
            if t:
                gr2[tag] = round(t / 3600, 2)
    ce, co = ceilings(root), controls(root)
    tot = {"training_h": round(sum(tr2.values()), 1), "evals_h": round(sum(ev2.values()), 1),
           "guardrails_h": round(sum(gr2.values()), 1), "ceilings_h": round(sum(ce.values()), 1),
           "controls_h": round(sum(co.values()), 1)}
    tot["total_h"] = round(sum(tot.values()), 1)
    return {"scope": "after the campaign, 2026-09-28/29: the partner-free bridge and the two replicates, their "
                     "evaluations, the stored-token and partner-free guard-rails (generated rows only), the "
                     "teacher ceilings and the content controls of the 9B arms",
            "not_counted": "model loading, self-tests, the building of the stored tokens, and the judges "
                           "(language-model agents, not run on this machine's GPU)",
            "totals": tot, "training": tr2, "evals": ev2, "guardrails": gr2, "ceilings": ce, "controls": co}


PSILM2_TRAINING = ("stage2c_", "stage2_qwen35")
PSILM2_GUARDRAILS = ("const_", "dual_qwen35_")


def scoped(tr, ev, gr, rs, exclude):
    """The PsiLM-2 paper's runs, by name prefix, minus --exclude."""
    def keep_run(name):
        # the Ternary Bonsai bridges (stage2c_bonsai*) are the chat app's, not the paper's
        return (name.startswith(PSILM2_TRAINING) and "bonsai" not in name and name not in exclude
                and not name.startswith(FOLLOWUP_RUNS))
    tr2 = {k: v for k, v in tr.items() if keep_run(k)}
    ev2 = {k: v for k, v in ev.items() if keep_run(Path(k).parts[1] if len(Path(k).parts) > 1 else "")}
    gr2 = {k: v for k, v in gr.items() if k.startswith(PSILM2_GUARDRAILS) and "bonsai" not in k
           and k not in FOLLOWUP_GUARDRAILS
           and not any(k.startswith("const_" + x[len("stage2c_"):]) for x in exclude if x.startswith("stage2c_"))}
    rs2 = {k: v for k, v in rs.items() if k.startswith(PSILM2_GUARDRAILS) or k == "guardrail_qwen35"}
    tot = {"training_h": round(sum(tr2.values()), 1), "evals_h": round(sum(ev2.values()), 1),
           "guardrails_h": round(sum(gr2.values()), 1), "rescores_h": round(sum(rs2.values()), 1)}
    tot["total_h"] = round(sum(tot.values()), 1)
    return {"scope": "training runs stage2c_* and stage2_qwen35, their eval_*.log files, guard-rails const_* and "
                     "dual_qwen35_*, and the KL-read rescoring of those guard-rails and of guardrail_qwen35",
            "excluded_runs": sorted(exclude), "totals": tot,
            "training": tr2, "evals": ev2, "guardrails": gr2, "rescores": rs2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/compute_summary.json")
    ap.add_argument("--exclude", default="", help="comma-separated run names left out of the psilm2 block (runs still in progress)")
    a = ap.parse_args()
    tr, ev, gr, rs = training(a.root), evals(a.root), guardrails(a.root), rescores(a.root)
    tot = {"training_h": round(sum(tr.values()), 1), "evals_h": round(sum(ev.values()), 1),
           "guardrails_h": round(sum(gr.values()), 1), "rescores_h": round(sum(rs.values()), 1)}
    tot["total_h"] = round(sum(tot.values()), 1)
    exclude = {x for x in a.exclude.split(",") if x}
    out = {"totals": tot, "training": tr, "evals": ev, "guardrails": gr, "rescores": rs,
           "psilm2": scoped(tr, ev, gr, rs, exclude),
           "psilm2_followup": followup(tr, ev, gr, a.root),
           "excluded": "value-neuron collection and probes, data building, 0.5B ablations, the partner fine-tune"}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"training {tot['training_h']} h over {len(tr)} runs | evals {tot['evals_h']} h over {len(ev)} | "
          f"guard-rails {tot['guardrails_h']} h over {len(gr)} | KL rescoring {tot['rescores_h']} h over "
          f"{len(rs)} | total {tot['total_h']} h -> {a.out}")
    p = out["psilm2"]["totals"]
    print(f"psilm2 scope: training {p['training_h']} h over {len(out['psilm2']['training'])} runs | "
          f"evals {p['evals_h']} h | guard-rails {p['guardrails_h']} h over {len(out['psilm2']['guardrails'])} | "
          f"KL rescoring {p['rescores_h']} h | "
          f"total {p['total_h']} h (excluded: {', '.join(sorted(exclude)) or 'none'})")
    f = out["psilm2_followup"]["totals"]
    print(f"psilm2 follow-up: training {f['training_h']} h | evals {f['evals_h']} h | guard-rails "
          f"{f['guardrails_h']} h | ceilings {f['ceilings_h']} h | controls {f['controls_h']} h | total {f['total_h']} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
