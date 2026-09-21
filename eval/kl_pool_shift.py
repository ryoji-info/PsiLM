#!/usr/bin/env python3
"""How far the recorded KLs sit from the corrected read of the coupling.

Until 2026-09-22 the teacher-forced KL pass let the coupling read the prompt AND
the base continuation; every arm's decode reads the prompt alone. Every KL in
results/bench -- the constitution and dual runs of ΨLM-2 and the physics bridges'
leaky and shuffled sweeps alike -- was recorded on that one basis, so comparisons
between arms of one read stand. This
aggregates the rescoring runs (bench_guardrail.py --kl-rescore, driven by
eval/kl_rescore_all.py), which recompute each recorded KL under both reads on the
run's own base continuations:

  recorded   the number in the run's rows file
  full       the legacy read, recomputed now -- must reproduce `recorded`
  prompt     the corrected read

  python3 eval/kl_pool_shift.py

Writes results/constitution/kl_pool_shift.json.
"""
import argparse, glob, json, os, statistics as st
from pathlib import Path

SUFFIX = "_klpool_guardrail.rows.jsonl"
# contrasts the paper states as ratios of mean KL: (label, dataset, numerator tags, denominator tags)
CONTRASTS = [
    ("probe-best 410 / four matched 410 controls", "mmlu", ["const_qwen35_vn10e"],
     ["const_qwen35_vn10ebot", "const_qwen35_match0_rg", "const_qwen35_match1_rg", "const_qwen35_match2_rg"]),
    ("value neurons (41) at parity / matched 41", "mmlu", ["const_qwen35_vn1e"], ["const_qwen35_match41"]),
    ("top 5% (205) at parity / matched 205", "mmlu", ["const_qwen35_vn5e"], ["const_qwen35_match205"]),
    ("probe-best 410 / whole stream", "mmlu", ["const_qwen35_vn10e"], ["const_qwen35_all"]),
    ("dual stack / constitution alone", "mmlu", ["dual_qwen35_both"], ["const_qwen35_all"]),
    ("probe-best 410 / probe-worst 410", "redteam", ["const_qwen35_vn10e"], ["const_qwen35_vn10ebot"]),
]


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r, i = [0.0] * len(v), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bench")
    ap.add_argument("--out", default="results/constitution/kl_pool_shift.json")
    a = ap.parse_args()
    runs, means = {}, {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*" + SUFFIX))):
        tag = os.path.basename(f)[:-len(SUFFIX)]
        by = {}
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["kl_full"]["mean"] is not None:
                    by.setdefault((r["dataset"], r["arm"]), []).append(r)
        for (ds, arm), rs in sorted(by.items()):
            rec = [r["kl_recorded"]["mean"] for r in rs]
            full = [r["kl_full"]["mean"] for r in rs]
            prm = [r["kl_prompt"]["mean"] for r in rs]
            ratios = [p / q for p, q in zip(prm, full) if q > 0]
            cell = {"n": len(rs),
                    "recorded_mean": round(st.mean(rec), 6), "full_mean": round(st.mean(full), 6),
                    "prompt_mean": round(st.mean(prm), 6),
                    "prompt_over_full": round(st.mean(prm) / st.mean(full), 4) if st.mean(full) > 0 else None,
                    "per_item_ratio_median": round(st.median(ratios), 4) if ratios else None,
                    "per_item_ratio_min": round(min(ratios), 4) if ratios else None,
                    "per_item_ratio_max": round(max(ratios), 4) if ratios else None,
                    "items_prompt_higher": sum(p > q for p, q in zip(prm, full)),
                    "spearman_prompt_full": round(spearman(prm, full), 4) if len(rs) > 2 and spearman(prm, full) is not None else None,
                    # the check that the rescoring rebuilt the recorded run
                    "max_abs_full_minus_recorded": max(abs(x - y) for x, y in zip(full, rec)),
                    "items_reproduced_to_1e-6": sum(abs(x - y) <= 1e-6 for x, y in zip(full, rec))}
            runs.setdefault(tag, {})[f"{ds}:{arm}"] = cell
            if arm == "psilm":
                means[(tag, ds)] = (cell["full_mean"], cell["prompt_mean"])
    contrasts = []
    for label, ds, num, den in CONTRASTS:
        if all((t, ds) in means for t in num + den):
            f_ = st.mean(means[(t, ds)][0] for t in num) / st.mean(means[(t, ds)][0] for t in den)
            p_ = st.mean(means[(t, ds)][1] for t in num) / st.mean(means[(t, ds)][1] for t in den)
            contrasts.append({"contrast": label, "dataset": ds, "full": round(f_, 2), "prompt": round(p_, 2)})
    cells = [c for r in runs.values() for c in r.values()]
    secs = sum(r.get("sec", 0.0) for f in glob.glob(os.path.join(a.dir, "*" + SUFFIX))
               for r in (json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()))
    out = {"convention": __doc__.split("\n\n")[1].replace("\n", " "),
           "n_runs": len(runs), "n_kls": sum(c["n"] for c in cells),
           "all_reproduced": all(c["items_reproduced_to_1e-6"] == c["n"] for c in cells) if cells else None,
           "timed_gpu_hours": round(secs / 3600, 2),
           "prompt_over_full_range": [min(c["prompt_over_full"] for c in cells if c["prompt_over_full"]),
                                      max(c["prompt_over_full"] for c in cells if c["prompt_over_full"])] if cells else None,
           "contrasts": contrasts, "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'run':34s} {'dataset:arm':22s} {'n':>4s} {'recorded':>10s} {'prompt':>10s} {'p/f':>7s} "
          f"{'item med':>8s} {'rho':>6s} {'repro':>7s}")
    for tag, r in runs.items():
        for k, c in r.items():
            print(f"{tag:34s} {k:22s} {c['n']:4d} {c['recorded_mean']:10.6f} {c['prompt_mean']:10.6f} "
                  f"{str(c['prompt_over_full']):>7s} {str(c['per_item_ratio_median']):>8s} "
                  f"{str(c['spearman_prompt_full']):>6s} {c['items_reproduced_to_1e-6']:>3d}/{c['n']:<3d}")
    for c in contrasts:
        print(f"{c['contrast']} ({c['dataset']}): {c['full']}x recorded -> {c['prompt']}x corrected")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
