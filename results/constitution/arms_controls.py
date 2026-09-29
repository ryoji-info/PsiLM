#!/usr/bin/env python3
"""One table of the temperature control and the content controls, every 9B arm.

Reads what eval/teacher_ceiling.py and eval/constitution_compress.py wrote beside
each arm's checkpoint (results/stage2c_qwen35_<arm>/teacher_ceiling.json and
compress.json) and writes results/constitution/arms_controls.json. Nothing is
computed here that those files do not hold, except the share of the raw CE gain
that survives the cross-fitted temperature.

  python results/constitution/arms_controls.py            # writes the json, prints the table
  python results/constitution/arms_controls.py --check    # exits 1 unless every arm is complete and ok
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["all", "allplain", "vn10e", "vn10ebot", "match0", "match1", "match2",
        "vn5e", "match205", "vn5", "vn1e", "match41", "vn"]
SPLITS = {"test": "constitution_test_qwen35", "helpful": "constitution_helpful_test_qwen35"}
WHAT = {
    "all": "full width, fine-tuned partner",
    "allplain": "full width, plain partner",
    "vn10e": "value neurons, top 410, energy parity",
    "vn10ebot": "value ranking's bottom 410, energy parity",
    "match0": "magnitude-matched random 410, set 0",
    "match1": "magnitude-matched random 410, set 1",
    "match2": "magnitude-matched random 410, set 2",
    "vn5e": "value neurons, top 205, energy parity",
    "match205": "magnitude-matched random 205",
    "vn5": "value neurons, top 205, cap 0.2",
    "vn1e": "value neurons, top 41, energy parity",
    "match41": "magnitude-matched random 41",
    "vn": "value neurons, top 41, cap 0.2",
}


TAUS = [round(0.30 + 0.05 * i, 2) for i in range(27)]      # eval/teacher_ceiling.py's grid
PAIRS = [("all", ["allplain"]), ("vn10e", ["match0", "match1", "match2"]), ("vn10e", ["vn10ebot"]),
         ("vn5e", ["match205"]), ("vn1e", ["match41"])]
DRAWS = 2000


def curves(tag):
    """{split: (sources, base CE [items x taus], psilm CE [items x taus])}, whole continuation."""
    rows = [json.loads(ln) for ln in (ROOT / f"results/stage2c_qwen35_{tag}/teacher_ceiling.rows.jsonl")
            .read_text().splitlines() if ln.strip()]
    out = {}
    for short, split in SPLITS.items():
        r = sorted((x for x in rows if x["split"] == split), key=lambda x: x["source"])
        ce = lambda arm: np.array([[x[arm]["all"][f"ce@{t}"] for t in TAUS] for x in r])
        out[short] = ([x["source"] for x in r], ce("base"), ce("psilm"))
    return out


def best_tau_gain(c, short, idx_eval, idx_pick):
    """Base minus psilm, each at the temperature that is best for it on the OTHER split."""
    other = "helpful" if short == "test" else "test"
    _, b, p = c[short]
    _, bo, po = c[other]
    kb, kp = bo[idx_pick].mean(0).argmin(), po[idx_pick].mean(0).argmin()
    return b[idx_eval, kb].mean() - p[idx_eval, kp].mean()


def resampled(cs, short, seed=0):
    """DRAWS paired resamples of the items; every arm in `cs` sees the same draw."""
    rng = np.random.default_rng(seed)
    other = "helpful" if short == "test" else "test"
    n, m = len(cs[0][short][0]), len(cs[0][other][0])
    got = np.empty((DRAWS, len(cs)))
    for d in range(DRAWS):
        ie, ip = rng.integers(0, n, n), rng.integers(0, m, m)
        got[d] = [best_tau_gain(c, short, ie, ip) for c in cs]
    return got


def ci(v):
    """Six decimals: a value rounded here and again where it is printed can land on the wrong digit."""
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def variant(comp, name):
    for v in comp["variants"]:
        if v["variant"] == name:
            return v
    raise SystemExit(f"{comp['ckpt']}: no {name} variant")


def arm(tag):
    d = ROOT / f"results/stage2c_qwen35_{tag}"
    meta = json.loads((d / "bridges.npz.meta").read_text())
    ceil = json.loads((d / "teacher_ceiling.json").read_text())
    comp = json.loads((d / "compress.json").read_text())
    wd = meta["write_dims"]
    out = {"arm": tag, "what": WHAT[tag], "n_write": len(wd) if isinstance(wd, list) else wd,
           "inj_cap": meta["args"]["inj_cap"], "partner": meta["const_model"], "step": meta["step"],
           "ceiling_ok": bool(ceil.get("ok")), "controls_ok": bool(comp.get("ok")),
           "evaluator_reproduced": all(c["reproduced"] for c in comp["evaluator_checks"].values())}
    trained, base, shuf, zero = (variant(comp, n) for n in ("fp32+native", "base", "shuffle", "softzero"))
    for short, split in SPLITS.items():
        a = ceil["splits"][split]["all"]
        t = a["temperature_crossfit"]
        raw, best = a["ce_gain"], t["ce_gain_at_best_tau"]
        out[short] = {
            "base_ce": a["base_ce"], "teacher_ce": a["teacher_ce"], "psilm_ce": a["psilm_ce"],
            "ce_gain": raw, "ce_gain_ci95": a["ci95"]["ce_gain"],
            "tau_base": t["base_ce"]["tau"], "tau_psilm": t["psilm_ce"]["tau"],
            "base_ce_at_tau": t["base_ce"]["at_tau"], "psilm_ce_at_tau": t["psilm_ce"]["at_tau"],
            "tau_at_edge": bool(t["base_ce"]["at_edge"] or t["psilm_ce"]["at_edge"]),
            "ce_gain_at_best_tau": best,
            "share_surviving": round(best / raw, 4) if abs(raw) > 1e-9 else None,
            "kl_teacher_base": a["base_kl"], "kl_teacher_psilm": a["psilm_kl"],
            "kl_fraction": a["kl_fraction"], "kl_fraction_at_best_tau": t["kl_fraction_at_best_tau"],
            "items_psilm_below_teacher": a["items_psilm_below_teacher"],
            "gate_cont": trained[split]["gate_cont"],
            "kl_write": trained[split]["kl_to_base"],
            "kl_base": base[split]["kl"], "kl_shuffle": shuf[split]["kl"], "kl_softzero": zero[split]["kl"],
            "gain_retained_shuffle": shuf[split]["gain_retained"],
            "gain_retained_softzero": zero[split]["gain_retained"],
            "same_top1_shuffle": shuf[split]["same_top1"], "same_top1_softzero": zero[split]["same_top1"],
        }
    nh = "noharm_heldout"
    out[nh] = {"n": trained[nh]["n"], "gate_cont": trained[nh]["gate_cont"], "kl_write": trained[nh]["kl_to_base"],
               "kl_shuffle": shuf[nh]["kl"], "kl_softzero": zero[nh]["kl"]}
    return out


def intervals(rows):
    """Paired bootstrap of the gain at the best temperature, and of the paper's paired contrasts."""
    cs = {t: curves(t) for t in ARMS}
    for t in ARMS[1:]:
        for short in SPLITS:
            if cs[t][short][0] != cs["all"][short][0]:
                raise SystemExit(f"{t}: not the items of `all` on {short}")
    contrasts = []
    for short in SPLITS:
        got = resampled([cs[t] for t in ARMS], short)
        for k, r in enumerate(rows):
            full = np.arange(len(cs[r["arm"]][short][0]))
            other = np.arange(len(cs[r["arm"]]["helpful" if short == "test" else "test"][0]))
            point = float(best_tau_gain(cs[r["arm"]], short, full, other))
            if abs(point - r[short]["ce_gain_at_best_tau"]) > 2e-5:
                raise SystemExit(f"{r['arm']} {short}: {point:.5f} here, {r[short]['ce_gain_at_best_tau']} recorded")
            one = TAUS.index(1.0)
            raw = float(cs[r["arm"]][short][1][:, one].mean() - cs[r["arm"]][short][2][:, one].mean())
            if abs(raw - r[short]["ce_gain"]) > 2e-5:
                raise SystemExit(f"{r['arm']} {short}: raw gain {raw:.5f} here, {r[short]['ce_gain']} recorded")
            # from the rows, unrounded: what a table should be printed from
            r[short]["ce_gain"], r[short]["ce_gain_at_best_tau"] = round(raw, 6), round(point, 6)
            r[short]["share_surviving"] = round(point / raw, 4) if abs(raw) > 1e-9 else None
            r[short]["ce_gain_at_best_tau_ci95"] = ci(got[:, k])
        for a, bs in PAIRS:
            d = got[:, ARMS.index(a)] - np.mean([got[:, ARMS.index(b)] for b in bs], axis=0)
            pt = (next(r for r in rows if r["arm"] == a)[short]["ce_gain_at_best_tau"]
                  - np.mean([next(r for r in rows if r["arm"] == b)[short]["ce_gain_at_best_tau"] for b in bs]))
            contrasts.append({"split": short, "arm": a, "minus": bs, "difference": round(float(pt), 6),
                              "ci95": ci(d)})
    return contrasts


def table(rows):
    f = lambda x, n=4: "—" if x is None else f"{x:.{n}f}"
    lines = []
    for short in SPLITS:
        lines += [f"\n{short}: CE gain, raw and at each arm's own best temperature (cross-fitted); "
                  "KL of the trained write against three reads",
                  "| arm | dims | cap | base CE | psilm CE | gain | gain at best τ | 95% | share | "
                  "KL(T‖base) | KL(T‖psilm) | write | shuffle | softzero | kept shuf | kept zero |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            s = r[short]
            lines.append(f"| `{r['arm']}` | {r['n_write']} | {r['inj_cap']} | {f(s['base_ce'])} | {f(s['psilm_ce'])} "
                         f"| {f(s['ce_gain'])} | {f(s['ce_gain_at_best_tau'])} "
                         f"| {s['ce_gain_at_best_tau_ci95'][0]:+.3f} to {s['ce_gain_at_best_tau_ci95'][1]:+.3f} "
                         f"| {f(s['share_surviving'], 2)} "
                         f"| {f(s['kl_teacher_base'])} | {f(s['kl_teacher_psilm'])} "
                         f"| {f(s['kl_write'])} | {f(s['kl_shuffle'], 5)} | {f(s['kl_softzero'], 5)} "
                         f"| {f(s['gain_retained_shuffle'], 2)} | {f(s['gain_retained_softzero'], 2)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/constitution/arms_controls.json"))
    a = ap.parse_args()
    rows = [arm(t) for t in ARMS]
    bad = [r["arm"] for r in rows if not (r["ceiling_ok"] and r["controls_ok"] and r["evaluator_reproduced"])]
    edge = [f"{r['arm']}/{s}" for r in rows for s in SPLITS if r[s]["tau_at_edge"]]
    contrasts = intervals(rows)
    if a.check:
        print(f"[arms] {len(rows)} arms, not ok: {bad or 'none'}, a best temperature at the grid's edge: {edge or 'none'}")
        return 1 if bad else 0
    Path(a.out).write_text(json.dumps({
        "what": "temperature control (eval/teacher_ceiling.py, cross-fitted across the two splits) and content "
                "controls (eval/constitution_compress.py: base = write off, shuffle = another prompt's tokens, "
                "softzero = the partner fed zeros) for every 9B constitution arm; 50 items per split",
        "kl_columns": "kl_write is KL(trained || base) as the evaluator reads it; kl_base, kl_shuffle and "
                      "kl_softzero are KL(trained || that read) on the same positions",
        "intervals": f"paired bootstrap over items, {DRAWS} draws; both splits are resampled, so the choice of "
                     "the temperatures is inside the interval",
        "not_ok": bad, "tau_at_edge": edge, "contrasts_at_best_tau": contrasts, "arms": rows}, indent=1) + "\n")
    print(table(rows))
    print("\ngain at the best temperature, paired contrasts")
    for c in contrasts:
        print(f"  {c['split']:8s} {c['arm']} minus {'+'.join(c['minus'])}: {c['difference']:+.4f} "
              f"({c['ci95'][0]:+.4f} to {c['ci95'][1]:+.4f})")
    print(f"\n[arms] not ok: {bad or 'none'}; best temperature at the grid's edge: {edge or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
