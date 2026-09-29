#!/usr/bin/env python3
"""One table of the temperature control and the content controls, every arm of a backbone.

Reads what eval/teacher_ceiling.py and eval/constitution_compress.py wrote beside
each arm's checkpoint (results/stage2c_qwen35_<arm>/teacher_ceiling.json and
compress.json) and writes results/constitution/arms_controls.json. Nothing is
computed here that those files do not hold, except the share of the raw CE gain
that survives the cross-fitted temperature.

  python results/constitution/arms_controls.py            # writes the json, prints the table
  python results/constitution/arms_controls.py --check    # exits 1 unless every arm is complete and ok
  python results/constitution/arms_controls.py --set qwen0.5b
      the nine 0.5B bridges (results/constitution/tempcontrol_qwen0.5b_preregistration.json),
      100 items a split, into results/constitution/arms_controls_qwen0.5b.json. Only the
      full-width one has content controls there; the contrasts are given as evaluated
      (temperature 1) beside their value at the best temperatures, from the same draws.
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
RUN = "results/stage2c_qwen35_{}"
OUT = "results/constitution/arms_controls.json"
ITEMS = 50
BACKBONE = "9B"
AS_EVALUATED = False        # the contrasts at temperature 1 too (a set that asks for them)

SETS = {"qwen0.5b": {
    "ARMS": ["all", "allplain", "vn", "plainpartner", "vn5", "magmatch", "magmatch1", "magmatch2", "rand"],
    "SPLITS": {"test": "constitution_test_qwen0.5b", "helpful": "constitution_helpful_test_qwen0.5b"},
    "WHAT": {"all": "full width, fine-tuned partner", "allplain": "full width, plain partner",
             "vn": "value neurons, top 9", "plainpartner": "value neurons, top 9, plain partner",
             "vn5": "value neurons, top 45", "magmatch": "magnitude-matched random 9, draw 0",
             "magmatch1": "magnitude-matched random 9, draw 1", "magmatch2": "magnitude-matched random 9, draw 2",
             "rand": "random 9"},
    # the contrasts of results/constitution/tempcontrol_qwen0.5b_preregistration.json, C1 to C5,
    # and the value neurons against each matched draw (reported beside C3, not ruled)
    "PAIRS": [("plainpartner", ["vn"]), ("allplain", ["all"]), ("vn", ["magmatch", "magmatch1", "magmatch2"]),
              ("vn", ["rand"]), ("vn5", ["vn"]), ("vn", ["magmatch"]), ("vn", ["magmatch1"]), ("vn", ["magmatch2"])],
    "RUN": "results/stage2c_qwen0.5b_{}", "OUT": "results/constitution/arms_controls_qwen0.5b.json",
    "ITEMS": 100, "BACKBONE": "0.5B", "AS_EVALUATED": True}}


def curves(tag):
    """{split: (sources, base CE [items x taus], psilm CE [items x taus])}, whole continuation."""
    rows = [json.loads(ln) for ln in (ROOT / RUN.format(tag) / "teacher_ceiling.rows.jsonl")
            .read_text().splitlines() if ln.strip()]
    if AS_EVALUATED:       # a rows file keeps the rows of a run that was started again under another id
        ident = json.loads((ROOT / RUN.format(tag) / "teacher_ceiling.json").read_text())["id"]
        rows = [x for x in rows if x.get("id") == ident]
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


def raw_gain(c, short, idx_eval, idx_pick=None):
    """Base minus psilm as evaluated: both at temperature 1."""
    one = TAUS.index(1.0)
    _, b, p = c[short]
    return b[idx_eval, one].mean() - p[idx_eval, one].mean()


def resampled(cs, short, seed=0, gain=best_tau_gain):
    """DRAWS paired resamples of the items; every arm in `cs` sees the same draw, and so does
    every `gain` asked of the same seed."""
    rng = np.random.default_rng(seed)
    other = "helpful" if short == "test" else "test"
    n, m = len(cs[0][short][0]), len(cs[0][other][0])
    got = np.empty((DRAWS, len(cs)))
    for d in range(DRAWS):
        ie, ip = rng.integers(0, n, n), rng.integers(0, m, m)
        got[d] = [gain(c, short, ie, ip) for c in cs]
    return got


SHARES = np.round(np.arange(-3.0, 5.0005, 0.001), 3)


def share_ci(md, vd):
    """The 95% interval of md / vd that does not divide draw by draw: the r for which the
    interval of (md - r * vd) includes zero. An end at the range's own end is open (None)."""
    d = md[None, :] - SHARES[:, None] * vd[None, :]
    lo, hi = np.percentile(d, 2.5, axis=1), np.percentile(d, 97.5, axis=1)
    ok = SHARES[(lo <= 0) & (0 <= hi)]
    if not len(ok):
        return None
    return [None if ok.min() == SHARES[0] else float(ok.min()), None if ok.max() == SHARES[-1] else float(ok.max())]


def ci(v):
    """Six decimals: a value rounded here and again where it is printed can land on the wrong digit."""
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def variant(comp, name):
    for v in comp["variants"]:
        if v["variant"] == name:
            return v
    raise SystemExit(f"{comp['ckpt']}: no {name} variant")


def shown(path: str) -> str:
    """A partner as a table may print it: relative to the repository if it is inside it."""
    return path[len(str(ROOT)) + 1:] if path.startswith(str(ROOT) + "/") else path


def arm_ceiling_only(tag, d, meta, ceil):
    """An arm with the temperature control and no content controls."""
    wd = meta["write_dims"]
    made = {split: (ceil["splits"][split]["checks"].get("evaluator") or {}).get("reproduced")
            for split in SPLITS.values()}
    out = {"arm": tag, "what": WHAT[tag], "n_write": len(wd) if isinstance(wd, list) else wd,
           "inj_cap": meta["args"]["inj_cap"], "partner": shown(meta["const_model"]), "step": meta["step"],
           "ceiling_ok": bool(ceil.get("ok")), "controls_ok": None,
           # None where the evaluator was never run on a split: nothing to reproduce there
           "evaluator_reproduced": all(v for v in made.values() if v is not None),
           "evaluator_run_on": sorted(short for short, split in SPLITS.items() if made[split] is not None),
           # the re-decode of two items a split: recorded and reported, not part of ok
           "greedy_decode_diverged": sum(g["first_divergence"] is not None for split in SPLITS.values()
                                         for g in ceil["splits"][split]["checks"].get("greedy_decode", []))}
    for short, split in SPLITS.items():
        if ceil["splits"][split]["n"] != ITEMS:
            raise SystemExit(f"{tag} {short}: {ceil['splits'][split]['n']} items, the set is of {ITEMS}")
        a = ceil["splits"][split]["all"]
        t, ts = a["temperature_crossfit"], a["temperature"]
        raw, best = a["ce_gain"], t["ce_gain_at_best_tau"]
        out[short] = {
            "base_ce": a["base_ce"], "teacher_ce": a["teacher_ce"], "psilm_ce": a["psilm_ce"],
            "ce_gain": raw, "ce_gain_ci95": a["ci95"]["ce_gain"],
            "tau_base": t["base_ce"]["tau"], "tau_psilm": t["psilm_ce"]["tau"],
            "base_ce_at_tau": t["base_ce"]["at_tau"], "psilm_ce_at_tau": t["psilm_ce"]["at_tau"],
            "tau_at_edge": bool(t["base_ce"]["at_edge"] or t["psilm_ce"]["at_edge"]),
            "tau_base_at_edge": bool(t["base_ce"]["at_edge"]), "tau_psilm_at_edge": bool(t["psilm_ce"]["at_edge"]),
            # beside the cross-fit, nothing is read from these: both sides at their in-split best
            "tau_base_in_split": ts["base_ce"]["tau"], "tau_psilm_in_split": ts["psilm_ce"]["tau"],
            "ce_gain_at_in_split_tau": ts["ce_gain_at_best_tau"],
            "crossfit_cost_base": round(t["base_ce"]["at_tau"] - ts["base_ce"]["at_tau"], 6),
            "crossfit_cost_psilm": round(t["psilm_ce"]["at_tau"] - ts["psilm_ce"]["at_tau"], 6),
            "ce_gain_at_best_tau": best,
            "share_surviving": round(best / raw, 4) if abs(raw) > 1e-9 else None,
            "kl_teacher_base": a["base_kl"], "kl_teacher_psilm": a["psilm_kl"],
            "kl_fraction": a["kl_fraction"], "kl_fraction_at_best_tau": t["kl_fraction_at_best_tau"],
            "items_psilm_below_teacher": a["items_psilm_below_teacher"],
        }
    return out


def arm(tag):
    d = ROOT / RUN.format(tag)
    meta = json.loads((d / "bridges.npz.meta").read_text())
    ceil = json.loads((d / "teacher_ceiling.json").read_text())
    if AS_EVALUATED:                       # a set read for its temperature control alone
        return arm_ceiling_only(tag, d, meta, ceil)
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
        raw_draws = resampled([cs[t] for t in ARMS], short, gain=raw_gain) if AS_EVALUATED else None
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
            if AS_EVALUATED:                # the same contrast at temperature 1, on the same draws
                full = np.arange(len(cs[a][short][0]))
                e = raw_draws[:, ARMS.index(a)] - np.mean([raw_draws[:, ARMS.index(b)] for b in bs], axis=0)
                contrasts[-1]["as_evaluated"] = {
                    "difference": round(float(raw_gain(cs[a], short, full)
                                              - np.mean([raw_gain(cs[b], short, full) for b in bs])), 6),
                    "ci95": ci(e)}
                contrasts[-1]["gain_draws_below_zero"] = round(float((d < 0).mean()), 4)
                if len(bs) > 1:             # the magnitude share of the split, on the same draws
                    vd, md = got[:, ARMS.index(a)], np.mean([got[:, ARMS.index(b)] for b in bs], axis=0)
                    ve, me = raw_draws[:, ARMS.index(a)], np.mean([raw_draws[:, ARMS.index(b)] for b in bs], axis=0)
                    at = lambda arm_, k: next(r for r in rows if r["arm"] == arm_)[short][k]      # noqa: E731
                    mm = float(np.mean([at(b, "ce_gain_at_best_tau") for b in bs]))
                    mr = float(np.mean([at(b, "ce_gain") for b in bs]))
                    vb, vr = at(a, "ce_gain_at_best_tau"), at(a, "ce_gain")     # a gain of 0.000000: no share
                    contrasts[-1]["magnitude_share"] = {
                        "at_best_tau": {"share": round(mm / vb, 4) if vb else None, "ci95": share_ci(md, vd)},
                        "as_evaluated": {"share": round(mr / vr, 4) if vr else None, "ci95": share_ci(me, ve)},
                        "matched_mean_gain": round(mm, 6), "matched_mean_gain_ci95": ci(md),
                        "interval": "the r for which the interval of (matched mean gain - r * the first arm's "
                                    "gain) includes zero, r from -3 to 5 in steps of 0.001; null is an open end"}
    return contrasts


def table_ceiling_only(rows):
    f = lambda x, n=4: "—" if x is None else f"{x:.{n}f}"
    lines = []
    for short in SPLITS:
        lines += [f"\n{short}: CE gain, as evaluated and at each system's own best temperature (cross-fitted)",
                  "| arm | dims | base CE | psilm CE | gain | τ base | τ psilm | gain at best τ | 95% | share kept | "
                  "KL(T‖base) | KL(T‖psilm) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            s = r[short]
            edge = lambda k: "*" if s[k] else ""
            lines.append(f"| `{r['arm']}` | {r['n_write']} | {f(s['base_ce'])} | {f(s['psilm_ce'])} | {f(s['ce_gain'])} "
                         f"| {s['tau_base']}{edge('tau_base_at_edge')} | {s['tau_psilm']}{edge('tau_psilm_at_edge')} "
                         f"| {f(s['ce_gain_at_best_tau'])} "
                         f"| {s['ce_gain_at_best_tau_ci95'][0]:+.4f} to {s['ce_gain_at_best_tau_ci95'][1]:+.4f} "
                         f"| {f(s['share_surviving'], 2)} | {f(s['kl_teacher_base'])} | {f(s['kl_teacher_psilm'])} |")
    return "\n".join(lines) + "\n(* a best temperature at the edge of the grid)"


def table(rows):
    if AS_EVALUATED:
        return table_ceiling_only(rows)
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
    ap.add_argument("--set", default="qwen35", choices=["qwen35", *SETS])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    globals().update(SETS.get(a.set, {}))
    a.out = a.out or str(ROOT / OUT)
    rows = [arm(t) for t in ARMS]
    bad = [r["arm"] for r in rows if not (r["ceiling_ok"] and r["controls_ok"] is not False
                                          and r["evaluator_reproduced"])]
    edge = [f"{r['arm']}/{s}" for r in rows for s in SPLITS if r[s]["tau_at_edge"]]
    contrasts = intervals(rows)
    if a.check:
        print(f"[arms] {len(rows)} arms, not ok: {bad or 'none'}, a best temperature at the grid's edge: {edge or 'none'}")
        return 1 if bad else 0
    if AS_EVALUATED:
        Path(a.out).write_text(json.dumps({
            "what": f"temperature control (eval/teacher_ceiling.py, cross-fitted across the two splits) for every "
                    f"{BACKBONE} constitution arm; {ITEMS} items per split",
            "preregistration": "results/constitution/tempcontrol_qwen0.5b_preregistration.json",
            "intervals": f"paired bootstrap over items, {DRAWS} draws; both splits are resampled, so the choice "
                         "of the temperatures is inside the interval; every arm and every contrast on the same draws",
            "contrasts": "difference and ci95 are at the best temperatures; as_evaluated is the same contrast at "
                         "temperature 1 on the same draws",
            "not_ok": bad, "tau_at_edge": edge, "contrasts_at_best_tau": contrasts, "arms": rows}, indent=1) + "\n")
        print(table(rows))
        print("\npaired contrasts: at the best temperatures | as evaluated")
        for c in contrasts:
            e = c["as_evaluated"]
            print(f"  {c['split']:8s} {c['arm']} minus {'+'.join(c['minus'])}: {c['difference']:+.4f} "
                  f"({c['ci95'][0]:+.4f} to {c['ci95'][1]:+.4f}) | {e['difference']:+.4f} "
                  f"({e['ci95'][0]:+.4f} to {e['ci95'][1]:+.4f})")
        print(f"\n[arms] not ok: {bad or 'none'}; best temperature at the grid's edge: {edge or 'none'}")
        return 0
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
