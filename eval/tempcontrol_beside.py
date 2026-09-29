#!/usr/bin/env python3
"""Beside the readings of the temperature control at 0.5B: what its intervals are made of.

WRITTEN AFTER THE RUN (2026-09-30). Nothing here is a reading and nothing here changes
one. The readings are results/constitution/tempcontrol_qwen0.5b_reading.json, by the rules
of results/constitution/tempcontrol_qwen0.5b_preregistration.json, as posed; that file says
that a reading which turns out to be badly posed is reported as posed, with what was wrong
with it said beside it. This computes what is said beside them, from the same rows, on the
table's own draws (results/constitution/arms_controls.py, which is imported, not changed):

  temperatures_chosen   how often each temperature is chosen in the 2000 draws
  modes                 the draws' gains by whether the coupled system's temperature is the
                        backbone's or one step of the grid away
  one_temperature       both sides held at ONE temperature, 0.50 to 1.00
  in_split              both sides at their best temperature on the read split itself
  curves                where each mean curve's minimum lies between the grid's points
  the_backbone_alone    what the backbone gains from temperature with no bridge
  other_draws           the intervals under other seeds and more draws
  named                 each named reading as posed, in-split, and at each one temperature

  python eval/tempcontrol_beside.py      # writes results/constitution/tempcontrol_qwen0.5b_beside.json

The labels are the pre-registration's words (eval/tempcontrol_reading.py, imported). Where
they are applied to something other than the cross-fitted gains they are descriptions, not
readings.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NINE = ["vn", "plainpartner", "magmatch", "magmatch1", "magmatch2", "rand"]
MATCHED = ["magmatch", "magmatch1", "magmatch2"]
COMMON = [round(0.50 + 0.05 * i, 2) for i in range(11)]
NAMES = {("plainpartner", ("vn",)): "C1_plain_partner_at_nine", ("allplain", ("all",)): "C2_plain_partner_at_full_width",
         ("vn", tuple(MATCHED)): "C3_identity_at_nine", ("vn", ("rand",)): "C4_value_neurons_against_random",
         ("vn5", ("vn",)): "C5_width", ("vn", ("magmatch",)): "vn minus magmatch",
         ("vn", ("magmatch1",)): "vn minus magmatch1", ("vn", ("magmatch2",)): "vn minus magmatch2"}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ac = module("results/constitution/arms_controls.py", "arms_controls")
vars(ac).update(ac.SETS["qwen0.5b"])
rd = module("eval/tempcontrol_reading.py", "tempcontrol_reading")
TAUS, ARMS, PAIRS = ac.TAUS, ac.ARMS, ac.PAIRS
ONE = TAUS.index(1.0)
r6 = lambda x: round(float(x), 6)                                                   # noqa: E731


def other(short):
    return "helpful" if short == "test" else "test"


def draws(n, m, seed=0, k=None):
    """The table's draws (arms_controls.resampled): the read split's indices, then the other's."""
    rng = np.random.default_rng(seed)
    k = k or ac.DRAWS
    ie, ip = np.empty((k, n), int), np.empty((k, m), int)
    for d in range(k):
        ie[d], ip[d] = rng.integers(0, n, n), rng.integers(0, m, m)
    return ie, ip


def crossfit(cs, short, ie, ip):
    """Per arm: the draws' gains, and the index of the temperature chosen for each side."""
    out = {}
    for t in ARMS:
        _, b, p = cs[t][short]
        _, bo, po = cs[t][other(short)]
        kb, kp = bo[ip].mean(1).argmin(1), po[ip].mean(1).argmin(1)
        d = np.arange(len(ie))
        out[t] = (b[ie].mean(1)[d, kb] - p[ie].mean(1)[d, kp], kb, kp)
    return out


def contrast(g, a, bs):
    return g[a] - np.mean([g[b] for b in bs], axis=0)


def as_evaluated(table, short, a, bs):
    c = [x for x in table["contrasts_at_best_tau"] if x["split"] == short and x["arm"] == a and x["minus"] == list(bs)]
    return c[0]["as_evaluated"]


def label(table, short, a, bs, point, interval):
    """The pre-registration's word for a contrast with this interval, as the reading gives it."""
    e = as_evaluated(table, short, a, bs)
    name = NAMES[(a, tuple(bs))]
    null = (name in rd.NULL_AS_EVALUATED) if short == rd.READ else not rd.excludes_zero(e["ci95"])
    return rd.read_contrast({"difference": point, "ci95": interval, "as_evaluated": e}, null)["reading"]


def described(table, short, point, g):
    """Arms, contrasts and the split for one way of choosing the temperatures: `point` the gains
    on the items as they are, `g` the draws' gains."""
    arms = {t: {"gain": r6(point[t]), "ci95": ac.ci(g[t]), "label": rd.gain_reading(ac.ci(g[t]))} for t in ARMS}
    cons = {}
    for a, bs in PAIRS:
        d, pt = contrast(g, a, bs), point[a] - np.mean([point[b] for b in bs])
        cons[NAMES[(a, tuple(bs))]] = {"difference": r6(pt), "ci95": ac.ci(d), "draws_below_zero": int((d < 0).sum()),
                                       "label": label(table, short, a, bs, r6(pt), ac.ci(d))}
    md, mm = np.mean([g[m] for m in MATCHED], axis=0), float(np.mean([point[m] for m in MATCHED]))
    share = round(mm / point["vn"], 4) if round(float(point["vn"]), 6) else None
    c3 = cons["C3_identity_at_nine"]["label"]
    per = [cons[f"vn minus {m}"]["label"] for m in MATCHED]
    lab = rd.split_reading(share, arms["vn"]["ci95"][0] > 0, ac.ci(md), c3)
    interval = ac.share_ci(md, g["vn"])
    if lab is None:
        x = round(100 * share, 1)
        closed = interval is not None and None not in interval and 0 <= interval[0] and interval[1] <= 1
        lab = f"{x}% magnitude, {round(100 - x, 1)}% identity" + (
            "" if closed else "; the size of the split is not determined (its interval is not within 0 to 1)")
    sign = ("not firm at best temperature" if c3 != "survives" else
            "firm at best temperature" if per.count("survives") == 3 else
            f"C3 survives, and {per.count('survives')} of the three per-draw intervals exclude zero with the as-evaluated sign")
    return {"arms": arms, "contrasts": cons,
            "split": {"magnitude_share": share, "ci95": interval, "matched_mean_gain": r6(mm),
                      "matched_mean_gain_ci95": ac.ci(md), "label": lab},
            "identity_sign": sign,
            "value_neurons_small_signal": "survives" if arms["vn"]["ci95"][0] > 0 else arms["vn"]["label"]}


def counts(k):
    return {str(TAUS[i]): int(n) for i, n in enumerate(np.bincount(k, minlength=len(TAUS))) if n}


def explained(g, kb, kp):
    """The share of the draws' variance that lies between the (backbone, coupled) temperature pairs."""
    pair = kb * 100 + kp
    between = sum((pair == q).sum() * (g[pair == q].mean() - g.mean()) ** 2 for q in np.unique(pair))
    return round(float(between / ((g - g.mean()) ** 2).sum()), 4)


def group(g, mask):
    if not mask.any():
        return {"draws": 0}
    lo, hi = np.percentile(g[mask], [2.5, 97.5])
    return {"draws": int(mask.sum()), "mean": r6(g[mask].mean()), "below_zero": int((g[mask] < 0).sum()),
            "p2.5": r6(lo), "p97.5": r6(hi)}


def vertex(c):
    """Grid minimum, and the minimum of the parabola through it and its two neighbours."""
    k = int(c.argmin())
    out = {"grid_temperature": TAUS[k], "grid_minimum": r6(c[k])}
    if 0 < k < len(TAUS) - 1:
        den = c[k + 1] - 2 * c[k] + c[k - 1]
        v = c[k] - (c[k + 1] - c[k - 1]) ** 2 / (8 * den)
        out.update(vertex_temperature=round(TAUS[k] + 0.05 * float((c[k - 1] - c[k + 1]) / (2 * den)), 5),
                   vertex_value=r6(v), grid_minimum_above_the_vertex=r6(c[k] - v),
                   one_step_down=r6(c[k - 1] - c[k]), one_step_up=r6(c[k + 1] - c[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/constitution/tempcontrol_qwen0.5b_beside.json"))
    a = ap.parse_args()
    table = json.loads((ROOT / ac.OUT).read_text())
    reading = json.loads((ROOT / "results/constitution/tempcontrol_qwen0.5b_reading.json").read_text())
    rows = {r["arm"]: r for r in table["arms"]}
    cs = {t: ac.curves(t) for t in ARMS}
    out = {
        "written": "2026-09-30, AFTER the run. Nothing here is a reading and no reading is changed: the readings are "
                   "results/constitution/tempcontrol_qwen0.5b_reading.json, as posed by "
                   "results/constitution/tempcontrol_qwen0.5b_preregistration.json (commit 32aee47). This is what is "
                   "said beside them.",
        "from": "the rows of the run (results/stage2c_qwen0.5b_<arm>/teacher_ceiling.rows.jsonl) and the table's own "
                f"draws: numpy default_rng(0), {ac.DRAWS} draws, the read split's items and then the other split's. "
                "Labels are the pre-registration's words; applied to anything but the cross-fitted gains they "
                "describe, they are not readings.",
        "reproduces_the_table": None, "temperatures_chosen": {}, "modes": {}, "one_temperature": {}, "in_split": {},
        "curves": {}, "the_backbone_alone": {}, "other_draws": {}, "named": {}}
    worst = 0.0
    for short in ac.SPLITS:
        n, m = len(cs["all"][short][0]), len(cs["all"][other(short)][0])
        ie, ip = draws(n, m)
        cf = crossfit(cs, short, ie, ip)
        g = {t: cf[t][0] for t in ARMS}
        for t in ARMS:                                   # these are the table's draws, or nothing below is
            worst = max(worst, *(abs(x - y) for x, y in zip(ac.ci(g[t]), rows[t][short]["ce_gain_at_best_tau_ci95"])))
        for c in table["contrasts_at_best_tau"]:
            if c["split"] == short:
                worst = max(worst, *(abs(x - y) for x, y in zip(ac.ci(contrast(g, c["arm"], c["minus"])), c["ci95"])))

        # --- the temperatures the draws choose
        out["temperatures_chosen"][short] = {"chosen_on": other(short), "backbone": counts(cf["all"][1]),
                                             **{t: counts(cf[t][2]) for t in ARMS}}
        # --- the two modes of a nine-coordinate arm's draws
        md = {}
        for t in NINE:
            gt, kb, kp = cf[t]
            md[t] = {"variance_between_temperature_pairs": explained(gt, kb, kp),
                     "coupled_at_the_backbones_temperature": group(gt, kp == kb),
                     "coupled_one_step_above": group(gt, kp == kb + 1),
                     "coupled_one_step_below": group(gt, kp == kb - 1),
                     "other": int((abs(kp - kb) > 1).sum())}
        for t in ("all", "allplain", "vn5"):
            md[t] = {"variance_between_temperature_pairs": explained(*cf[t])}
        for a_, bs in PAIRS:
            d = contrast(g, a_, bs)
            same = np.all([cf[b][2] == cf[a_][2] for b in bs], axis=0)
            md[NAMES[(a_, tuple(bs))]] = {"its_arms_at_one_coupled_temperature": group(d, same),
                                          "otherwise": group(d, ~same)}
        out["modes"][short] = md

        # --- both sides at one temperature
        eb = {t: cs[t][short][1][ie].mean(1) for t in ARMS}          # the backbone, draws x temperatures
        ep = {t: cs[t][short][2][ie].mean(1) for t in ARMS}
        one = {}
        for tau in COMMON:
            k = TAUS.index(tau)
            point = {t: cs[t][short][1][:, k].mean() - cs[t][short][2][:, k].mean() for t in ARMS}
            one[str(tau)] = described(table, short, point, {t: eb[t][:, k] - ep[t][:, k] for t in ARMS})
        side = lambda c: "above" if c[0] > 0 else "below" if c[1] < 0 else "includes zero"      # noqa: E731
        summary = {name: {s: [tau for tau in COMMON if side(one[str(tau)]["contrasts"][name]["ci95"]) == s]
                          for s in ("above", "includes zero", "below")} for name in NAMES.values()}
        summary.update({t: {s: [tau for tau in COMMON if side(one[str(tau)]["arms"][t]["ci95"]) == s]
                            for s in ("above", "includes zero", "below")} for t in ARMS})
        out["one_temperature"][short] = {
            "what": "the backbone and the coupled system both read at the one temperature; items of the read "
                    "split resampled on the table's draws",
            "the_cross_fitted_temperature_of_the_backbone": rows["all"][short]["tau_base"],
            "where_the_interval_lies": summary, "at": one}

        # --- in-split: both sides at their own best temperature on the read split, chosen again in each draw
        d_ = np.arange(len(ie))
        gi = {t: eb[t][d_, eb[t].argmin(1)] - ep[t][d_, ep[t].argmin(1)] for t in ARMS}
        point = {t: cs[t][short][1].mean(0).min() - cs[t][short][2].mean(0).min() for t in ARMS}
        out["in_split"][short] = {
            "what": "reported beside the cross-fit by the pre-registration, which reads nothing from it; the "
                    "temperatures are chosen on the items they are scored on, in each draw again, with no "
                    "correction for that choice",
            **described(table, short, point, gi)}

        # --- the curves between the grid's points
        cv = {"backbone": vertex(cs["all"][short][1].mean(0)), **{t: vertex(cs[t][short][2].mean(0)) for t in ARMS}}
        out["curves"][short] = {
            "curves": cv,
            "largest_grid_minimum_above_its_vertex": max(v["grid_minimum_above_the_vertex"] for v in cv.values()),
            "gain_from_the_vertices": {t: r6(cv["backbone"]["vertex_value"] - cv[t]["vertex_value"]) for t in ARMS},
            "the_cross_fit_costs": {"backbone": rows["all"][short]["crossfit_cost_base"],
                                    **{t: rows[t][short]["crossfit_cost_psilm"] for t in ARMS}},
            "one_step_of_the_grid_at_the_cross_fitted_temperature": {
                name: {"down": r6(c[TAUS.index(tau) - 1] - c[TAUS.index(tau)]),
                       "up": r6(c[TAUS.index(tau) + 1] - c[TAUS.index(tau)])}
                for name, c, tau in [("backbone", cs["all"][short][1].mean(0), rows["all"][short]["tau_base"])]
                + [(t, cs[t][short][2].mean(0), rows[t][short]["tau_psilm"]) for t in ARMS]}}

        # --- what temperature alone gives the backbone
        b = cs["all"][short][1].mean(0)
        own = float(b[ONE] - b.min())
        out["the_backbone_alone"][short] = {
            "cross_entropy_as_evaluated": r6(b[ONE]), "best_temperature": TAUS[int(b.argmin())],
            "cross_entropy_at_it": r6(b.min()), "gain_from_temperature": r6(own),
            "over_each_arms_gain_as_evaluated": {t: round(own / rows[t][short]["ce_gain"], 4) for t in ARMS},
            "temperatures_at_which_it_is_at_or_below_the_arm_as_evaluated": {
                t: [TAUS[i] for i in range(len(TAUS)) if b[i] <= cs[t][short][2].mean(0)[ONE]] for t in ARMS}}

        # --- other draws
        od = {}
        for tag, seed, k in (("seed 1", 1, ac.DRAWS), ("seed 2", 2, ac.DRAWS), ("seed 0, 10000 draws", 0, 10000)):
            cf2 = crossfit(cs, short, *draws(n, m, seed, k))
            g2 = {t: cf2[t][0] for t in ARMS}
            od[tag] = {**{t: ac.ci(g2[t]) for t in ARMS},
                       **{NAMES[(x, tuple(bs))]: ac.ci(contrast(g2, x, bs)) for x, bs in PAIRS}}
        asis = {**{t: rows[t][short]["ce_gain_at_best_tau_ci95"] for t in ARMS},
                **{NAMES[(c["arm"], tuple(c["minus"]))]: c["ci95"] for c in table["contrasts_at_best_tau"]
                   if c["split"] == short}}
        od["as_in_the_table"] = asis
        od["whose_side_of_zero_changes"] = sorted(
            x for x in asis if len({side(v[x]) for v in od.values() if isinstance(v, dict)}) > 1)
        out["other_draws"][short] = od

    out["reproduces_the_table"] = {"largest_difference_of_an_interval_end": r6(worst), "holds": bool(worst <= 1e-6)}
    if not out["reproduces_the_table"]["holds"]:
        raise SystemExit(f"these are not the table's draws: an interval's end differs by {worst}")

    # --- each named reading: as posed, in-split, and at each one temperature (the read split)
    R = rd.READ
    posed, ins, one = reading["named"], out["in_split"][R], out["one_temperature"][R]["at"]
    pick = {"the_value_neurons_small_signal": lambda x: x["value_neurons_small_signal"],
            "the_plain_partners_advantage_of_0.0040": lambda x: x["contrasts"]["C1_plain_partner_at_nine"]["label"],
            "the_full_width_partner_null": lambda x: x["contrasts"]["C2_plain_partner_at_full_width"]["label"],
            "the_identity_sign": lambda x: x["identity_sign"],
            "the_63_to_37_split": lambda x: x["split"]["label"]}
    for name, f in pick.items():
        was = posed[name]["reading"] if isinstance(posed[name], dict) else posed[name]
        out["named"][name] = {"as_posed": was, "in_split": f(ins),
                              "at_one_temperature": {tau: f(one[tau]) for tau in one}}
    for name in ("C4_value_neurons_against_random", "C5_width"):
        out["named"][name] = {"as_posed": reading["contrasts"][name][R]["reading"],
                              "in_split": ins["contrasts"][name]["label"],
                              "at_one_temperature": {tau: one[tau]["contrasts"][name]["label"] for tau in one}}
    out["named"]["full_width_gain"] = {"as_posed": reading["arms"]["all"][R]["reading"],
                                       "in_split": ins["arms"]["all"]["label"],
                                       "at_one_temperature": {tau: one[tau]["arms"]["all"]["label"] for tau in one}}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print(f"the table's intervals are reproduced (largest difference {worst:.1e})")
    for short in ac.SPLITS:
        cv = out["curves"][short]["curves"]["backbone"]
        print(f"{short}: the backbone's minimum is at {cv['vertex_temperature']} (grid {cv['grid_temperature']}); the "
              f"cross-fit costs it {out['curves'][short]['the_cross_fit_costs']['backbone']}")
    for name, v in out["named"].items():
        same = sorted({*v["at_one_temperature"].values()})
        print(f"{name}: as posed '{v['as_posed']}'; in-split '{v['in_split']}'; at one temperature "
              + ("the same at all" if same == [v["as_posed"]] else "; ".join(
                  f"'{s}' at {', '.join(t for t, x in v['at_one_temperature'].items() if x == s)}" for s in same)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
