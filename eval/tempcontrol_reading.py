#!/usr/bin/env python3
"""The temperature control at 0.5B, read by the rules written before it ran.

  results/constitution/tempcontrol_qwen0.5b_preregistration.json   the rules
  results/constitution/arms_controls_qwen0.5b.json                 the table
                                                (results/constitution/arms_controls.py --set qwen0.5b)

  python eval/tempcontrol_reading.py            # writes results/constitution/tempcontrol_qwen0.5b_reading.json
  python eval/tempcontrol_reading.py --self-test

Nothing is measured here. Each rule is applied to the table's numbers as the
pre-registration words it, on the red-team split, with the helpful split beside
it; a rule that turns out to be badly posed is applied as posed. An arm that is
not ok keeps its numbers and loses its readings, and so does everything read
from it.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READ = "test"                                   # the split the paper's paired numbers are on
SPLITS = ("test", "helpful")
MATCHED = ("magmatch", "magmatch1", "magmatch2")
#: the contrasts, and what each was as evaluated in the record
#: (results/constitution/paired_bootstrap_qwen0.5b.json, mean_ce, in gains)
CONTRASTS = {"C1_plain_partner_at_nine": ("plainpartner", ["vn"], 0.004032),
             "C2_plain_partner_at_full_width": ("allplain", ["all"], 0.009631),
             "C3_identity_at_nine": ("vn", list(MATCHED), 0.006761),
             "C4_value_neurons_against_random": ("vn", ["rand"], 0.015720),
             "C5_width": ("vn5", ["vn"], 0.042135)}
NULL_AS_EVALUATED = ("C2_plain_partner_at_full_width",)     # on the read split, by the record
PER_DRAW = {"magmatch": 0.004529, "magmatch1": 0.006299, "magmatch2": 0.009454}   # vn minus each draw, the record
SHARE_RECORD = 0.6289                           # the split as evaluated, from the unrounded record
RECORD_TOL, SHARE_TOL = 0.0005, 0.01
GRID = 0.0004                                   # how far a minimum over the grid lay above its curve's at 9B, at most
NOT_READ = "not read, not ok"

KEEPS = "keeps a gain"
ZERO = "not distinguishable from zero at best temperature"
BELOW = "below the backbone at best temperature"


def excludes_zero(ci):
    """An interval with zero at one of its ends includes zero."""
    return ci[0] > 0 or ci[1] < 0


def near_zero(ci):
    """An interval that excludes zero and ends within the grid's resolution of it: read as it
    says, and flagged."""
    return excludes_zero(ci) and min(abs(ci[0]), abs(ci[1])) <= GRID


def gain_reading(ci):
    return KEEPS if ci[0] > 0 else BELOW if ci[1] < 0 else ZERO


def read_contrast(c, null):
    """The reading of one contrast. `null`: its as-evaluated difference was not established, so
    it is read as a null that holds or fails; otherwise as a difference that survives or not.
    The side of the INTERVAL is read, never the point's."""
    lo, hi = c["ci95"]
    was = c["as_evaluated"]["difference"]
    if null:
        how = ("still indistinguishable" if not excludes_zero(c["ci95"]) else
               "the first arm keeps more at the best temperatures" if lo > 0 else
               "the second arm keeps more at the best temperatures")
    elif was == 0:
        how = "no as-evaluated sign: " + ("includes zero" if not excludes_zero(c["ci95"]) else "excludes zero")
    elif not excludes_zero(c["ci95"]):
        how = "does not survive"
    else:
        how = "survives" if (lo > 0) == (was > 0) else "reverses"
    return {"reading": how, "read_as": "a null" if null else "a difference",
            "at_best_temperatures": c["difference"], "ci95": c["ci95"],
            "ends_within_the_grids_resolution_of_zero": near_zero(c["ci95"]),
            "as_evaluated": was, "as_evaluated_ci95": c["as_evaluated"]["ci95"],
            "as_evaluated_excluded_zero": excludes_zero(c["as_evaluated"]["ci95"])}


def row(table, short, a, bs):
    c = [x for x in table["contrasts_at_best_tau"] if x["split"] == short and x["arm"] == a and x["minus"] == bs]
    if len(c) != 1:
        raise SystemExit(f"{a} minus {bs} on {short}: {len(c)} rows in the table")
    return c[0]


def not_read(r, hit):
    return {**r, "reading": f"{NOT_READ}: {', '.join(hit)}"}


def split_reading(share, vn_keeps, matched_ci, c3):
    """The label of the 63/37 split, the first that applies; None if it is stated as a split."""
    if not vn_keeps:
        return "undefined at best temperature"
    if c3 == "reverses":
        return "identity term reversed at best temperature"
    if c3 != "survives":
        return "identity term not distinguishable from zero at best temperature"
    if matched_ci[1] < 0:
        return "magnitude term below the backbone at best temperature"
    if not matched_ci[0] > 0:
        return "magnitude term not distinguishable from zero at best temperature"
    if share is None:                             # the value neurons' gain is 0.000000 in the table
        return "no share: not a split"
    if not 0 <= share <= 1:                       # cannot be, with the three above: said if it is
        return "outside 0 to 1: not a split"
    return None


def edge_direction(s):
    """What a best temperature at the grid's edge does to the gain, where the rule states it:
    one side flagged, and the read split's own best temperature at the same end."""
    fb, fp = s["tau_base_at_edge"], s["tau_psilm_at_edge"]
    if fb == fp:                                  # neither, or both: no direction
        return None
    if fb:
        return "the gain is if anything overstated" if s["tau_base_in_split"] == s["tau_base"] else None
    return "the gain is if anything understated" if s["tau_psilm_in_split"] == s["tau_psilm"] else None


def read(table):
    arms = {r["arm"]: r for r in table["arms"]}
    bad = set(table["not_ok"])
    out = {"the_record_is_reproduced": None, "named": None,
           "table": "results/constitution/arms_controls_qwen0.5b.json",
           "preregistration": "results/constitution/tempcontrol_qwen0.5b_preregistration.json",
           "split_read": READ, "not_ok": sorted(bad), "arms": {}, "contrasts": {}, "beside_C3": {}}
    for tag, r in arms.items():
        out["arms"][tag] = {"what": r["what"], "n_write": r["n_write"], "ok": tag not in bad,
                            "greedy_redecode_diverged_on": r["greedy_decode_diverged"]}
        for short in SPLITS:
            s = r[short]
            ok = tag not in bad
            out["arms"][tag][short] = {
                "gain_as_evaluated": s["ce_gain"], "gain_at_best_temperatures": s["ce_gain_at_best_tau"],
                "ci95": s["ce_gain_at_best_tau_ci95"],
                "reading": gain_reading(s["ce_gain_at_best_tau_ci95"]) if ok else NOT_READ,
                "keeps_a_gain": (s["ce_gain_at_best_tau_ci95"][0] > 0) if ok else None,
                "ends_within_the_grids_resolution_of_zero": near_zero(s["ce_gain_at_best_tau_ci95"]),
                "share_kept": s["share_surviving"],
                "temperature_backbone": s["tau_base"], "temperature_coupled": s["tau_psilm"],
                "backbone_at_the_grids_edge": s["tau_base_at_edge"], "coupled_at_the_grids_edge": s["tau_psilm_at_edge"],
                "at_the_grids_edge": edge_direction(s) if ok else None,
                "beside_in_split": {"temperature_backbone": s["tau_base_in_split"],
                                    "temperature_coupled": s["tau_psilm_in_split"],
                                    "gain": s["ce_gain_at_in_split_tau"],
                                    "the_cross_fit_costs_the_backbone": s["crossfit_cost_base"],
                                    "the_cross_fit_costs_the_coupled_system": s["crossfit_cost_psilm"]},
                "kl_teacher_backbone": s["kl_teacher_base"], "kl_teacher_arm": s["kl_teacher_psilm"],
                "arm_is_nearer_the_teacher_than_the_backbone": (s["kl_teacher_psilm"] < s["kl_teacher_base"]) if ok else None,
                "kl_fraction_removed_at_best_temperatures": s["kl_fraction_at_best_tau"]}
    for name, (a, bs, recorded) in CONTRASTS.items():
        out["contrasts"][name] = {"first_arm": a, "second_arm": bs}
        hit = sorted(bad & {a, *bs})
        for short in SPLITS:
            c = row(table, short, a, bs)
            null = (name in NULL_AS_EVALUATED) if short == READ else not excludes_zero(c["as_evaluated"]["ci95"])
            r = read_contrast(c, null)
            if short == READ:
                r["as_evaluated_in_the_record"] = recorded
                r["the_table_reproduces_the_record"] = abs(r["as_evaluated"] - recorded) <= RECORD_TOL
            out["contrasts"][name][short] = not_read(r, hit) if hit else r
    c3row, c3 = row(table, READ, "vn", list(MATCHED)), out["contrasts"]["C3_identity_at_nine"][READ]
    # beside C3: the value neurons against each draw, and the spread of the draws
    per, hit3 = {}, sorted(bad & {"vn", *MATCHED})
    for m in MATCHED:
        r = read_contrast(row(table, READ, "vn", [m]), null=False)
        r["as_evaluated_in_the_record"] = PER_DRAW[m]
        r["the_table_reproduces_the_record"] = abs(r["as_evaluated"] - PER_DRAW[m]) <= RECORD_TOL
        per[m] = not_read(r, sorted(bad & {"vn", m})) if bad & {"vn", m} else r
    n_sur = sum(p["reading"] == "survives" for p in per.values())
    gains = [arms[m][READ]["ce_gain_at_best_tau"] for m in MATCHED]
    raws = [arms[m][READ]["ce_gain"] for m in MATCHED]
    sd = lambda v: round((sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5, 6)   # noqa: E731
    n_read = f"{NOT_READ}: {', '.join(hit3)}" if hit3 else None
    out["beside_C3"] = {"value_neurons_minus_each_draw": per,
                        "draws_whose_interval_excludes_zero_with_the_sign": n_read or n_sur,
                        "draws_below_the_value_neurons_at_best_temperatures":
                            n_read or sum(g < arms["vn"][READ]["ce_gain_at_best_tau"] for g in gains),
                        "sd_of_the_draws_gains": {"at_best_temperatures": sd(gains), "as_evaluated": sd(raws)}}
    vn = out["arms"]["vn"][READ]
    sh = c3row["magnitude_share"]
    label = (f"{NOT_READ}: {', '.join(hit3)}" if hit3 else
             split_reading(sh["at_best_tau"]["share"], vn["keeps_a_gain"], sh["matched_mean_gain_ci95"], c3["reading"]))
    ci = sh["at_best_tau"]["ci95"]
    split = {"magnitude_share": sh["at_best_tau"]["share"], "ci95": ci,
             "matched_mean_gain": sh["matched_mean_gain"], "matched_mean_gain_ci95": sh["matched_mean_gain_ci95"],
             "matched_mean_gain_ends_within_the_grids_resolution_of_zero": near_zero(sh["matched_mean_gain_ci95"]),
             "as_evaluated": sh["as_evaluated"], "reading": label}
    if label is None:
        x = round(100 * sh["at_best_tau"]["share"], 1)
        closed = ci is not None and None not in ci and 0 <= ci[0] and ci[1] <= 1
        split["reading"] = (f"{x}% magnitude, {round(100 - x, 1)}% identity"
                            + ("" if closed else "; the size of the split is not determined (its interval is not within 0 to 1)"))
    sign = (f"{NOT_READ}: {', '.join(hit3)}" if hit3 else
            "not firm at best temperature" if c3["reading"] != "survives" else
            "firm at best temperature" if n_sur == 3 else
            f"C3 survives, and {n_sur} of the three per-draw intervals exclude zero with the as-evaluated sign")
    out["named"] = {
        "the_value_neurons_small_signal": (f"{NOT_READ}: vn" if "vn" in bad else
                                           "survives" if vn["keeps_a_gain"] else vn["reading"]),
        "the_plain_partners_advantage_of_0.0040": out["contrasts"]["C1_plain_partner_at_nine"][READ]["reading"],
        "the_full_width_partner_null": out["contrasts"]["C2_plain_partner_at_full_width"][READ]["reading"],
        "the_identity_sign": sign,
        "the_63_to_37_split": split}
    rec = {n: c[READ]["the_table_reproduces_the_record"] for n, c in out["contrasts"].items()}
    rec.update({f"vn minus {m}": p["the_table_reproduces_the_record"] for m, p in per.items()})
    was = sh["as_evaluated"]["share"]
    rec["the_split_as_evaluated"] = was is not None and abs(was - SHARE_RECORD) <= SHARE_TOL
    out["the_record_is_reproduced"] = {"all": all(rec.values()), **rec}
    return out


def self_test():
    assert gain_reading([0.0001, 0.02]) == KEEPS and gain_reading([0.0, 0.02]) == ZERO
    assert gain_reading([-0.001, 0.02]) == ZERO and gain_reading([-0.001, 0.0]) == ZERO
    assert gain_reading([-0.02, -0.001]) == BELOW
    assert excludes_zero([0.001, 0.002]) and excludes_zero([-0.002, -0.001])
    assert not excludes_zero([-0.001, 0.002]) and not excludes_zero([0.0, 0.002]) and not excludes_zero([-0.002, 0.0])
    assert near_zero([0.0001, 0.02]) and near_zero([0.0004, 0.02]) and near_zero([-0.02, -0.0004])
    assert not near_zero([0.000401, 0.02]) and not near_zero([0.0, 0.02]) and not near_zero([-0.0001, 0.0002])

    def c(d, lo, hi, was, wlo=0.001, whi=0.008):
        return {"difference": d, "ci95": [lo, hi], "as_evaluated": {"difference": was, "ci95": [wlo, whi]}}
    rc = lambda *a, null=False: read_contrast(c(*a), null)["reading"]                     # noqa: E731
    assert rc(0.003, 0.001, 0.005, 0.004) == "survives" and rc(0.003, -0.001, 0.005, 0.004) == "does not survive"
    assert rc(-0.003, -0.005, -0.001, 0.004) == "reverses" and rc(-0.003, -0.005, -0.001, -0.004) == "survives"
    assert rc(0.003, 0.0, 0.005, 0.004) == "does not survive"                  # zero at an end is inside
    # the interval's side, not the point's: a point outside its own interval does not decide
    assert rc(-0.001, 0.001, 0.005, 0.004) == "survives" and rc(0.001, -0.005, -0.001, 0.004) == "reverses"
    assert rc(0.003, 0.001, 0.005, 0.0).startswith("no as-evaluated sign")
    # a null as evaluated: never survives or reverses, whatever the sign of its point estimate was
    for was in (0.0096, -0.0096):
        assert rc(0.003, -0.001, 0.005, was, null=True) == "still indistinguishable"
        assert rc(0.003, 0.001, 0.005, was, null=True) == "the first arm keeps more at the best temperatures"
        assert rc(-0.003, -0.005, -0.001, was, null=True) == "the second arm keeps more at the best temperatures"

    assert split_reading(0.6, False, [0.001, 0.01], "survives") == "undefined at best temperature"
    assert split_reading(1.2, True, [0.001, 0.01], "reverses").startswith("identity term reversed")
    assert split_reading(0.9, True, [0.001, 0.01], "does not survive").startswith("identity term not distinguishable")
    assert split_reading(0.1, True, [-0.001, 0.01], "survives").startswith("magnitude term not distinguishable")
    assert split_reading(0.1, True, [0.0, 0.01], "survives").startswith("magnitude term not distinguishable")
    assert split_reading(0.1, True, [-0.01, 0.0], "survives").startswith("magnitude term not distinguishable")
    assert split_reading(-1.2, True, [-0.005, -0.004], "survives") == "magnitude term below the backbone at best temperature"
    assert split_reading(None, False, [0.001, 0.01], "survives") == "undefined at best temperature"
    assert split_reading(None, True, [0.001, 0.01], "survives") == "no share: not a split"
    assert split_reading(0.6, True, [0.001, 0.01], "survives") is None
    assert split_reading(1.2, True, [0.001, 0.01], "survives") == "outside 0 to 1: not a split"

    e = {"tau_base": 0.3, "tau_psilm": 0.7, "tau_base_in_split": 0.3, "tau_psilm_in_split": 0.7,
         "tau_base_at_edge": True, "tau_psilm_at_edge": False}
    assert edge_direction(e) == "the gain is if anything overstated"
    assert edge_direction({**e, "tau_base_in_split": 0.35}) is None             # the read split's own is not there
    assert edge_direction({**e, "tau_psilm_at_edge": True}) is None             # both flagged
    assert edge_direction({**e, "tau_base_at_edge": False}) is None             # neither
    e = {**e, "tau_base": 0.6, "tau_base_in_split": 0.6, "tau_base_at_edge": False,
         "tau_psilm": 1.6, "tau_psilm_in_split": 1.6, "tau_psilm_at_edge": True}
    assert edge_direction(e) == "the gain is if anything understated"
    assert edge_direction({**e, "tau_psilm_in_split": 0.3}) is None             # the other end

    def s(raw, best, lo, hi, kb=0.2, kp=0.3):
        return {"ce_gain": raw, "ce_gain_at_best_tau": best, "ce_gain_at_best_tau_ci95": [lo, hi],
                "share_surviving": round(best / raw, 4) if abs(raw) > 1e-9 else None,
                "tau_base": 0.6, "tau_psilm": 0.7, "tau_base_at_edge": False, "tau_psilm_at_edge": False,
                "tau_base_in_split": 0.6, "tau_psilm_in_split": 0.7, "ce_gain_at_in_split_tau": best,
                "crossfit_cost_base": 0.0, "crossfit_cost_psilm": 0.0,
                "kl_teacher_base": kb, "kl_teacher_psilm": kp, "kl_fraction_at_best_tau": -0.1}

    def arm(tag, raw, best, lo, hi):
        return {"arm": tag, "what": tag, "n_write": 9, "greedy_decode_diverged": 0,
                "test": s(raw, best, lo, hi), "helpful": s(raw, best, lo, hi)}
    gains = {"all": (0.3587, 0.2, 0.15, 0.25), "allplain": (0.3683, 0.21, 0.16, 0.26),
             "vn": (0.018216, 0.008, 0.002, 0.014), "plainpartner": (0.022248, 0.009, 0.003, 0.015),
             "vn5": (0.060351, 0.03, 0.02, 0.04), "magmatch": (0.013687, 0.004, -0.001, 0.009),
             "magmatch1": (0.011917, 0.002, -0.003, 0.007), "magmatch2": (0.008762, 0.003, -0.002, 0.008),
             "rand": (0.002496, 0.0, -0.004, 0.004)}
    pairs = [(a, bs, rec) for a, bs, rec in CONTRASTS.values()] + [("vn", [m], PER_DRAW[m]) for m in MATCHED]

    def table(share=0.375, share_ci=(0.1, 0.7), mm_ci=(0.0005, 0.006), c3=(0.005, 0.001, 0.009), not_ok=(), per=(0.004, 0.001, 0.007)):
        rows = []
        for sp in SPLITS:
            for a, bs, rec in pairs:
                d = c3 if (a, bs) == ("vn", list(MATCHED)) else per if (a == "vn" and len(bs) == 1 and bs[0] in MATCHED) \
                    else (0.001, -0.002, 0.004)
                rows.append({"split": sp, "arm": a, "minus": bs, **c(*d, rec)})
                if (a, bs) == ("vn", list(MATCHED)):
                    rows[-1]["magnitude_share"] = {
                        "at_best_tau": {"share": share, "ci95": list(share_ci)},
                        "as_evaluated": {"share": 0.6289, "ci95": [0.52, 0.75]},
                        "matched_mean_gain": 0.003, "matched_mean_gain_ci95": list(mm_ci)}
        return {"not_ok": list(not_ok), "arms": [arm(t, *g) for t, g in gains.items()], "contrasts_at_best_tau": rows}
    out = read(table())
    assert list(out)[:2] == ["the_record_is_reproduced", "named"] and out["the_record_is_reproduced"]["all"] is True
    n = out["named"]
    assert n["the_value_neurons_small_signal"] == "survives" and n["the_plain_partners_advantage_of_0.0040"] == "does not survive"
    assert n["the_full_width_partner_null"] == "still indistinguishable" and n["the_identity_sign"] == "firm at best temperature"
    assert n["the_63_to_37_split"]["reading"] == "37.5% magnitude, 62.5% identity"
    assert out["arms"]["rand"]["test"]["reading"] == ZERO and out["arms"]["rand"]["test"]["share_kept"] == 0.0
    assert out["arms"]["vn"]["test"]["arm_is_nearer_the_teacher_than_the_backbone"] is False
    assert out["contrasts"]["C5_width"]["helpful"]["read_as"] == "a difference"
    assert "as_evaluated_in_the_record" not in out["contrasts"]["C5_width"]["helpful"]
    assert out["beside_C3"]["draws_whose_interval_excludes_zero_with_the_sign"] == 3
    assert out["beside_C3"]["draws_below_the_value_neurons_at_best_temperatures"] == 3
    # an interval of the share that leaves 0 to 1, or does not close: the size is not determined
    for ci in ((0.1, 1.3), (None, 0.7), (-0.2, 0.7)):
        assert read(table(share_ci=ci))["named"]["the_63_to_37_split"]["reading"].endswith("not within 0 to 1)")
    assert read(table(c3=(0.002, -0.001, 0.005)))["named"]["the_identity_sign"] == "not firm at best temperature"
    assert read(table(c3=(0.002, -0.001, 0.005)))["named"]["the_63_to_37_split"]["reading"].startswith("identity term not")
    assert read(table(per=(0.002, -0.001, 0.005)))["named"]["the_identity_sign"].startswith("C3 survives, and 0 of")
    assert read(table(mm_ci=(-0.001, 0.006)))["named"]["the_63_to_37_split"]["reading"].startswith("magnitude term not")
    assert read(table(mm_ci=(-0.006, -0.001)))["named"]["the_63_to_37_split"]["reading"].startswith("magnitude term below")
    assert read(table(share=None))["named"]["the_63_to_37_split"]["reading"] == "no share: not a split"
    flag = "matched_mean_gain_ends_within_the_grids_resolution_of_zero"
    assert out["named"]["the_63_to_37_split"][flag] is False              # [0.0005, 0.006]
    near = read(table(mm_ci=(0.0002, 0.006)))["named"]["the_63_to_37_split"]
    assert near[flag] is True and near["reading"] == "37.5% magnitude, 62.5% identity"
    # a C3 with no as-evaluated sign counts as not surviving
    t = table()
    for x in t["contrasts_at_best_tau"]:
        if x["split"] == "test" and x["minus"] == list(MATCHED):
            x["as_evaluated"]["difference"] = 0.0
    o = read(t)
    assert o["contrasts"]["C3_identity_at_nine"]["test"]["reading"] == "no as-evaluated sign: excludes zero"
    assert o["named"]["the_identity_sign"] == "not firm at best temperature"
    assert o["named"]["the_63_to_37_split"]["reading"].startswith("identity term not distinguishable")
    assert out["arms"]["vn"]["test"]["at_the_grids_edge"] is None
    assert out["arms"]["vn"]["test"]["ends_within_the_grids_resolution_of_zero"] is False
    assert out["contrasts"]["C1_plain_partner_at_nine"]["test"]["ends_within_the_grids_resolution_of_zero"] is False
    assert read(table(c3=(0.002, 0.0002, 0.005)))["contrasts"]["C3_identity_at_nine"]["test"][
        "ends_within_the_grids_resolution_of_zero"] is True
    t = table()
    for x in t["contrasts_at_best_tau"]:
        if "magnitude_share" in x:
            x["magnitude_share"]["as_evaluated"]["share"] = None
    assert read(t)["the_record_is_reproduced"]["the_split_as_evaluated"] is False
    gains["vn"] = (0.018216, -0.004, -0.009, -0.001)                     # the value neurons are below the backbone
    out = read(table())
    assert out["named"]["the_value_neurons_small_signal"] == BELOW
    assert out["named"]["the_63_to_37_split"]["reading"] == "undefined at best temperature"
    gains["vn"] = (0.018216, 0.008, 0.002, 0.014)
    # an arm that is not ok: its numbers stay, its readings go, and so do those that use it
    out = read(table(not_ok=["vn"]))
    assert out["arms"]["vn"]["ok"] is False and out["arms"]["vn"]["test"]["reading"] == NOT_READ
    assert out["arms"]["vn"]["test"]["keeps_a_gain"] is None and out["arms"]["vn"]["test"]["ci95"] == [0.002, 0.014]
    t = table(not_ok=["vn"])                       # no direction at the grid's edge from an arm that is not ok
    for r in t["arms"]:
        r["test"].update(tau_base=0.3, tau_base_in_split=0.3, tau_base_at_edge=True)
    o = read(t)
    assert o["arms"]["vn"]["test"]["at_the_grids_edge"] is None
    assert o["arms"]["rand"]["test"]["at_the_grids_edge"] == "the gain is if anything overstated"
    for name in ("C1_plain_partner_at_nine", "C3_identity_at_nine", "C4_value_neurons_against_random", "C5_width"):
        assert out["contrasts"][name]["test"]["reading"] == f"{NOT_READ}: vn", name
        assert out["contrasts"][name]["test"]["ci95"]                          # the numbers are kept
    assert out["contrasts"]["C2_plain_partner_at_full_width"]["test"]["reading"] == "still indistinguishable"
    assert out["named"]["the_value_neurons_small_signal"] == f"{NOT_READ}: vn"
    assert out["named"]["the_63_to_37_split"]["reading"] == f"{NOT_READ}: vn" == out["named"]["the_identity_sign"]
    assert out["beside_C3"]["draws_whose_interval_excludes_zero_with_the_sign"] == f"{NOT_READ}: vn"
    out = read(table(not_ok=["magmatch1"]))
    assert out["named"]["the_63_to_37_split"]["reading"] == f"{NOT_READ}: magmatch1"
    assert out["beside_C3"]["draws_whose_interval_excludes_zero_with_the_sign"] == f"{NOT_READ}: magmatch1"
    assert out["beside_C3"]["draws_below_the_value_neurons_at_best_temperatures"] == f"{NOT_READ}: magmatch1"
    assert out["named"]["the_value_neurons_small_signal"] == "survives"
    assert out["contrasts"]["C1_plain_partner_at_nine"]["test"]["reading"] == "does not survive"
    assert out["beside_C3"]["value_neurons_minus_each_draw"]["magmatch"]["reading"] == "survives"
    assert out["beside_C3"]["value_neurons_minus_each_draw"]["magmatch1"]["reading"] == f"{NOT_READ}: magmatch1"
    # the record: a table whose as-evaluated number is not the record's says so, first
    t = table()
    for x in t["contrasts_at_best_tau"]:
        if x["split"] == "test" and x["arm"] == "plainpartner":
            x["as_evaluated"]["difference"] = 0.0046
    out = read(t)
    assert out["the_record_is_reproduced"]["all"] is False
    assert out["the_record_is_reproduced"]["C1_plain_partner_at_nine"] is False
    t["contrasts_at_best_tau"] = [x for x in t["contrasts_at_best_tau"] if not (x["split"] == "test" and x["arm"] == "vn5")]
    try:
        read(t)
    except SystemExit:
        pass
    else:
        raise AssertionError("a table without one of the contrasts was read")
    print("[self-test] eval/tempcontrol_reading.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "results/constitution/arms_controls_qwen0.5b.json"))
    ap.add_argument("--out", default=str(ROOT / "results/constitution/tempcontrol_qwen0.5b_reading.json"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    out = read(json.loads(Path(a.table).read_text()))
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    rec = out["the_record_is_reproduced"]
    print(f"the record is reproduced: {rec['all']}" + ("" if rec["all"] else f" (not: {[k for k, v in rec.items() if v is False and k != 'all']})"))
    for tag, r in out["arms"].items():
        t = r["test"]
        print(f"{tag:<13} red-team: gain {t['gain_as_evaluated']:.4f} as evaluated, {t['gain_at_best_temperatures']:+.4f} "
              f"at best temperatures ({t['ci95'][0]:+.4f} to {t['ci95'][1]:+.4f}): {t['reading']}; "
              f"KL to the teacher {t['kl_teacher_arm']:.4f} (backbone {t['kl_teacher_backbone']:.4f})")
    for n, c in out["contrasts"].items():
        t = c["test"]
        print(f"{n}: {t['reading']} ({t['at_best_temperatures']:+.4f}, {t['ci95'][0]:+.4f} to {t['ci95'][1]:+.4f}; "
              f"as evaluated {t['as_evaluated']:+.4f}, the record {t['as_evaluated_in_the_record']:+.4f})")
    for m, p in out["beside_C3"]["value_neurons_minus_each_draw"].items():
        print(f"  vn minus {m}: {p['reading']} ({p['at_best_temperatures']:+.4f}, {p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f})")
    print(json.dumps(out["named"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
