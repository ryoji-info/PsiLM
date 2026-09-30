#!/usr/bin/env python3
"""The verdict on the Bonsai physics bridge, by the criteria written before it was trained.

Reads results/bonsai/physics_preregistration.json (what was promised), the held-out
evaluation (results/stage2_bonsai27b/final_eval.json) and the guard-rail
(results/bench/guardrail_bonsai27b_guardrail_summary.json and its rows), and writes
results/bonsai/physics_verdict.json: each criterion with the numbers it was decided
on, and one of the three verdicts. It prints numbers only.

  python results/bonsai/physics_verdict.py              # exit 0 once the verdict is written
  python results/bonsai/physics_verdict.py --self-test
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NONPHYS = ("gsm8k", "mmlu", "boolq")


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(rows, dataset):
    """(n pairs, base right and psilm wrong, base wrong and psilm right) over the items both arms answered."""
    ok = {}
    for r in rows:
        if r["dataset"] == dataset and r["arm"] in ("base", "psilm"):
            ok.setdefault(r["qid"], {})[r["arm"]] = bool(r["ok"])
    both = [v for v in ok.values() if len(v) == 2]
    return len(both), sum(v["base"] and not v["psilm"] for v in both), sum(v["psilm"] and not v["base"] for v in both)


def verdict(held, summary, rows):
    """The criteria and the verdict from a held-out summary, a guard-rail summary and its rows."""
    out = {}
    hp, hb = held["psilm"]["acc"], held["baseline"]["acc"]
    out["A_answers"] = {"n": held["n"], "psilm": hp, "baseline": hb, "oracle": held.get("oracle", {}).get("acc"),
                        "psilm_mae": held["psilm"].get("mae"), "holds": held["n"] >= 60 and hp >= 0.90 and hp - hb >= 0.50}
    s = summary["summary"]
    phys = s["physics"]["arms"]
    out["B_answers_in_the_decoder_the_app_runs"] = {
        "n": phys["psilm"]["n"], "psilm": phys["psilm"]["acc"], "base": phys["base"]["acc"],
        "zeroed": phys.get("zeroed", {}).get("acc"), "holds": phys["psilm"]["n"] >= 100 and phys["psilm"]["acc"] >= 0.90}
    c, c_ok = {}, True
    for d in NONPHYS:
        a = s[d]["arms"]
        n, b_only, p_only = paired(rows, d)
        drop = a["base"]["acc"] - a["psilm"]["acc"]
        p = mcnemar(b_only, p_only)
        good = a["psilm"]["n"] >= 100 and n >= 100 and drop <= 0.03 + 1e-9 and (drop <= 0 or p >= 0.05)
        c[d] = {"n": a["psilm"]["n"], "base": a["base"]["acc"], "psilm": a["psilm"]["acc"], "paired_items": n,
                "base_only_right": b_only, "psilm_only_right": p_only, "mcnemar_p": round(p, 4), "holds": good}
        c_ok &= good
    out["C_no_harm"] = {**c, "holds": c_ok}
    g = phys["psilm"]["sigma"]
    gd = {"physics": {"gate_mean": g["mean"], "open_rate": g["open_rate"]}}
    d_ok = g["mean"] >= 0.5 and g["open_rate"] >= 0.9
    for d in NONPHYS:
        sg = s[d]["arms"]["psilm"]["sigma"]
        gd[d] = {"gate_mean": sg["mean"], "open_rate": sg["open_rate"]}
        d_ok &= sg["mean"] <= 0.05 and sg["open_rate"] <= 0.05
    out["D_selective_gate"] = {**gd, "holds": d_ok}
    ab = out["A_answers"]["holds"] and out["B_answers_in_the_decoder_the_app_runs"]["holds"]
    out["verdict"] = ("works" if ab and c_ok and d_ok else "answers_but_is_not_selective" if ab else "does_not")
    return out


def self_test():
    def summ(phys_acc, accs, gates, opens=(1.0, 0.0, 0.0, 0.0)):
        names = ("physics",) + NONPHYS
        return {"summary": {d: {"arms": {"base": {"n": 100, "acc": 0.01 if d == "physics" else 0.8},
                                         "psilm": {"n": 100, "acc": phys_acc if d == "physics" else accs[i - 1],
                                                   "sigma": {"mean": gates[i], "open_rate": opens[i]}},
                                         "zeroed": {"n": 100, "acc": 0.1}}} for i, d in enumerate(names)}}

    def rows(losses):                               # per dataset: (base only right, psilm only right)
        out = []
        for d, (b, p) in zip(NONPHYS, losses):
            for i in range(100):
                out.append({"dataset": d, "qid": f"{d}:{i}", "arm": "base", "ok": i < b or i >= b + p + 20})
                out.append({"dataset": d, "qid": f"{d}:{i}", "arm": "psilm", "ok": b <= i < b + p or i >= b + p + 20})
        return out

    held = {"n": 60, "psilm": {"acc": 1.0, "mae": 0.015}, "baseline": {"acc": 0.03}, "oracle": {"acc": 0.98}}
    good = verdict(held, summ(0.99, (0.8, 0.8, 0.79), (0.8, 0.01, 0.01, 0.01)), rows(((1, 1), (0, 0), (2, 1))))
    assert good["verdict"] == "works", good
    assert abs(mcnemar(0, 0) - 1.0) < 1e-12 and abs(mcnemar(1, 1) - 1.0) < 1e-12 and abs(mcnemar(0, 5) - 0.0625) < 1e-12
    assert abs(mcnemar(8, 0) - 0.0078125) < 1e-12 and abs(mcnemar(3, 1) - 0.625) < 1e-12
    low = verdict({**held, "psilm": {"acc": 0.85}}, summ(0.99, (0.8,) * 3, (0.8, 0.01, 0.01, 0.01)), rows(((0, 0),) * 3))
    assert low["verdict"] == "does_not" and not low["A_answers"]["holds"]
    near = verdict({**held, "psilm": {"acc": 0.95}, "baseline": {"acc": 0.5}}, summ(0.99, (0.8,) * 3, (0.8, 0.01, 0.01, 0.01)),
                   rows(((0, 0),) * 3))
    assert near["verdict"] == "does_not"                     # not 0.50 above the backbone alone
    harm = verdict(held, summ(0.99, (0.72, 0.8, 0.8), (0.8, 0.01, 0.01, 0.01)), rows(((8, 0), (0, 0), (0, 0))))
    assert harm["verdict"] == "answers_but_is_not_selective" and not harm["C_no_harm"]["gsm8k"]["holds"]
    small = verdict(held, summ(0.99, (0.78, 0.8, 0.8), (0.8, 0.01, 0.01, 0.01)), rows(((3, 1), (0, 0), (0, 0))))
    assert small["verdict"] == "works" and small["C_no_harm"]["gsm8k"]["mcnemar_p"] == 0.625
    sig = verdict(held, summ(0.99, (0.77, 0.8, 0.8), (0.8, 0.01, 0.01, 0.01)), rows(((9, 6), (0, 0), (0, 0))))
    assert sig["verdict"] == "works"                          # 3 points below, p 0.61
    leak = verdict(held, summ(0.99, (0.8,) * 3, (0.8, 0.01, 0.2, 0.01), (1.0, 0.0, 0.6, 0.0)), rows(((0, 0),) * 3))
    assert leak["verdict"] == "answers_but_is_not_selective" and not leak["D_selective_gate"]["holds"]
    shut = verdict(held, summ(0.99, (0.8,) * 3, (0.3, 0.01, 0.01, 0.01), (1.0, 0.0, 0.0, 0.0)), rows(((0, 0),) * 3))
    assert shut["verdict"] == "answers_but_is_not_selective"
    dec = verdict(held, summ(0.80, (0.8,) * 3, (0.8, 0.01, 0.01, 0.01)), rows(((0, 0),) * 3))
    assert dec["verdict"] == "does_not" and dec["A_answers"]["holds"]
    short = verdict({**held, "n": 30}, summ(0.99, (0.8,) * 3, (0.8, 0.01, 0.01, 0.01)), rows(((0, 0),) * 3))
    assert short["verdict"] == "does_not"                     # fewer held-out items than promised
    print("[self-test] results/bonsai/physics_verdict.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--held-out", default="results/stage2_bonsai27b/final_eval.json")
    ap.add_argument("--tag", default="guardrail_bonsai27b")
    ap.add_argument("--out", default="results/bonsai/physics_verdict.json")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    pre = json.loads((ROOT / "results/bonsai/physics_preregistration.json").read_text())
    held = json.loads((ROOT / a.held_out).read_text())["summary"]
    summary = json.loads((ROOT / f"results/bench/{a.tag}_guardrail_summary.json").read_text())
    rows = [json.loads(l) for l in (ROOT / f"results/bench/{a.tag}_guardrail.rows.jsonl").read_text().splitlines() if l.strip()]
    rows = [{k: r[k] for k in ("dataset", "qid", "arm", "ok")} for r in rows]         # no text is kept
    for d in ("physics",) + NONPHYS:                # the rows must be the ones the summary was made from
        for arm in ("base", "psilm"):
            got = [r["ok"] for r in rows if r["dataset"] == d and r["arm"] == arm]
            want = summary["summary"][d]["arms"][arm]
            if len(got) != want["n"] or sum(map(bool, got)) != want["n_correct"]:
                raise SystemExit(f"{a.tag}: the rows give {sum(map(bool, got))} of {len(got)} on {d}/{arm}, "
                                 f"the summary {want['n_correct']} of {want['n']}; no verdict from mismatched files")
    out = verdict(held, summary, rows)
    ops = ROOT / Path(a.held_out).with_name("final_eval_ops.json")
    rec = {"criteria_from": "results/bonsai/physics_preregistration.json", "criteria": pre["criteria"],
           "checkpoint_step": held.get("step"), "couple": held.get("couple"),
           "guard_rail_checkpoint_step": summary.get("ckpt_step"), **out,
           "held_out_on_the_training_numerics": (json.loads(ops.read_text())["summary"].get("psilm") if ops.exists() else None),
           "what_follows": pre["what_follows"][out["verdict"]]}
    (ROOT / a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "criteria"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
