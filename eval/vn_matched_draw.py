#!/usr/bin/env python3
"""Magnitude-matched random control masks for a value-neuron write set.

The 410-coordinate draws (match0/1/2_layer24.json) were built inline; this is
the same procedure as a script, for the 41- and 205-coordinate controls that
pair with the energy-parity arms vn1e and vn5e (results/qwen35/parity_widths.sh).

Procedure. Take the treatment coordinates (the probe ranking's first N), sort
them by activation RMS descending, and for each pick an unused coordinate from
the pool whose RMS is among the three nearest, the choice broken by a seeded
RNG. The pool is every coordinate outside the probe-best 410 (the top 10%) and
outside the attention sink 3994, for every N: a 41-coordinate control drawn
from ranks 42-410 would still be coordinates the probe favours, and the
question is whether the probe's choice matters against coordinates it did not
pick at all. The output is a mask file the trainer resolves with
"<file>:top<N>" (psilm/mlx/constitution.py), plus the parity cap
0.2 * sqrt(E_full / E_mask) that gives the control the same write energy as a
full-width write at cap 0.2 (eval/vn_identity_spread.py::geometry).

  python3 eval/vn_matched_draw.py --n 41  --seed 2041 --out results/value_neurons/qwen35/match41_layer24.json
  python3 eval/vn_matched_draw.py --n 205 --seed 2205 --out results/value_neurons/qwen35/match205_layer24.json
"""
import argparse, json, math, random
from pathlib import Path

VN = Path("results/value_neurons/qwen35")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="control size; the treatment is the ranking's first n")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--probe", default=str(VN / "layer24.json"))
    ap.add_argument("--rms", default=str(VN / "rms_layer24_full.json"))
    ap.add_argument("--exclude-top", type=int, default=410, help="pool excludes the probe's first this-many")
    ap.add_argument("--exclude-dims", default="3994", help="comma-separated coordinates never drawn (attention sink)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    probe = json.loads(Path(a.probe).read_text())
    rms = json.loads(Path(a.rms).read_text())["rms"]
    d = len(rms)
    ranking = probe["ranking"]
    treat = ranking[: a.n]
    banned = set(ranking[: a.exclude_top]) | {int(x) for x in a.exclude_dims.split(",") if x}
    pool = sorted((i for i in range(d) if i not in banned), key=lambda i: -rms[i])
    assert len(pool) >= a.n, (len(pool), a.n)

    rng = random.Random(a.seed)
    used, control, err = set(), [], []
    for t in sorted(treat, key=lambda i: -rms[i]):
        cands = sorted((i for i in pool if i not in used), key=lambda i: abs(rms[i] - rms[t]))[:3]
        c = rng.choice(cands)
        used.add(c); control.append(c); err.append(abs(rms[c] - rms[t]))
    assert len(set(control)) == a.n and not (set(control) & set(treat))

    e_full = sum(r * r for r in rms)
    e_treat = sum(rms[i] ** 2 for i in treat)
    e_ctrl = sum(rms[i] ** 2 for i in control)
    cap_treat = 0.2 * math.sqrt(e_full / e_treat)
    cap_ctrl = 0.2 * math.sqrt(e_full / e_ctrl)
    note = (f"MAGNITUDE-MATCHED random control for the probe-best {a.n} at layer {probe['layer']} of Qwen3.5 9B. "
            f"{a.n} coordinates from outside the probe-best {a.exclude_top} (attention sink {a.exclude_dims} excluded), "
            f"matched one to one on activation RMS to the probe-best {a.n} in descending order, the choice among the three "
            f"nearest candidates broken by random.Random({a.seed}). Not a probe output. Summed RMS^2 {e_ctrl:.1f} against the "
            f"treatment's {e_treat:.1f}, mean per-coordinate match error {sum(err) / len(err):.4f}, max {max(err):.4f}; "
            f"--inj-cap {cap_ctrl:.4f} gives energy parity with a full-width write at cap 0.2 (the treatment's parity cap is "
            f"{cap_treat:.4f}). Magnitudes from {Path(a.rms).name}.")
    out = {"layer": probe["layer"], "d_model": d, "n_train": probe.get("n_train"), "n_test": probe.get("n_test"),
           "auc_full": None, "auc_random99": [], "epochs": 0, "sec_per_epoch": 0.0,
           "ranking": control, "treatment": treat, "seed": a.seed, "pool_excludes_top": a.exclude_top,
           "sum_rms2": round(e_ctrl, 2), "treatment_sum_rms2": round(e_treat, 2),
           "parity_cap": round(cap_ctrl, 4), "treatment_parity_cap": round(cap_treat, 4),
           "match_err_mean": round(sum(err) / len(err), 4), "match_err_max": round(max(err), 4), "_note": note}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}: n={a.n} sum_rms2 {e_ctrl:.1f} vs treatment {e_treat:.1f} ({e_ctrl / e_treat:.3f}); "
          f"parity cap {cap_ctrl:.4f} (treatment {cap_treat:.4f}); match err mean {sum(err) / len(err):.4f} max {max(err):.4f}; "
          f"rms median ctrl {sorted(rms[i] for i in control)[a.n // 2]:.2f} treat {sorted(rms[i] for i in treat)[a.n // 2]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
