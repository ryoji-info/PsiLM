#!/usr/bin/env python3
"""Does it have to be the probe's coordinates? The identity effect with a spread.

At 410 coordinates and energy parity with a full-width write, the probe-best set
beat the probe-worst set on held-out CE. That was one pair, so it had no error
bar. This collects every 410-coordinate control trained on the same schedule --
the deterministic probe-worst set and the seeded magnitude-matched draws -- and
reports the identity effect as a point estimate with the control spread under it.

Every arm here matches the probe-best 410 on three things: 410 coordinates, total
write energy (by construction, via the cap), and per-coordinate activation
magnitude (by matching, for the match* draws; by accident, for the probe-worst
set). What differs is whether the probe chose them.

  python3 eval/vn_identity_spread.py [--step 500]

Writes results/constitution/identity_spread_qwen35.json.
"""
import argparse, json, statistics as st
from pathlib import Path

ARMS = [("vn10e", "probe-best 410", "treatment"),
        ("vn10ebot", "probe-worst 410", "control"),
        ("match0", "magnitude-matched draw 0", "control"),
        ("match1", "magnitude-matched draw 1", "control"),
        ("match2", "magnitude-matched draw 2", "control"),
        ("all", "whole stream (4096)", "reference")]


def val_ce(tag, step):
    f = Path(f"results/stage2c_qwen35_{tag}/train_log.jsonl")
    if not f.exists():
        return None, None
    best = None
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval") and r.get("step") == step:
            best = r["eval"]
    if not best:
        return None, None
    return best["psilm"]["ce"], best["base"]["ce"]


def test_ce(tag, step):
    """The 50-item test CE, only if the checkpoint it was evaluated from is at
    `step`. An earlier version ignored the step and pooled step-1000 arms with
    step-500 replicates under a heading that said 500."""
    f = Path(f"results/stage2c_qwen35_{tag}/eval_test.json")
    m = Path(f"results/stage2c_qwen35_{tag}/bridges.npz.meta")
    if not f.exists() or not m.exists():
        return None, None
    if int(json.loads(m.read_text()).get("step", -1)) != step:
        return None, None
    d = json.loads(f.read_text())
    a = d.get("arms", d)
    return a["psilm"]["ce"], a["base"]["ce"]


def geometry(tag, rms, e_full):
    f = Path(f"results/stage2c_qwen35_{tag}/bridges.npz.meta")
    if not f.exists():
        return {}
    m = json.loads(f.read_text())
    w = m.get("write_dims") or list(range(len(rms)))
    e = sum(rms[i] ** 2 for i in w)
    cap = float(m["inj_cap"])
    return {"dims": len(w), "cap": cap,
            "energy_rel": round(e / e_full * (cap / 0.2) ** 2, 4),
            "sum_rms2": round(e, 1), "sum_rms": round(sum(rms[i] for i in w), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=500)
    ap.add_argument("--out", default="results/constitution/identity_spread_qwen35.json")
    a = ap.parse_args()
    rms = json.loads(Path("results/value_neurons/qwen35/rms_layer24_full.json").read_text())["rms"]
    e_full = sum(v ** 2 for v in rms)

    recs = {}
    for tag, label, role in ARMS:
        v, vb = val_ce(tag, a.step)
        t, tb = test_ce(tag, a.step)
        if v is None and t is None:
            continue
        recs[tag] = {"label": label, "role": role,
                     "val_ce": v, "val_base": vb,
                     "val_gain": round(vb - v, 4) if v is not None else None,
                     "test_ce": t, "test_base": tb,
                     "test_gain": round(tb - t, 4) if t is not None else None,
                     **geometry(tag, rms, e_full)}

    out = {"step": a.step, "arms": recs,
           "convention": ("val CE is the trainer's own 32-item held-out split at this step; "
                          "test CE is the 50-item red-team split at 24 generated tokens. "
                          "energy_rel is the written set's summed squared activation relative "
                          "to the whole stream, times (cap/0.2)^2 -- 1.0 is parity with a "
                          "full-width write at the default cap.")}

    ctl = [r for r in recs.values() if r["role"] == "control" and r["val_gain"] is not None]
    trt = recs.get("vn10e")
    if ctl and trt and trt["val_gain"]:
        for key in ("val_gain", "test_gain"):
            vals = [r[key] for r in ctl if r.get(key) is not None]
            if not vals or trt.get(key) is None:
                continue
            m = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else None
            out.setdefault("identity_effect", {})[key] = {
                "n_controls": len(vals), "treatment": trt[key],
                "control_mean": round(m, 4),
                "control_sd": round(sd, 4) if sd is not None else None,
                "control_min": min(vals), "control_max": max(vals),
                "ratio": round(trt[key] / m, 3) if m else None,
                "control_share_of_treatment": round(m / trt[key], 3),
                # how many control SDs above the control mean the treatment sits
                "z": round((trt[key] - m) / sd, 2) if sd else None}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'arm':26s} {'role':10s} {'dims':>5s} {'cap':>7s} {'energy':>7s} "
          f"{'val CE':>7s} {'gain':>7s} {'test CE':>8s} {'gain':>7s}")
    for tag, r in recs.items():
        print(f"{r['label']:26s} {r['role']:10s} {r.get('dims','-'):>5} {r.get('cap','-'):>7} "
              f"{r.get('energy_rel','-'):>7} {str(r['val_ce'] or '-'):>7} "
              f"{str(r['val_gain'] or '-'):>7} {str(r['test_ce'] or '-'):>8} "
              f"{str(r['test_gain'] or '-'):>7}")
    for key, e in (out.get("identity_effect") or {}).items():
        sd = f"{e['control_sd']:.4f}" if e["control_sd"] is not None else "n/a"
        z = f"{e['z']}" if e["z"] is not None else "n/a"
        print(f"\n{key}: probe-best {e['treatment']} against {e['n_controls']} controls "
              f"{e['control_mean']} +/- {sd} (range {e['control_min']}-{e['control_max']})")
        print(f"  identity is worth {e['ratio']}x; any matched 410 coordinates reach "
              f"{e['control_share_of_treatment']:.0%} of it; z = {z}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
