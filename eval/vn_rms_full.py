#!/usr/bin/env python3
"""Full per-dimension activation RMS at a value-neuron layer, plus width sets.

eval/vn_rms_summary.py writes the top-20 and a few named sets, which is enough
to check the claims made about the top 1%. Choosing a WIDTH for the write mask
needs the whole vector: a control at 410 dims has to be matched to the RMS of
the top 410, and nothing in the repository records what that is. So this writes
the 4096 numbers themselves (~60 KB) and the summed RMS of every width the
sweep might use, including the bottom-ranked sets that serve as the identity
control at fixed width.

Reads the trajectories with numpy, not MLX, so it can run while an MLX job has
the GPU -- mx.load would open a Metal context next to the trainer.

Usage:  python3 eval/vn_rms_full.py --tag qwen35 --layer 24
"""
import argparse, json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--root", default="results/value_neurons")
    ap.add_argument("--widths", default="41,205,410,1024,2048")
    a = ap.parse_args()
    d = Path(a.root) / a.tag
    key = f"h{a.layer}"
    idx = [json.loads(l) for l in (d / "index.jsonl").read_text().splitlines() if l.strip()]

    sq = None
    n = 0
    n_traj = 0
    pos0 = None
    for r in idx:
        f = d / "traj" / f"{r['idx']}.npz"
        if not f.is_file():
            continue
        with np.load(f) as z:
            if key not in z:
                continue
            H = np.asarray(z[key], dtype=np.float64)
        p = np.abs(H[0])
        pos0 = p if pos0 is None else pos0 + p
        S = H[int(r["prompt_len"]) - 1:]         # the slice eval/vn_probe.py trains on
        s = (S * S).sum(axis=0)
        sq = s if sq is None else sq + s
        n += S.shape[0]
        n_traj += 1
    if sq is None:
        raise SystemExit(f"no {key} states under {d}/traj/")
    rms = np.sqrt(sq / n)
    order = np.argsort(-rms)                      # descending RMS
    rank = np.empty(rms.size, dtype=np.int64)
    rank[order] = np.arange(1, rms.size + 1)      # 1-indexed
    p0 = pos0 / n_traj

    lay = json.loads((d / f"layer{a.layer}.json").read_text())
    ranking = [int(v) for v in lay["ranking"]]    # probe importance, best first

    def describe(dims, label):
        dims = [int(v) for v in dims]
        r = rms[dims]
        return {"label": label, "n": len(dims),
                "summed_rms": round(float(r.sum()), 3),
                "mean_rms": round(float(r.mean()), 4),
                "median_rms": round(float(np.median(r)), 4),
                "max_rms": round(float(r.max()), 4),
                "mean_rms_rank": round(float(rank[dims].mean()), 1)}

    sets = {"value_neurons_top1pct": describe(lay["top1pct"], "probe top 1%"),
            "top5pct": describe(lay["top5pct"], "probe top 5%")}
    for w in [int(x) for x in a.widths.split(",") if x]:
        if w <= len(ranking):
            sets[f"top{w}"] = describe(ranking[:w], f"probe-best {w}")
            sets[f"bot{w}"] = describe(ranking[-w:], f"probe-worst {w}")
    for f in sorted(d.glob(f"magmatch*_layer{a.layer}.json")):
        sets[f.stem] = describe(json.loads(f.read_text())["top1pct"], "magnitude-matched draw")

    out = {"tag": a.tag, "layer": a.layer, "d_model": int(rms.size),
           "n_trajectories": n_traj, "n_states": int(n),
           "convention": ("per-dimension RMS over rows prompt_len-1.. of every trajectory at "
                          "this layer -- the slice eval/vn_probe.py trains on; ranks 1-indexed "
                          "by descending RMS. Read with numpy from the float16 store, "
                          "accumulated in float64."),
           "rms_median": round(float(np.median(rms)), 4),
           "rms_mean": round(float(rms.mean()), 4),
           "rms_max": {"dim": int(order[0]), "rms": round(float(rms[order[0]]), 4)},
           "sink_at_pos0": {"max_dim": int(np.argmax(p0)), "max_abs": round(float(p0.max()), 1),
                            "that_dim_rms_elsewhere": round(float(rms[int(np.argmax(p0))]), 4)},
           "sets": sets,
           "rms": [round(float(v), 4) for v in rms],
           "rank_by_rms": [int(v) for v in rank]}
    dest = d / f"rms_layer{a.layer}_full.json"
    dest.write_text(json.dumps(out) + "\n")
    print(f"wrote {dest}  ({dest.stat().st_size / 1024:.0f} KB, {n_traj} trajectories, {n} states)")
    print(f"  median RMS {out['rms_median']}  largest dim {out['rms_max']['dim']} "
          f"at {out['rms_max']['rms']}  sink dim {out['sink_at_pos0']['max_dim']} "
          f"|h|={out['sink_at_pos0']['max_abs']} at pos 0")
    print(f"  {'set':26s} {'n':>5s} {'sum':>9s} {'mean':>8s} {'max':>8s} {'mean rank':>10s}")
    for k, v in sets.items():
        print(f"  {k:26s} {v['n']:5d} {v['summed_rms']:9.2f} {v['mean_rms']:8.4f} "
              f"{v['max_rms']:8.4f} {v['mean_rms_rank']:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
