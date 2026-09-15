#!/usr/bin/env python
"""Per-dimension activation RMS at a value-neuron layer, as a committed artifact.

Every magnitude claim about the value neurons -- that they are large but not the
largest, that a damaging random draw holds big coordinates, that the matched sets
are matched -- was computed from results/value_neurons/<tag>/traj/*.npz, which is
5.5 GB at 0.5B and 15 GB at 9B and is gitignored. So a reader of the repository
could not check any of them. This writes the small summary they need:

  rms          per-dimension RMS over the probe's own state slice (row
               prompt_len-1 onward, every trajectory) at the chosen layer
  ranking      dimensions by descending RMS, 1-indexed positions for the sets
               below
  sets         the value neurons, the top-5%, each random and matched draw --
               summed RMS, the 1-indexed rank of each member, and the largest
  sink         the attention-sink detector's reading at position 0

Usage:  python eval/vn_rms_summary.py --tag qwen0.5b --layer 16
"""
import argparse, glob, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--root", default="results/value_neurons")
    a = ap.parse_args()
    d = Path(a.root) / a.tag
    idx = [json.loads(l) for l in (d / "index.jsonl").read_text().splitlines() if l.strip()]
    key = f"h{a.layer}"

    import mlx.core as mx
    sq = None
    n = 0
    pos0 = []
    for r in idx:
        f = d / "traj" / f"{r['idx']}.npz"
        if not f.is_file():
            continue
        z = mx.load(str(f))
        if key not in z:
            continue
        H = np.asarray(z[key], dtype=np.float64)
        pos0.append(np.abs(H[0]))                       # the sink lives at row 0
        S = H[int(r["prompt_len"]) - 1:]                # the probe's own slice
        s = (S * S).sum(axis=0)
        sq = s if sq is None else sq + s
        n += S.shape[0]
    if sq is None:
        raise SystemExit(f"no states for {key} under {d}/traj/")
    rms = np.sqrt(sq / n)
    order = np.argsort(-rms)
    rank = {int(v): i + 1 for i, v in enumerate(order)}   # 1-indexed
    p0 = np.mean(np.stack(pos0), axis=0)

    def describe(dims):
        dims = [int(v) for v in dims]
        return {"n": len(dims), "summed_rms": round(float(rms[dims].sum()), 4),
                "ranks": sorted(rank[v] for v in dims),
                "largest_member": {"dim": int(dims[int(np.argmax(rms[dims]))]),
                                   "rms": round(float(rms[dims].max()), 4)}}

    sets = {}
    lay = json.loads((d / f"layer{a.layer}.json").read_text())
    for name, kk in (("value_neurons_top1pct", "top1pct"), ("top5pct", "top5pct")):
        if kk in lay:
            sets[name] = describe(lay[kk])
    for f in sorted(d.glob(f"magmatch*_layer{a.layer}.json")):
        sets[f.stem] = describe(json.loads(f.read_text())["top1pct"])
    for f in sorted(Path("results").glob(f"stage2c_{a.tag}_*/bridges.npz.meta")):
        m = json.loads(f.read_text())
        if m.get("write_dims") and len(m["write_dims"]) < rms.size:
            sets["bridge_write_" + f.parent.name.split("_")[-1]] = describe(m["write_dims"])

    out = {"tag": a.tag, "layer": a.layer, "d_model": int(rms.size),
           "n_trajectories": sum(1 for r in idx if (d / "traj" / f"{r['idx']}.npz").is_file()),
           "n_states": int(n),
           "convention": ("per-dimension RMS over rows prompt_len-1.. of every trajectory at "
                          "this layer, which is exactly the slice eval/vn_probe.py trains on; "
                          "ranks are 1-indexed by descending RMS"),
           "rms_median": round(float(np.median(rms)), 4),
           "rms_max": {"dim": int(order[0]), "rms": round(float(rms[order[0]]), 4)},
           "top20_by_rms": [{"dim": int(v), "rms": round(float(rms[v]), 4)} for v in order[:20]],
           "sink_at_pos0": {"max_dim": int(np.argmax(p0)), "max_abs": round(float(p0.max()), 1),
                            "that_dim_rms_elsewhere": round(float(rms[int(np.argmax(p0))]), 4)},
           "sets": sets}
    dest = d / f"rms_layer{a.layer}_summary.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    print(f"  median RMS {out['rms_median']}, largest dim {out['rms_max']['dim']} "
          f"at {out['rms_max']['rms']}")
    for k, v in sets.items():
        print(f"  {k:28s} sum {v['summed_rms']:8.3f}  ranks {v['ranks'][:6]}"
              f"{'...' if len(v['ranks']) > 6 else ''}")


if __name__ == "__main__":
    main()
