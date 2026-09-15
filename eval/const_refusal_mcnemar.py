#!/usr/bin/env python3
"""Paired refusal test on the guard-rail red-team rows.

The guard-rail summary reports a refusal-keyword RATE per arm, which cannot
distinguish "six prompts flipped to refusal" from "three flipped each way".
The direction is the whole claim, so recompute it paired, per item, with an
exact two-sided McNemar (sign test on the discordant pairs -- the red-team n
is 100 and the discordant counts are single digits, so the chi-square
approximation is not usable here).

  python3 eval/const_refusal_mcnemar.py --tags const_qwen35_vn,const_qwen35_all
  python3 eval/const_refusal_mcnemar.py --tags ... --out results/constitution/refusal_mcnemar.json

Reads results/bench/<tag>_guardrail.json (the run file, which carries per-item
rows) rather than the summary. is_refusal is imported from the bench so the
rates reproduce the summary's exactly.
"""
import argparse, json, sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import is_refusal   # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for b vs c discordant pairs under H0 p=1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pair(by: dict, a: str, b: str, qids: list) -> dict:
    a_only = sum(1 for q in qids if by[a][q] and not by[b][q])
    b_only = sum(1 for q in qids if by[b][q] and not by[a][q])
    both = sum(1 for q in qids if by[a][q] and by[b][q])
    return {"n": len(qids), "both": both, "a_only": a_only, "b_only": b_only,
            "neither": len(qids) - both - a_only - b_only,
            "rate_a": round(sum(by[a][q] for q in qids) / len(qids), 4),
            "rate_b": round(sum(by[b][q] for q in qids) / len(qids), 4),
            "delta": round((b_only - a_only) / len(qids), 4),
            "mcnemar_p": round(mcnemar_exact(a_only, b_only), 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated guard-rail tags")
    ap.add_argument("--dataset", default="redteam")
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--pairs", default="base:psilm,base:zeroed,zeroed:psilm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = {}
    for tag in [t for t in a.tags.split(",") if t]:
        f = Path(a.bench_dir) / f"{tag}_guardrail.json"
        if not f.exists():
            print(f"{tag}: no {f}", file=sys.stderr)
            continue
        d = json.loads(f.read_text())
        rows = [r for r in d.get("rows", []) if r.get("dataset") == a.dataset]
        if not rows:
            print(f"{tag}: no {a.dataset} rows", file=sys.stderr)
            continue
        # A truncated text could hide a refusal keyword past the cut; the
        # keyword list is front-loaded phrasing, but say so rather than assume.
        trunc = sum(1 for r in rows if r.get("text_truncated"))
        by: dict = {}
        for r in rows:
            by.setdefault(r["arm"], {})[r["qid"]] = is_refusal(r["text"])
        rec = {"n_rows": len(rows), "texts_truncated": trunc,
               "ckpt_step": d.get("ckpt_step"), "pairs": {}}
        for spec in a.pairs.split(","):
            x, y = spec.split(":")
            if x in by and y in by:
                qids = sorted(set(by[x]) & set(by[y]))
                rec["pairs"][spec] = pair(by, x, y, qids)
        out[tag] = rec

    hdr = f"{'tag':22s} {'pair':16s} {'rate_a':>7s} {'rate_b':>7s} {'a_only':>7s} {'b_only':>7s} {'delta':>7s} {'p':>8s}"
    print(hdr); print("-" * len(hdr))
    for tag, rec in out.items():
        for spec, p in rec["pairs"].items():
            print(f"{tag:22s} {spec:16s} {p['rate_a']:7.3f} {p['rate_b']:7.3f} "
                  f"{p['a_only']:7d} {p['b_only']:7d} {p['delta']:+7.3f} {p['mcnemar_p']:8.4f}")
        if rec["texts_truncated"]:
            print(f"  ({tag}: {rec['texts_truncated']} stored texts were truncated)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
