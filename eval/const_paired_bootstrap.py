#!/usr/bin/env python
"""Paired per-item cross-entropy, and bootstrap intervals, across write locations.

The magnitude-vs-identity argument for the constitution bridge rests on paired
differences of a few thousandths of a nat between variants that share their hundred
held-out items. Those intervals were computed in-session and written into
docs/technical-notes.md, but nothing recorded them: eval_test.jsonl keeps each
item's text, refusal and gate, and no per-item CE. A reader could not check the
numbers the argument turns on. This regenerates them and records both the per-item
values and the intervals.

  python eval/const_paired_bootstrap.py --tag qwen0.5b \
      --variants vn,vn5,all,rand,magmatch,magmatch1,magmatch2,plainpartner,allplain

Teacher-forced CE on each item's stored teacher continuation, one arm per variant
plus a single shared base arm (the base arm does not depend on which bridge is
loaded, and every run reproduces it to the last digit -- which is itself the check
that the harness is deterministic). Cheap at 0.5B: one forward per item per arm.
"""
import argparse, itertools, json, random
from pathlib import Path

import mlx.core as mx


def bootstrap(diff, n=20000, seed=0):
    rr = random.Random(seed)
    k = len(diff)
    means = sorted(sum(diff[rr.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return (sum(diff) / k, means[int(0.025 * n)], means[int(0.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen0.5b")
    ap.add_argument("--variants", default="vn,vn5,all,rand,magmatch,magmatch1,magmatch2,"
                                          "plainpartner,allplain")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pairs", default=None,
                    help="comma-separated a:b; default is every variant against base "
                         "plus the comparisons the write-up makes")
    a = ap.parse_args()
    data = a.data or f"data/constitution_test_{a.tag}.json"
    items = json.loads(Path(data).read_text())[:a.n]
    variants = a.variants.split(",")

    from psilm.mlx.constitution import PsiConstitutionMLX, load_constitution_stack
    from psilm.mlx.gemma_loader import load_backbone_any
    from transformers import AutoTokenizer
    model, _stock, tok = load_backbone_any(a.model)
    AutoTokenizer.from_pretrained(a.hf_tokenizer)

    per_item, base_seen = {}, None
    for v in variants:
        ck = Path(f"results/stage2c_{a.tag}_{v}/bridges.npz")
        if not ck.is_file():
            print(f"  {v}: no checkpoint, skipped"); continue
        bridges, const, meta = load_constitution_stack(ck)
        psi = PsiConstitutionMLX(model, tok, const, bridges,
                                 l_fwd=meta["l_fwd"], l_rev=meta["l_rev"])
        ces, bases = [], []
        for it in items:
            if not it.get("teacher_ids"):
                continue
            ce, _, _ = psi.teacher_forced(it["prompt_ids"], it["teacher_ids"], "psilm")
            b, _, _ = psi.teacher_forced(it["prompt_ids"], it["teacher_ids"], "base")
            ces.append(ce); bases.append(b)
        per_item[v] = ces
        if base_seen is None:
            base_seen = bases
        else:
            same = all(abs(x - y) < 1e-12 for x, y in zip(base_seen, bases))
            print(f"  {v}: base arm identical to the first run: {same}")
        print(f"  {v}: n={len(ces)} mean CE {sum(ces)/len(ces):.4f} "
              f"(base {sum(bases)/len(bases):.4f}), {len(bridges.write_dims)} dims")
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    per_item["base"] = base_seen

    pairs = ([tuple(s.split(":")) for s in a.pairs.split(",")] if a.pairs else
             [(v, "base") for v in variants if v in per_item]
             + [("vn", "magmatch"), ("magmatch", "rand"), ("vn", "rand"),
                ("plainpartner", "vn"), ("allplain", "all")])
    out = {"tag": a.tag, "data": data, "n_items": len(base_seen),
           "resamples": 20000, "seed": 0,
           "convention": ("teacher-forced CE on each item's stored teacher_ids; paired "
                          "over the same items; percentile bootstrap"),
           "mean_ce": {k: sum(v) / len(v) for k, v in per_item.items()},
           "per_item_ce": per_item, "pairs": {}}
    print()
    for x, y in pairs:
        if x not in per_item or y not in per_item:
            continue
        d = [p - q for p, q in zip(per_item[x], per_item[y])]
        m, lo, hi = bootstrap(d)
        better = sum(1 for v in d if v < 0)
        out["pairs"][f"{x} vs {y}"] = {"mean_diff": m, "ci95": [lo, hi],
                                       "n_items_lower_ce": better, "n": len(d)}
        print(f"  {x:14s} vs {y:12s} {m:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
              f"{better}/{len(d)} items lower")
    dest = Path(f"results/constitution/paired_bootstrap_{a.tag}.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
