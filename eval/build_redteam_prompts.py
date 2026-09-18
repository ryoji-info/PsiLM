#!/usr/bin/env python3
"""Extend the guard-rail's red-team set, prompts only, without the GPU.

The paired refusal test counts DISCORDANT pairs, so its floor is six clean
one-directional flips whatever the item count: at n = 100 a four-flip effect is
unresolvable (exact p = 0.375) and the project's energy-matched arm produced
exactly that. More prompts is the whole fix -- at the same per-prompt flip rate
400 items turn 4:1 into roughly 16:4, p = 0.012.

eval/bench_common.py's load_redteam reads ONLY prompt_text (with prompt_ids for a
tokenisation check and source for provenance); it says in as many words that the
stored teacher and base continuations are training material and are not read at
eval time. So the extension needs no teacher generation and no weights -- just
held-out human turns, encoded.

The first N items of the output are the existing test file's, verbatim and in
order, so a 400-item run's first 100 rows are the same items as every recorded
n = 100 run and stay comparable one for one. The appended items carry no
teacher_ids or base_ids and MUST NOT be used for the CE evaluations --
eval/mlx_constitution_eval.py needs those fields and should keep reading
data/constitution_test_<tag>.json.

Dedup is against every split of the training data by normalised text, using the
builder's own _norm and load_pool so the exclusion is identical.

  python3 eval/build_redteam_prompts.py --n 400 --tag qwen35
"""
import os
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import SYSTEM, chat_ids                      # noqa: E402
from build_constitution_data import _norm, load_pool, SOURCES  # noqa: E402


def load_pool_from_cache(split: str, min_chars: int, max_chars: int, root: str):
    """load_pool's body against the cached Arrow file directly.

    Offline, `datasets` will not resolve data_dir="harmless-base" OR the config
    name it prints as available, so neither load_dataset path works. The Arrow
    file is sitting there, so read it: glob every cached hh-rlhf-<split>.arrow and
    take the one whose row count matches harmless-base (2312 test / 42537 train),
    which distinguishes it from helpful-base (2354 / 43835). verify_pool then
    proves the choice against the carried items' own recorded row indices.
    """
    import glob
    from datasets import Dataset
    from build_constitution_data import first_human_turn
    want = {"test": 2312, "train": 42537}[split]
    cands = sorted(glob.glob(f"{root}/Anthropic___hh-rlhf/**/hh-rlhf-{split}.arrow", recursive=True))
    if not cands:
        raise SystemExit(f"no cached hh-rlhf-{split}.arrow under {root}")
    ds = None
    for f in cands:
        d = Dataset.from_file(f)
        print(f"  cached {f.split('/')[-3][:24]}/{split}: {len(d)} rows"
              f"{'  <- harmless-base' if len(d) == want else ''}")
        if len(d) == want:
            ds = d
    if ds is None:
        raise SystemExit(f"none of {len(cands)} cached {split} files has {want} rows "
                         f"(harmless-base); refusing to guess which subset to use")
    seen, out = set(), []
    for i, row in enumerate(ds):
        turn = first_human_turn(row["chosen"])
        if turn is None or not (min_chars <= len(turn) <= max_chars):
            continue
        key = _norm(turn)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": f"hh:harmless-base:{split}:{i}", "user": turn, "key": key})
    return out


def verify_pool(pool, carried):
    """The fallback config must be the subset the carried items came from: each of
    their sources names a row index, and the pool entry with that source must
    carry the identical turn. Anything else means the wrong subset loaded."""
    by_source = {p["source"]: p["user"] for p in pool}
    checked = miss = 0
    for r in carried:
        s = r.get("source")
        if not s or not s.startswith("hh:harmless-base:"):
            continue
        checked += 1
        if by_source.get(s) != r["prompt_text"]:
            miss += 1
    if not checked:
        raise SystemExit("cannot verify the fallback config: no carried item names a source row")
    if miss:
        raise SystemExit(f"fallback config is the WRONG subset: {miss} of {checked} carried "
                         f"items do not match the turn at their own recorded row index")
    print(f"fallback config verified: {checked} carried items match their recorded row indices")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen35")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    # These MUST match eval/build_constitution_data.py's own defaults (20/400).
    # Widening them changes the pool the original test split was drawn from, and
    # verify_pool then reports carried items as absent -- which is what happened
    # with 40/600: 24 of the 100 carried prompts are shorter than 40 characters.
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--tokenizer", default=os.environ.get("PSILM_BACKBONE", "ryoji-info/Qwen3.5-9B-PsiLM"),
                    help="the backbone tokenizer (default: $PSILM_BACKBONE, else the Hub id)")
    ap.add_argument("--cache-root", default=os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets"),
                    help="where the hh-rlhf Arrow cache lives, read directly when an offline "
                         "load by data_dir cannot resolve the config")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"data/redteam_{a.tag}_n{a.n}.json")

    base = json.loads(Path(f"data/constitution_test_{a.tag}.json").read_text())
    keep = base[:a.n]
    print(f"carrying {len(keep)} existing test items forward verbatim")
    if len(keep) >= a.n:
        out.write_text(json.dumps(keep) + "\n")
        print(f"wrote {out} ({len(keep)} items, no extension needed)")
        return 0

    # Everything already used anywhere, so no appended prompt was ever trained on.
    used = set()
    for split in ("train", "val", "test", "helpful_test"):
        f = Path(f"data/constitution_{split}_{a.tag}.json")
        if f.exists():
            recs = json.loads(f.read_text())
            used |= {_norm(r["prompt_text"]) for r in recs}
            print(f"  {split}: {len(recs)} items excluded")
    data_dir, hf_split = SOURCES["test"]
    try:
        pool = load_pool(data_dir, hf_split, a.min_chars, a.max_chars)
    except ValueError as e:
        # Offline, the hh-rlhf cache is keyed by a resolved config hash and
        # data_dir="harmless-base" no longer maps onto it. Fall back to the
        # config name, then PROVE it is the right subset: every carried-over item
        # records its own row index as "hh:harmless-base:test:<i>", so the turns
        # at those indices must match the prompts we already have.
        print(f"load_pool by data_dir failed offline ({e}); falling back to --config")
        pool = load_pool_from_cache(hf_split, a.min_chars, a.max_chars, a.cache_root)
        verify_pool(pool, keep)
    clean = [p for p in pool if p["key"] not in used]
    need = a.n - len(keep)
    print(f"pool {len(pool)} from {data_dir}:{hf_split}, {len(clean)} after dedupe, need {need}")
    if need > len(clean):
        raise SystemExit(f"only {len(clean)} unused turns available, need {need}")
    sel = random.Random(a.seed).sample(clean, need)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for p in sel:
        keep.append({"prompt_ids": [int(i) for i in chat_ids(tok, p["user"], system=SYSTEM)],
                     "prompt_text": p["user"], "source": p["source"],
                     # deliberately absent: teacher_ids, base_ids, differs. This file is
                     # for --redteam-data only; the CE evaluations must not read it.
                     "guardrail_only": True})
    out.write_text(json.dumps(keep) + "\n")
    n_new = sum(1 for r in keep if r.get("guardrail_only"))
    print(f"wrote {out}: {len(keep)} items, first {len(keep)-n_new} carried over, {n_new} new")
    # The check load_redteam's caller performs, done here so a mismatch surfaces now.
    bad = [i for i, r in enumerate(keep)
           if r.get("prompt_ids") and list(chat_ids(tok, r["prompt_text"], system=SYSTEM)) != list(r["prompt_ids"])]
    print(f"tokenisation self-check: {len(keep)-len(bad)} of {len(keep)} reproduce their prompt_ids")
    if bad:
        print(f"  MISMATCH at {bad[:5]}{'...' if len(bad) > 5 else ''} -- these were tokenised by "
              f"a different tokenizer revision; the harness will flag them too")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
