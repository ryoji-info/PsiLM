"""Self-distillation data for the constitution bridges.

The teacher and the student are the SAME frozen backbone. The teacher reads a
verbatim excerpt of Claude's constitution (Anthropic, CC0 --
``data/constitution/system_excerpt.md``) as its system prompt; the student gets
the ordinary ``"You are a helpful assistant."``. Whatever the excerpt changes in
the continuation is the only signal the bridges can learn, so this script
records both continuations per prompt and reports how often they differ -- if
the teacher_differs rate is near zero the excerpt is inert at this scale and no
amount of bridge training will find anything. That check is the point of the
smoke run.

Prompts are the FIRST human turn of ``Anthropic/hh-rlhf`` dialogues (the
``chosen`` column; only the human turn is used, never the assistant's reply, so
nothing of the preference data's own answers enters the targets):

    train, val        harmless-base TRAIN, disjoint seeded samples
    test              harmless-base TEST     -> the guard-rail's redteam set
    helpful_test      helpful-base TEST      -> ordinary requests, the
                                                no-harm-on-chat control

Splits are deduplicated against each other by normalised text, and the two test
splits are additionally cleared of anything appearing anywhere in harmless-base
train, so a prompt the bridges trained on can never be scored as held out.

Cost. The teacher's system block is ~3.4k tokens and is identical for every
prompt, so it is prefilled ONCE into an mlx_lm prompt cache and copied per
prompt: 2.05 s/item against 5.21 s/item uncached on the 0.5B, i.e. 75 minutes
instead of 190 for 2,400 items. The copy has to be exact:
``mlx_lm.models.cache.KVCache`` writes into its own arrays, so a cache handed to
``generate_step`` is consumed. ``copy.deepcopy`` produces independent arrays on
this mlx (checked at startup, including that a write to the copy leaves the
original alone, with a re-prefill-from-``state`` fallback).

Two things about the cache are then verified on the first --verify-n prompts.
(1) The shared prefix must be a TOKEN prefix of each full prompt, or the cache
describes a different string than the one being decoded; this is a hard
assertion. (2) The cached continuation is compared against an uncached run. It
usually matches exactly, but not always: the uncached path prefills ~3.4k tokens
in 2048-token chunks while the cached path prefills the prefix and the user turn
separately, and float addition is not associative, so a greedy near-tie can flip
and the continuations diverge from there (observed once in three prompts on the
0.5B, at token 48 of 160). That is reported -- count and first divergence index,
in the log and in the stats file -- rather than asserted away: every item in a
given run is built the same way, so the data is internally consistent, and a
target that hinges on a last-bit tie was arbitrary in either direction. Pass
--no-cache for the slow, reference path.

Outputs (``--out-dir``, default ``data``):
    constitution_{train,val,test,helpful_test}_<tag>.json
        [{prompt_ids, teacher_ids, base_ids, prompt_text, source, differs}]
        prompt_ids: chat_ids(hf_tok, user, system=SYSTEM)  -- the STUDENT prompt
        teacher_ids/base_ids: greedy continuations, the eos id kept when the
        model stopped on it (the no-harm convention of eval/build_noharm.py)
        prompt_text: the raw human turn, so the benchmark can rebuild the ids
        differs: teacher_ids[:16] != base_ids[:16]
    constitution_<tag>_stats.json    per-split rates, timing, provenance

Resume: one jsonl per split under ``--work-dir`` (default results/constitution),
keyed by source; the JSON lists are assembled from them at the end. Re-running
with the same tag continues where it stopped.

Usage
  .venv/bin/python eval/build_constitution_data.py --tag smoke --out-dir results/constitution \
      --n-train 8 --n-val 2 --n-test 2 --n-helpful-test 2
  # full dev-backbone run: results/constitution/build_data_qwen0.5b.sh
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.bench_common import (  # noqa: E402
    REFUSAL_KEYWORDS, SYSTEM, chat_ids, eos_id_set, is_refusal,
)

TEACHER_PREAMBLE = ("Follow the principles below, excerpted verbatim from Claude's "
                    "constitution (Anthropic, CC0), in everything you say.\n\n")

DEFAULT_EXCERPT = "data/constitution/system_excerpt.md"
SPLITS = ("train", "val", "test", "helpful_test")
# (data_dir, split) each output split is drawn from
SOURCES = {"train": ("harmless-base", "train"), "val": ("harmless-base", "train"),
           "test": ("harmless-base", "test"), "helpful_test": ("helpful-base", "test")}


# ----------------------------------------------------------------------------
# the excerpt -> the teacher's system prompt
# ----------------------------------------------------------------------------

def excerpt_body(path: str) -> str:
    """The excerpt's sections, from its first '### ' heading on.

    system_excerpt.md opens with a Markdown title and one paragraph of framing
    that were written for the repository, not copied from the constitution.
    Feeding them to the teacher would put a sentence about "the teacher's system
    prompt" inside the teacher's system prompt, so the body starts at the first
    section heading. '#### Hard constraints' does not match '### ' (its fourth
    character is '#'), so the first match is the first real section."""
    lines = Path(path).read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("### "):
            return "\n".join(lines[i:]).rstrip() + "\n"
    raise SystemExit(f"{path}: no '### ' section heading found")


def teacher_system(path: str) -> str:
    """The teacher's system prompt in one call, for anything that has to
    reproduce this teacher later (another backbone, an ablation on a different
    excerpt) without re-deriving the preamble."""
    return TEACHER_PREAMBLE + excerpt_body(path)


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------

def first_human_turn(dialogue: str) -> Optional[str]:
    """The text of the first '\\n\\nHuman: ...' turn, up to the first
    '\\n\\nAssistant:'. hh-rlhf dialogues always open with a human turn."""
    i = dialogue.find("\n\nHuman:")
    if i < 0:
        return None
    rest = dialogue[i + len("\n\nHuman:"):]
    j = rest.find("\n\nAssistant:")
    turn = (rest if j < 0 else rest[:j]).strip()
    return turn or None


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def load_pool(data_dir: str, split: str, min_chars: int, max_chars: int) -> List[Dict[str, str]]:
    """Deduplicated first human turns of one hh-rlhf subset, in dataset order
    (the seeded sample is taken from this list, so it is reproducible)."""
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split=split)
    seen, out = set(), []
    for i, row in enumerate(ds):
        turn = first_human_turn(row["chosen"])
        if turn is None or not (min_chars <= len(turn) <= max_chars):
            continue
        key = _norm(turn)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": f"hh:{data_dir}:{split}:{i}", "user": turn, "key": key})
    return out


def reused_prompt_sets(tag: str, out_dir: str, splits) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any]]:
    """Another backbone's four prompt lists, verbatim and in order, from its data files.

    A new backbone trained on the same prompts is comparable item for item with the
    one that chose them, and nothing here imports `datasets` (which has killed a
    large MLX load in the same process on this machine). The disjointness the
    original run proved carries over, since the lists are identical."""
    sets = {}
    for s in splits:
        f = Path(out_dir) / f"constitution_{s}_{tag}.json"
        if not f.exists():
            raise SystemExit(f"--prompts-from {tag}: no {f}")
        rows = json.loads(f.read_text())
        sets[s] = [{"source": r["source"], "user": r["prompt_text"], "key": _norm(r["prompt_text"])}
                   for r in rows]
    return sets, {"reused_from": tag, "pool": {s: len(v) for s, v in sets.items()}, "excluded": {}}


def build_prompt_sets(args) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any]]:
    """The four prompt lists plus the bookkeeping that proves they are disjoint."""
    if getattr(args, "prompts_from", None):
        return reused_prompt_sets(args.prompts_from, args.out_dir, SPLITS)
    rng = random.Random(args.seed)
    harm_train = load_pool("harmless-base", "train", args.min_chars, args.max_chars)
    # every normalised harmless-base TRAIN turn, not only the sampled ones: a
    # held-out prompt that also exists in the pool the bridges were trained from
    # is not held out, even if this run happened not to sample it
    train_all = {p["key"] for p in harm_train}
    need = args.n_train + args.n_val
    if need > len(harm_train):
        raise SystemExit(f"harmless-base train has {len(harm_train)} usable turns, need {need}")
    picked = rng.sample(harm_train, need)
    sets = {"train": picked[:args.n_train], "val": picked[args.n_train:]}
    stats = {"pool": {"harmless_train": len(harm_train)}, "excluded": {}}
    used = {p["key"] for p in picked}
    for split, n in (("test", args.n_test), ("helpful_test", args.n_helpful_test)):
        data_dir, hf_split = SOURCES[split]
        pool = load_pool(data_dir, hf_split, args.min_chars, args.max_chars)
        stats["pool"][f"{data_dir}_{hf_split}"] = len(pool)
        clean = [p for p in pool if p["key"] not in train_all and p["key"] not in used]
        stats["excluded"][split] = len(pool) - len(clean)
        if n > len(clean):
            raise SystemExit(f"{split}: {len(clean)} usable turns after dedupe, need {n}")
        sel = rng.sample(clean, n)
        used |= {p["key"] for p in sel}
        sets[split] = sel
    return {k: sets[k] for k in SPLITS}, stats


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------

def greedy(model, ids: List[int], max_new: int, eos_ids: set, prompt_cache=None) -> List[int]:
    """Greedy continuation ids. The eos id is kept at the end when the model
    stopped on it, so a trainer sees where the backbone chose to stop (the
    convention of eval/build_noharm.py). prompt_cache is CONSUMED."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    out: List[int] = []
    for tok, _ in generate_step(mx.array(list(ids)), model, max_tokens=max_new,
                                prompt_cache=prompt_cache):
        t = int(tok)                       # mlx-lm 0.31 yields a python int, older ones an array
        out.append(t)
        if t in eos_ids:
            break
    return out


def prefix_ids_of(hf_tok, system: str) -> List[int]:
    """The templated system block: the longest common prefix of two chat
    promptings that differ only in the user text.

    Derived from the template rather than written out, so it stays correct for
    any backbone's chat format. Two sentinels with different first characters
    cannot share a first user token, so the common prefix cannot reach into the
    user turn."""
    a = chat_ids(hf_tok, "Alpha sentinel.", system=system)
    b = chat_ids(hf_tok, "Zeta sentinel.", system=system)
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return a[:k]


def make_cloner(model, prefix_cache):
    """(clone_fn, how). Each prompt needs its own copy of the prefilled prefix:
    KVCache.update_and_fetch writes into its own arrays. deepcopy yields
    independent arrays on this mlx; if it ever stops doing so, rebuild a cache
    and re-seed it from the saved state (the state is sliced to the offset, so
    the first write reallocates and never touches the shared arrays)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    try:
        probe = copy.deepcopy(prefix_cache)
        assert [c.offset for c in probe] == [c.offset for c in prefix_cache]
        k0 = prefix_cache[0].keys
        probe[0].keys[..., 0, :] = probe[0].keys[..., 0, :] + 1.0
        mx.eval(probe[0].keys, k0)
        assert not bool(mx.all(probe[0].keys[..., 0, :] == k0[..., 0, :]).item()), "aliased"
        del probe
        return (lambda: copy.deepcopy(prefix_cache)), "deepcopy"
    except Exception as e:                        # noqa: BLE001 - any failure -> the safe path
        state = [c.state for c in prefix_cache]

        def _copy(s):
            """A fresh buffer per prompt. Handing the SAME arrays to every prompt is
            safe only if no layer writes into them in place -- which a recurrent
            GatedDeltaNet layer does, so on Qwen3.5 the second prompt would start
            from a prefix the first one had already overwritten. `+ 0` is a real
            elementwise op, so it allocates rather than aliasing."""
            if s is None:
                return None
            if isinstance(s, (tuple, list)):
                return type(s)(_copy(x) for x in s)
            return s + 0

        def restore():
            fresh = make_prompt_cache(model)
            for c, s in zip(fresh, state):
                c.state = _copy(s)
            return fresh
        return restore, f"state-restore ({type(e).__name__}: {str(e)[:60]})"


# ----------------------------------------------------------------------------
# per-split work
# ----------------------------------------------------------------------------

def split_items(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            done[r["source"]] = r
    return done


def summarize_split(items: List[Dict[str, Any]], hf_tok, eos_ids: set) -> Dict[str, Any]:
    n = max(1, len(items))
    def rate(f):                                                  # noqa: E306
        return round(sum(bool(f(it)) for it in items) / n, 4)
    def refus(key):                                               # noqa: E306
        return round(sum(is_refusal(hf_tok.decode(it[key], skip_special_tokens=True))
                         for it in items) / n, 4)
    return {"n": len(items),
            "teacher_differs_rate": rate(lambda it: it["differs"]),
            "refusal_rate_base": refus("base_ids"),
            "refusal_rate_teacher": refus("teacher_ids"),
            "mean_len_base": round(sum(len(it["base_ids"]) for it in items) / n, 1),
            "mean_len_teacher": round(sum(len(it["teacher_ids"]) for it in items) / n, 1),
            "eos_rate_base": rate(lambda it: it["base_ids"] and it["base_ids"][-1] in eos_ids),
            "eos_rate_teacher": rate(lambda it: it["teacher_ids"] and it["teacher_ids"][-1] in eos_ids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-helpful-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="qwen0.5b")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--excerpt", default=DEFAULT_EXCERPT)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--out-dir", default="data",
                    help="where the JSON lists and the stats go (use results/constitution "
                         "for smoke runs: data/ is tracked)")
    ap.add_argument("--work-dir", default="results/constitution",
                    help="resume jsonl per split (untracked scratch)")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--no-cache", action="store_true",
                    help="prefill the teacher's system block per prompt (the slow reference path)")
    ap.add_argument("--verify-n", type=int, default=3,
                    help="prompts on which the cached teacher path is checked against an "
                         "uncached run before the real work starts")
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--prompts-from", default=None, metavar="TAG",
                    help="reuse the prompt lists of data/constitution_<split>_TAG.json verbatim "
                         "instead of sampling hh-rlhf (same prompts, a new backbone's continuations)")
    ap.add_argument("--dry-run", action="store_true", help="prompts only, no weights")
    args = ap.parse_args()

    splits = [s for s in args.splits.split(",") if s]
    for s in splits:
        if s not in SPLITS:
            raise SystemExit(f"unknown split {s} (of {SPLITS})")
    out_dir, work_dir = Path(args.out_dir), Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    body = excerpt_body(args.excerpt)
    tsys = TEACHER_PREAMBLE + body                # == teacher_system(args.excerpt)
    sets, pool_stats = build_prompt_sets(args)
    print(f"[prompts] " + " ".join(f"{k}={len(v)}" for k, v in sets.items())
          + f"  pools={pool_stats['pool']} excluded={pool_stats['excluded']}", flush=True)

    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    prefix = prefix_ids_of(hf_tok, tsys)
    excerpt_tokens = len(hf_tok.encode(body, add_special_tokens=False))
    print(f"[excerpt] {args.excerpt}: {len(body.split())} words, {excerpt_tokens} tokens; "
          f"teacher system block {len(prefix)} tokens "
          f"(sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]})", flush=True)
    if args.dry_run:
        for s in splits:
            print(f"\n### {s}")
            for p in sets[s][:3]:
                print(f"  {p['source']}: {p['user'][:200]!r}")
        return

    from psilm.mlx.gemma_loader import load_backbone_any
    t_load = time.time()
    _, model, mlx_tok = load_backbone_any(args.model)
    eos_ids = eos_id_set(hf_tok, mlx_tok)
    print(f"[model] {args.model} in {time.time() - t_load:.1f}s | eos {sorted(eos_ids)}", flush=True)

    clone, how = None, "off (--no-cache)"
    if not args.no_cache:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        t0 = time.time()
        prefix_cache = make_prompt_cache(model)
        model(mx.array([list(prefix)]), cache=prefix_cache)
        # .state, not .keys/.values: a hybrid backbone's recurrent layers get an
        # ArraysCache, which has neither (Qwen3.5 9B, 2026-09-13). Every cache
        # class in mlx_lm.models.cache implements .state. make_cloner's own probe
        # still reads .keys, but it is inside a try/except whose fallback is the
        # state-restore cloner, so a hybrid model lands there and says so.
        mx.eval([c.state for c in prefix_cache])
        clone, how = make_cloner(model, prefix_cache)
        print(f"[cache] {len(prefix)} prefix tokens prefilled in {time.time() - t0:.1f}s, "
              f"clone strategy: {how}", flush=True)

    def teacher_of(user: str, cached: bool) -> Tuple[List[int], bool]:
        """(continuation ids, used_the_cache). The cache is only valid when the
        prompt's own tokenisation starts with the prefilled prefix ids."""
        ids = chat_ids(hf_tok, user, system=tsys)
        if cached and clone is not None and ids[:len(prefix)] == prefix:
            return greedy(model, ids[len(prefix):], args.max_new, eos_ids, clone()), True
        return greedy(model, ids, args.max_new, eos_ids), False

    # ---- the cache equivalence check, before any real work -------------------
    check = {"strategy": how, "n": 0, "same": 0, "prefix_ok": 0, "first_diff": []}
    if clone is not None and args.verify_n > 0:
        probe = [p for s in splits for p in sets[s]][:args.verify_n]
        for p in probe:
            ids = chat_ids(hf_tok, p["user"], system=tsys)
            ok_prefix = ids[:len(prefix)] == prefix
            got, used = teacher_of(p["user"], cached=True)
            ref, _ = teacher_of(p["user"], cached=False)
            check["n"] += 1
            check["prefix_ok"] += int(ok_prefix and used)
            check["same"] += int(got == ref)
            if got != ref:
                d = next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
                check["first_diff"].append({"source": p["source"], "at": d,
                                            "n_cached": len(got), "n_uncached": len(ref)})
        print(f"[cache check] prefix ids match as a prefix: {check['prefix_ok']}/{check['n']}; "
              f"cached == uncached tokens: {check['same']}/{check['n']}"
              + (f"  first divergences {check['first_diff']}" if check["first_diff"] else ""),
              flush=True)
        if check["same"] != check["n"]:
            print("[WARN] the cached and uncached prefills disagree on a greedy tie "
                  "(different chunking of the same sum, not a cache bug): the run stays "
                  "self-consistent, but its targets are not reproducible by an uncached run. "
                  "--no-cache builds the reference version at ~2.5x the cost.", flush=True)
        if check["prefix_ok"] != check["n"]:
            raise SystemExit("the teacher's system block is not a token prefix of the full "
                             "prompt: the shared cache would be invalid")

    # ---- generate ------------------------------------------------------------
    t0, n_new, gen_tok, uncached = time.time(), 0, 0, 0
    item_sec: List[float] = []
    assembled: Dict[str, List[Dict[str, Any]]] = {}
    for s in splits:
        jl = work_dir / f"{args.tag}_{s}.jsonl"
        done = split_items(jl)
        todo = [p for p in sets[s] if p["source"] not in done]
        print(f"[{s}] {len(done)} done, {len(todo)} to go -> {jl}", flush=True)
        with jl.open("a") as fh:
            for k, p in enumerate(todo):
                t_item = time.time()
                p_ids = chat_ids(hf_tok, p["user"], system=SYSTEM)
                base = greedy(model, p_ids, args.max_new, eos_ids)
                teach, used = teacher_of(p["user"], cached=True)
                uncached += int(not used)
                # "sec" lives in the resume jsonl only, never in the data JSON:
                # it is what lets a resumed run still report real timing instead
                # of the near-zero wall clock of an assembly-only pass
                rec = {"prompt_ids": list(map(int, p_ids)),
                       "teacher_ids": list(map(int, teach)),
                       "base_ids": list(map(int, base)),
                       "prompt_text": p["user"], "source": p["source"],
                       "differs": teach[:16] != base[:16],
                       "sec": round(time.time() - t_item, 3)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[p["source"]] = rec
                n_new += 1
                gen_tok += len(base) + len(teach)
                if (k + 1) % args.print_every == 0 or k == len(todo) - 1:
                    el = time.time() - t0
                    print(f"  [{s} {k + 1}/{len(todo)}] {el / max(1, n_new):.2f}s/item, "
                          f"{gen_tok / max(1e-9, el):.1f} tok/s", flush=True)
        assembled[s] = [{k: v for k, v in done[p["source"]].items() if k != "sec"}
                        for p in sets[s] if p["source"] in done]
        item_sec += [done[p["source"]]["sec"] for p in sets[s]
                     if p["source"] in done and "sec" in done[p["source"]]]
        out = out_dir / f"constitution_{s}_{args.tag}.json"
        out.write_text(json.dumps(assembled[s]))
        print(f"[{s}] {len(assembled[s])} items -> {out}", flush=True)

    el = time.time() - t0
    # per-item cost of the items actually generated, whenever it is known: an
    # assembly-only pass over an earlier run's jsonl would otherwise report a
    # near-zero second per item and a nonsense estimate for the full run
    per_item = (sum(item_sec) / len(item_sec)) if item_sec else (el / n_new if n_new else None)
    total_requested = args.n_train + args.n_val + args.n_test + args.n_helpful_test
    stats = {"tag": args.tag, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
             "model": args.model, "hf_tokenizer": args.hf_tokenizer,
             "config": {k: v for k, v in vars(args).items()},
             "excerpt": {"path": args.excerpt, "words": len(body.split()),
                         "tokens": excerpt_tokens,
                         "sha256": hashlib.sha256(body.encode()).hexdigest(),
                         "teacher_system_tokens": len(hf_tok.encode(tsys, add_special_tokens=False)),
                         "prefix_tokens": len(prefix)},
             "prompts": pool_stats,
             "cache_check": check, "uncached_fallbacks": uncached,
             "refusal_keywords": list(REFUSAL_KEYWORDS),
             "splits": {s: summarize_split(assembled[s], hf_tok, eos_ids) for s in splits},
             "timing": {"items_generated": n_new, "sec": round(el, 1),
                        "items_timed": len(item_sec),
                        "sec_per_item": (round(per_item, 2) if per_item else None),
                        "gen_tokens": gen_tok,
                        "tok_per_s": (round(gen_tok / el, 1) if n_new else None),
                        "full_run_items": total_requested,
                        "full_run_estimate_minutes": (round(per_item * total_requested / 60, 1)
                                                      if per_item else None)}}
    sp = out_dir / f"constitution_{args.tag}_stats.json"
    sp.write_text(json.dumps(stats, indent=1))
    print("\n" + json.dumps({k: stats[k] for k in ("splits", "timing", "cache_check",
                                                   "uncached_fallbacks")}, indent=1))
    print(f"\nCONSTITUTION DATA -> {sp}", flush=True)


if __name__ == "__main__":
    main()
