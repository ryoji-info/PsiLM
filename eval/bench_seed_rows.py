#!/usr/bin/env python3
"""Start a guard-rail run from rows another run already generated.

The base arm is deterministic and a recorded arm does not change, so a run that
ADDS an arm to a recorded comparison need not generate the others again: seed its
rows file with the recorded rows and launch it with --resume, and the harness
generates only what is missing. Every borrowed row says where it came from
(`borrowed_from`), so nothing borrowed is mistaken for something regenerated.

--skip-first N leaves the first N prompts (in task order) unseeded, so the run
regenerates them: that is the reproduction check, made by the run itself
(eval/const_arm_compare.py --repro).

The runs recorded before 2026-09-22 have their KLs on the legacy read; a run on the
corrected read cannot hold both. --klpool-rows gives the rescored values
(eval/kl_rescore_all.py): each borrowed row's KL is replaced by its corrected-read
value, and a row with no rescored value loses its KL rather than keeping a legacy
one. A row whose KL is already on the corrected read (pool "prompt") keeps it.

A borrowed row's answer is read again by today's parser (bench_common.score) from
its text, unless the text was stored truncated: a run that adds an arm must score
every arm with one parser, and rows recorded before 2026-09-17 carry the answers
of a parser that has since been repaired. A row whose answer changes says what it
was (`rescored_from`).

  python eval/bench_seed_rows.py --from-tag const_qwen35_all_rt400 --arms base,psilm \\
      --klpool-rows results/bench/const_qwen35_all_rt400_klpool_guardrail.rows.jsonl \\
      --tasks-cache results/bench/tasks_rt400_qwen35_n400.json --skip-first 40 \\
      --out-tag const_qwen35_all_rt400_repro
  python eval/bench_seed_rows.py --self-test
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def read_rows(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def rescore(r):
    """The row with its answer as today's parser reads it."""
    from bench_common import score
    if r.get("protocol") in (None, "freeform") or r.get("text_truncated") or "gold" not in r:
        return r
    pred, ok = score(r["protocol"], r.get("text") or "", r["gold"])
    if (pred, bool(ok)) == (r.get("pred"), bool(r.get("ok"))):
        return r
    return {**r, "pred": pred, "ok": bool(ok), "rescored_from": {"pred": r.get("pred"), "ok": r.get("ok")}}


def seed(rows, arms, order, skip_first=0, klpool=None, source="?"):
    """Borrowed copies of `rows` for `arms`, in task order, minus the first prompts."""
    keep = set(order[skip_first:])
    # `kl_prompt` IS the corrected read, whether or not the entry repeats it in a label
    rescored = {(r["dataset"], r["qid"], r["arm"]): {**r["kl_prompt"], "pool": "prompt"}
                for r in (klpool or []) if r.get("kl_prompt") and r["kl_prompt"].get("mean") is not None}
    by = {(r["dataset"], r["qid"], r["arm"]): r for r in rows}
    out, dropped_kl = [], 0
    for ds, qid in order:
        if (ds, qid) not in keep:
            continue
        for arm in arms:
            r = by.get((ds, qid, arm))
            if r is None:
                raise SystemExit(f"{source}: no {arm} row for {qid}")
            if arm == "base" and "gen_ids" not in r:
                raise SystemExit(f"{source}: the base row of {qid} has no gen_ids (was it run with --kl?)")
            r = {**rescore(r), "borrowed_from": source}
            if r.get("kl") is not None:
                k = rescored.get((ds, qid, arm))
                if k is not None:
                    r["kl"] = k
                elif r["kl"].get("pool") != "prompt":           # a run on the corrected read keeps its own
                    del r["kl"]
                    dropped_kl += 1
            out.append(r)
    return out, dropped_kl


def task_order(cache_path):
    return [(t["dataset"], t["qid"]) for t in json.loads(Path(cache_path).read_text())["tasks"]]


def self_test():
    rows, kl = [], []
    order = [("redteam", f"q{i}") for i in range(6)]
    for ds, q in order:
        rows.append({"dataset": ds, "qid": q, "arm": "base", "text": "b", "gen_ids": [1, 2]})
        rows.append({"dataset": ds, "qid": q, "arm": "psilm", "text": "p", "kl": {"mean": 0.2}})
        rows.append({"dataset": ds, "qid": q, "arm": "zeroed", "text": "b", "kl": {"mean": 0.0}})
        if q != "q5":
            kl.append({"dataset": ds, "qid": q, "arm": "psilm",
                       "kl_prompt": {"mean": 0.19, "pool": "prompt"}, "kl_full": {"mean": 0.2}})
    out, dropped = seed(rows, ["base", "psilm"], order, skip_first=2, klpool=kl, source="rec")
    assert [(r["qid"], r["arm"]) for r in out] == [(f"q{i}", a) for i in range(2, 6) for a in ("base", "psilm")]
    assert all(r["borrowed_from"] == "rec" for r in out) and dropped == 1
    assert all(r["kl"] == {"mean": 0.19, "pool": "prompt"} for r in out if r["arm"] == "psilm" and r["qid"] != "q5")
    assert "kl" not in next(r for r in out if r["arm"] == "psilm" and r["qid"] == "q5")    # never a legacy KL
    assert "kl" not in out[0] and out[0]["gen_ids"] == [1, 2]
    assert "borrowed_from" not in rows[0]                                      # the record is not touched
    own = [{**r, "kl": {"mean": 0.3, "pool": "prompt"}} if r["arm"] == "psilm" else r for r in rows]
    kept, none = seed(own, ["base", "psilm"], order, source="rec")           # already on the corrected read
    assert none == 0 and all(r["kl"] == {"mean": 0.3, "pool": "prompt"} for r in kept if r["arm"] == "psilm")
    nul = [{**r, "kl": None} if r["arm"] == "base" else r for r in rows]      # a base row records no KL
    assert seed(nul, ["base"], order, source="rec")[1] == 0
    # a rescored value written without its label is still the corrected read
    bare = [{**k, "kl_prompt": {"mean": 0.19}} for k in kl] + [
        {"dataset": "redteam", "qid": "q5", "arm": "psilm", "kl_prompt": {"mean": None}}]
    got, lost = seed(rows, ["psilm"], order, klpool=bare, source="rec")
    assert lost == 1 and [r.get("kl") for r in got] == [{"mean": 0.19, "pool": "prompt"}] * 5 + [None]
    # the answers are today's parser's: a bare letter is an answer, a truncated text is left alone
    mm = [{"dataset": "mmlu", "qid": "m0", "arm": "base", "protocol": "letter", "gold": "B", "text": "B",
           "pred": None, "ok": False, "gen_ids": [1]},
          {"dataset": "mmlu", "qid": "m1", "arm": "base", "protocol": "letter", "gold": "B", "text": "B",
           "pred": None, "ok": False, "gen_ids": [1], "text_truncated": True},
          {"dataset": "mmlu", "qid": "m2", "arm": "base", "protocol": "letter", "gold": "C", "text": "Answer: C",
           "pred": "C", "ok": True, "gen_ids": [1]}]
    got, _ = seed(mm, ["base"], [("mmlu", "m0"), ("mmlu", "m1"), ("mmlu", "m2")], source="rec")
    assert (got[0]["pred"], got[0]["ok"], got[0]["rescored_from"]) == ("B", True, {"pred": None, "ok": False})
    assert (got[1]["pred"], got[1]["ok"]) == (None, False) and "rescored_from" not in got[1]
    assert got[2]["ok"] is True and "rescored_from" not in got[2] and mm[0]["pred"] is None
    none, _ = seed(rows, ["base"], order, skip_first=6, source="rec")
    assert none == []
    for bad_arms, bad_rows in ((["fixed"], rows), (["base"], [{k: v for k, v in r.items() if k != "gen_ids"}
                                                             for r in rows])):
        try:
            seed(bad_rows, bad_arms, order, source="rec")
            raise AssertionError("a missing arm or a base row without its tokens was seeded")
        except SystemExit:
            pass
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a_guardrail.rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp / "tasks.json").write_text(json.dumps({"key": "k", "tasks": [{"dataset": d, "qid": q} for d, q in order]}))
    a = ["--from-tag", "a", "--arms", "base", "--tasks-cache", str(tmp / "tasks.json"),
         "--out-tag", "b", "--bench-dir", str(tmp)]
    assert run(parser().parse_args(a)) == 0
    assert len(read_rows(tmp / "b_guardrail.rows.jsonl")) == 6
    try:
        run(parser().parse_args(a))                                            # never over an existing run
        raise AssertionError("an existing rows file was overwritten")
    except SystemExit:
        pass
    print("[self-test] eval/bench_seed_rows.py: all assertions passed")
    return 0


def run(a):
    bench = Path(a.bench_dir)
    out = bench / f"{a.out_tag}_guardrail.rows.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists: a run is seeded once, before it starts")
    rows = read_rows(bench / f"{a.from_tag}_guardrail.rows.jsonl")
    klpool = read_rows(a.klpool_rows) if a.klpool_rows else None
    seeded, dropped = seed(rows, a.arms.split(","), task_order(a.tasks_cache), a.skip_first, klpool,
                           a.from_tag)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in seeded))
    tmp.replace(out)
    print(f"[seed] {len(seeded)} rows of {a.from_tag} ({a.arms}) -> {out}; the first {a.skip_first} "
          f"prompts are left to regenerate; {dropped} legacy KLs dropped", flush=True)
    return 0


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-tag")
    ap.add_argument("--arms", default="base")
    ap.add_argument("--tasks-cache")
    ap.add_argument("--klpool-rows", default=None)
    ap.add_argument("--skip-first", type=int, default=0)
    ap.add_argument("--out-tag")
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--self-test", action="store_true")
    return ap


def main():
    ap = parser()
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.from_tag and a.tasks_cache and a.out_tag):
        ap.error("--from-tag, --tasks-cache and --out-tag are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
