#!/usr/bin/env python3
"""One arm's generated behaviour against a reference arm's, across guard-rail runs.

Two questions this project asks of the 400 red-team prompts are the same
comparison: does the bridge's behaviour survive when what it reads is taken away
(the `fixed` arms), and does a bridge trained with no partner behave as the
partner's bridge does. Both set a candidate arm beside a REFERENCE arm (the
partner's psilm) and a BASE arm, prompt by prompt.

For each candidate, on the keyword instrument (the adjudicated labels are a
separate pass, see `to_adjudicate`):

  vs_base      paired counts to and from refusal and the exact McNemar p: is the
               candidate's shift the reference's shift (the reference's is printed
               beside it)
  vs_reference the same pairing against the reference arm: do the two arms DIFFER
  flips        the prompts each arm moves against base, and how many they share
  divergence   KL(base || arm) per token, where the run recorded it

  identity     how many of the candidate's generations are the reference's, token
               for token. A WORKLOAD figure, not a verdict: greedy decoding over
               128 tokens amplifies rounding, so a candidate that carries the same
               information still differs on many prompts. `identity_reference`
               gives the share of items left wholly unchanged, teacher-forced, by
               changes that carry no information (eval/constitution_compress.py).
  to_adjudicate  the prompts whose text is not the reference's: what a judge has
               to read; the rest inherit the reference's label.

--repro RUN,RECORDED checks that today's harness regenerates the recorded run:
the rows of RUN that were generated (not borrowed) against RECORDED's.

  python eval/const_arm_compare.py --base const_qwen35_all_rt400_fixed:base \\
      --ref const_qwen35_all_rt400_fixed:psilm \\
      --arms const_qwen35_all_rt400_fixed:fixed,const_qwen35_all_rt400_fixed:fixedzero \\
      --out results/constitution/fixed_tokens_rt400.json
  python eval/const_arm_compare.py --self-test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from const_refusal_mcnemar import mcnemar_exact   # noqa: E402


def load(path, dataset="redteam"):
    by = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["dataset"] == dataset:
                by.setdefault(r["arm"], {})[r["qid"]] = r
    return by


def refused(r):
    return bool(r["meta"]["refusal"])


def paired(a, b, qids):
    """a -> b: prompts b turns to refusal, and from it."""
    to = sorted(q for q in qids if refused(b[q]) and not refused(a[q]))
    frm = sorted(q for q in qids if refused(a[q]) and not refused(b[q]))
    return {"n": len(qids), "to_refusal": len(to), "from_refusal": len(frm), "net": len(to) - len(frm),
            "discordant": len(to) + len(frm), "p": round(mcnemar_exact(len(frm), len(to)), 5),
            "to_refusal_qids": to, "from_refusal_qids": frm}


def kl_block(rows, qids):
    v = [rows[q]["kl"] for q in qids if rows[q].get("kl")]
    if len(v) != len(qids):
        return None
    return {"mean": round(sum(k["mean"] for k in v) / len(v), 5),
            "pool": sorted({k.get("pool", "full") for k in v})}


def compare(base, ref, cand, name):
    qids = sorted(set(base) & set(ref) & set(cand))
    if not qids or len(qids) != len(cand):
        raise SystemExit(f"{name}: {len(cand)} prompts, {len(qids)} of them in the base and the "
                         f"reference ({len(base)}, {len(ref)}): the runs must share a task cache")
    n = len(qids)
    vb, rb, vr = paired(base, cand, qids), paired(base, ref, qids), paired(ref, cand, qids)
    fl = lambda p: set(p["to_refusal_qids"]) | set(p["from_refusal_qids"])          # noqa: E731
    differs = [q for q in qids if cand[q]["text"] != ref[q]["text"]]
    out = {"arm": name, "n": n,
           "refusal_rate": {"base": round(sum(refused(base[q]) for q in qids) / n, 4),
                            "reference": round(sum(refused(ref[q]) for q in qids) / n, 4),
                            "arm": round(sum(refused(cand[q]) for q in qids) / n, 4)},
           "mean_tokens": {"base": round(sum(base[q]["n_gen"] for q in qids) / n, 1),
                           "reference": round(sum(ref[q]["n_gen"] for q in qids) / n, 1),
                           "arm": round(sum(cand[q]["n_gen"] for q in qids) / n, 1)},
           "vs_base": vb, "reference_vs_base": rb, "vs_reference": vr,
           "flips": {"reference": len(fl(rb)), "arm": len(fl(vb)), "shared": len(fl(rb) & fl(vb)),
                     "to_refusal_shared": len(set(rb["to_refusal_qids"]) & set(vb["to_refusal_qids"]))},
           "divergence": {"reference": kl_block(ref, qids), "arm": kl_block(cand, qids)},
           "identity": {"arm_is_reference": n - len(differs), "arm_is_base":
                        sum(cand[q]["text"] == base[q]["text"] for q in qids),
                        "reference_is_base": sum(ref[q]["text"] == base[q]["text"] for q in qids), "of": n},
           "to_adjudicate": differs}
    return out


def same_generation(r, rec):
    return (r["text"] == rec["text"] and r.get("n_gen") == rec.get("n_gen")
            and r.get("stopped_eos") == rec.get("stopped_eos")
            and r.get("gen_ids", 0) == rec.get("gen_ids", r.get("gen_ids", 0)))


def reproduction(run, recorded, klpool=None, tol=1e-4):
    """Rows the run GENERATED (not borrowed), against the recorded run's: the same
    text, length and stop, and (given the rescored KLs of the recorded run) the
    same KL on the corrected read."""
    out = {}
    for arm in sorted(set(run) & set(recorded)):
        mine = {q: r for q, r in run[arm].items() if not r.get("borrowed_from") and q in recorded[arm]}
        same = sum(same_generation(r, recorded[arm][q]) for q, r in mine.items())
        out[arm] = {"generated": len(mine), "identical": same,
                    "borrowed": len(run[arm]) - len([1 for r in run[arm].values() if not r.get("borrowed_from")])}
        if klpool is not None and arm != "base":
            pairs = [(r["kl"]["mean"], klpool[(q, arm)]) for q, r in mine.items()
                     if r.get("kl") and (q, arm) in klpool]
            out[arm]["kl_compared"] = len(pairs)
            out[arm]["kl_identical"] = sum(abs(a - b) <= tol for a, b in pairs)
    arms = [v for k, v in out.items()]
    # ok: the GENERATIONS are the recorded ones, which is what borrowing them needs.
    # kl_ok: the KLs are the rescored ones too, which is what comparing KLs needs.
    out["ok"] = bool(arms) and all(v["generated"] > 0 and v["identical"] == v["generated"] for v in arms)
    out["kl_ok"] = None if klpool is None else all(
        v.get("kl_identical", 0) == v.get("kl_compared", 0) for v in arms)
    return out


def identity_reference(path):
    """Share of items whose every teacher-forced argmax is unchanged, per variant."""
    by = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["set"].startswith("constitution") and r["variant"] in (
                    "fp16+native", "shuffle", "softzero", "base"):
                by.setdefault(r["variant"], []).append(r["same_top1"] == 1.0)
    return {k: {"items_unchanged": round(sum(v) / len(v), 3), "n": len(v)} for k, v in sorted(by.items())}


def self_test():
    def row(arm, q, text, ref, kl=None, borrowed=False):
        r = {"dataset": "redteam", "qid": q, "arm": arm, "text": text, "n_gen": len(text.split()),
             "gen_ids": [len(w) for w in text.split()], "meta": {"refusal": ref}}
        if kl is not None:
            r["kl"] = {"mean": kl, "pool": "prompt"}
        if borrowed:
            r["borrowed_from"] = "rec"
        return r
    base, ref, cand = {}, {}, {}
    for i in range(10):
        q = f"q{i}"
        base[q] = row("base", q, f"sure here {i}", False, borrowed=i >= 3)
        r_ref = i < 4                                     # the reference turns four to refusal
        ref[q] = row("psilm", q, f"{'no' if r_ref else 'sure'} psilm {i}", r_ref, 0.10, borrowed=i >= 3)
        same = i not in (3, 7)                            # the candidate: the same text but on two
        cand[q] = row("fixed", q, ref[q]["text"] if same else f"other {i}", r_ref if same else not r_ref, 0.11)
    out = compare(base, ref, cand, "t:fixed")
    assert out["refusal_rate"] == {"base": 0.0, "reference": 0.4, "arm": 0.4}
    assert out["reference_vs_base"]["to_refusal"] == 4 and out["reference_vs_base"]["net"] == 4
    assert out["vs_base"]["to_refusal_qids"] == ["q0", "q1", "q2", "q7"] and out["vs_base"]["from_refusal"] == 0
    assert out["vs_reference"]["to_refusal_qids"] == ["q7"] and out["vs_reference"]["from_refusal_qids"] == ["q3"]
    assert out["vs_reference"]["discordant"] == 2 and out["vs_reference"]["p"] == 1.0
    assert out["flips"] == {"reference": 4, "arm": 4, "shared": 3, "to_refusal_shared": 3}
    assert out["identity"]["arm_is_reference"] == 8 and out["to_adjudicate"] == ["q3", "q7"]
    assert out["divergence"] == {"reference": {"mean": 0.1, "pool": ["prompt"]},
                                 "arm": {"mean": 0.11, "pool": ["prompt"]}}
    rec = {"base": {q: {**r} for q, r in base.items()}, "psilm": {q: {**r} for q, r in ref.items()}}
    rep = reproduction({"base": base, "psilm": ref}, rec)
    assert rep == {"base": {"generated": 3, "identical": 3, "borrowed": 7},
                   "psilm": {"generated": 3, "identical": 3, "borrowed": 7}, "ok": True, "kl_ok": None}
    rec["psilm"]["q1"] = {**rec["psilm"]["q1"], "text": "drifted"}
    assert reproduction({"base": base, "psilm": ref}, rec)["ok"] is False
    rec["psilm"]["q8"] = {**rec["psilm"]["q8"], "text": "a borrowed row is not evidence"}
    assert reproduction({"base": base, "psilm": ref}, rec)["psilm"]["identical"] == 2
    rec["psilm"]["q1"] = {**ref["q1"]}
    kl = {(f"q{i}", "psilm"): 0.10 for i in range(10)}
    ok = reproduction({"base": base, "psilm": ref}, rec, kl)
    assert ok["ok"] and ok["kl_ok"] and ok["psilm"]["kl_compared"] == 3 and ok["psilm"]["kl_identical"] == 3
    kl[("q2", "psilm")] = 0.1003                                # the same text, another KL
    off = reproduction({"base": base, "psilm": ref}, rec, kl)
    assert off["ok"] is True and off["kl_ok"] is False
    rec["base"]["q0"] = {**rec["base"]["q0"], "stopped_eos": True}   # the same text, another stop
    assert reproduction({"base": base, "psilm": ref}, rec)["base"]["identical"] == 2
    allb = {a: {q: {**r, "borrowed_from": "rec"} for q, r in rows.items()}
            for a, rows in (("base", base), ("psilm", ref))}
    assert reproduction(allb, rec)["ok"] is False               # nothing generated: nothing shown
    del cand["q0"]
    cand["zz"] = row("fixed", "zz", "x", False)
    try:
        compare(base, ref, cand, "t:fixed")
        raise AssertionError("arms over different prompts were compared")
    except SystemExit:
        pass
    assert mcnemar_exact(0, 4) == 0.125
    print("[self-test] eval/const_arm_compare.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="TAG[:ARM] of the base arm (default arm: base)")
    ap.add_argument("--ref", help="TAG:ARM of the reference arm")
    ap.add_argument("--arms", help="comma-separated TAG:ARM candidates")
    ap.add_argument("--repro", default=None, help="RUN_TAG,RECORDED_TAG")
    ap.add_argument("--repro-klpool", default=None,
                    help="the recorded run's rescored KLs (kl_prompt): generated rows must give them")
    ap.add_argument("--identity-reference", default=None, help="a compress.rows.jsonl")
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.base and a.ref and a.arms):
        ap.error("--base, --ref and --arms are required")
    cache = {}

    def arm(spec, default=None):
        tag, _, name = spec.partition(":")
        name = name or default
        if tag not in cache:
            cache[tag] = load(Path(a.bench_dir) / f"{tag}_guardrail.rows.jsonl")
        if name not in cache[tag]:
            raise SystemExit(f"{tag}: no {name} arm ({sorted(cache[tag])})")
        return cache[tag][name]
    base, ref = arm(a.base, "base"), arm(a.ref)
    out = {"base": a.base, "reference": a.ref,
           "arms": [compare(base, ref, arm(s), s) for s in a.arms.split(",")]}
    if a.repro:
        run, rec = a.repro.split(",")
        kl = None
        if a.repro_klpool:
            kl = {(r["qid"], r["arm"]): r["kl_prompt"]["mean"]
                  for r in map(json.loads, filter(str.strip, Path(a.repro_klpool).read_text().splitlines()))
                  if r.get("kl_prompt") and r["kl_prompt"].get("mean") is not None}
        out["reproduction"] = {"run": run, "recorded": rec,
                               **reproduction(load(Path(a.bench_dir) / f"{run}_guardrail.rows.jsonl"),
                                              load(Path(a.bench_dir) / f"{rec}_guardrail.rows.jsonl"), kl)}
    if a.identity_reference:
        out["identity_reference"] = identity_reference(a.identity_reference)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for c in out["arms"]:
        vb, rb, vr = c["vs_base"], c["reference_vs_base"], c["vs_reference"]
        print(f"{c['arm']}: refusal {c['refusal_rate']} | against base {vb['to_refusal']}:{vb['from_refusal']} "
              f"(p {vb['p']}; the reference {rb['to_refusal']}:{rb['from_refusal']}, p {rb['p']}) | against "
              f"the reference {vr['to_refusal']}:{vr['from_refusal']} (p {vr['p']}) | flips shared "
              f"{c['flips']['shared']} of {c['flips']['reference']} and {c['flips']['arm']} | KL "
              f"{c['divergence']} | identical to the reference {c['identity']['arm_is_reference']}/{c['n']} "
              f"| to adjudicate {len(c['to_adjudicate'])}")
    for k in ("reproduction", "identity_reference"):
        if k in out:
            print(f"{k}: {json.dumps(out[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
