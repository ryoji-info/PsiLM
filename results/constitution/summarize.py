"""One table for the constitution-bridge variants of a backbone.

Reads, per variant, the trainer's final chunk eval (results/stage2c_<tag>_<v>/
train_log.jsonl), the two evaluator runs (eval_test.json, eval_helpful.json) and
the guard-rail summary (results/bench/const_<tag>_<v>_guardrail_summary.json when
it exists), and prints a Markdown table plus a JSON copy. Run it again as results
land; missing pieces print as '-'.

  python results/constitution/summarize.py --tag qwen0.5b --variants vn,vn5,all,rand,plainpartner
"""
import argparse
import json
from pathlib import Path


def last_eval(p):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    ev = [r["eval"] for r in rows if "eval" in r]
    return ev[-1] if ev else None


def guard(p):
    """results/bench/<tag>_guardrail.json -> {dataset: {arm: {acc, refusal, sigma, kl}}}."""
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    out = {}
    for ds, d in (s.get("summary") or {}).items():
        row = {}
        for a, r in (d.get("arms") or {}).items():
            if a in ("base", "psilm", "zeroed"):
                row[a] = {"acc": r.get("acc"), "refusal": r.get("refusal_rate"),
                          "sigma": (r.get("sigma") or {}).get("mean"),
                          "kl": (r.get("kl_to_base") or {}).get("mean")}
        out[ds] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen0.5b")
    ap.add_argument("--variants", default="vn,vn5,all,rand,plainpartner")
    ap.add_argument("--out", default=None)
    ap.add_argument("--track-tags", default=None,
                    help="comma-separated guard-rail tags to write tracked summaries for "
                         "and then exit, e.g. dual_qwen35_both -- for runs whose tag is not "
                         "const_<tag>_<variant>")
    a = ap.parse_args()
    if a.track_tags:
        for tg in [x for x in a.track_tags.split(",") if x]:
            if track(Path(f"results/bench/{tg}_guardrail.json")) is None:
                print(f"{tg}: no results/bench/{tg}_guardrail.json")
        return
    table = {}
    for v in a.variants.split(","):
        d = Path(f"results/stage2c_{a.tag}_{v}")
        meta = json.loads((d / "bridges.npz.meta").read_text()) if (d / "bridges.npz.meta").exists() else {}
        rec = {"write_dims": (len(meta["write_dims"]) if meta.get("write_dims") else meta.get("d_model")),
               "l_fwd": meta.get("l_fwd"), "l_rev": meta.get("l_rev"), "step": meta.get("step"),
               "const_model": meta.get("const_model"),
               "val": last_eval(d / "train_log.jsonl")}
        for split in ("test", "helpful"):
            p = d / f"eval_{split}.json"
            rec[split] = json.loads(p.read_text()) if p.exists() else None
        rec["guardrail"] = guard(Path(f"results/bench/const_{a.tag}_{v}_guardrail.json"))
        table[v] = rec
    f = lambda x, k=4: ("-" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x)))  # noqa: E731

    def arm(e, arm, key):
        if not e:
            return None
        arms = e.get("arms", e)
        r = arms.get(arm) if isinstance(arms, dict) else None
        return r.get(key) if isinstance(r, dict) else None

    print(f"| variant | write dims | val CE psilm/base | test CE psilm/base | test agree | test refusal base/psilm | helpful CE | helpful refusal | GSM8K base/psilm | MMLU | BoolQ | redteam refusal | redteam KL |")
    print("|---|---:|---|---|---|---|---|---|---|---|---|---|---|")
    for v, r in table.items():
        val = r["val"] or {}
        te, he, g = r["test"], r["helpful"], r["guardrail"] or {}
        gg = lambda ds, a_, k: ((g.get(ds) or {}).get(a_) or {}).get(k)  # noqa: E731
        print(f"| {v} | {r['write_dims']} | {f((val.get('psilm') or {}).get('ce'))}/{f((val.get('base') or {}).get('ce'))} "
              f"| {f(arm(te,'psilm','ce'))}/{f(arm(te,'base','ce'))} | {f(arm(te,'psilm','agree'))} "
              f"| {f(arm(te,'base','refusal'),2)}/{f(arm(te,'psilm','refusal'),2)} "
              f"| {f(arm(he,'psilm','ce'))}/{f(arm(he,'base','ce'))} | {f(arm(he,'base','refusal'),2)}/{f(arm(he,'psilm','refusal'),2)} "
              f"| {f(gg('gsm8k','base','acc'),2)}/{f(gg('gsm8k','psilm','acc'),2)} | {f(gg('mmlu','base','acc'),2)}/{f(gg('mmlu','psilm','acc'),2)} "
              f"| {f(gg('boolq','base','acc'),2)}/{f(gg('boolq','psilm','acc'),2)} | {f(gg('redteam','base','refusal'),2)}/{f(gg('redteam','psilm','refusal'),2)} "
              f"| {f(gg('redteam','psilm','kl'))} |")
    out = Path(a.out or f"results/constitution/summary_{a.tag}.json")
    out.write_text(json.dumps(table, indent=1))
    print(f"\nwrote {out}")
    write_tracked_summaries(a.tag, a.variants.split(","))




def track(p: Path):
    """One results/bench/<tag>_guardrail.json minus its rows -> the
    *_guardrail_summary.json the repo tracks (raw guard-rail JSONs are ignored)."""
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    s.pop("rows", None)
    q = p.with_name(p.name.replace("_guardrail.json", "_guardrail_summary.json"))
    q.write_text(json.dumps(s, indent=1))
    print(f"tracked summary -> {q}")
    return q


def write_tracked_summaries(tag: str, variants):
    for v in variants:
        track(Path(f"results/bench/const_{tag}_{v}_guardrail.json"))


if __name__ == "__main__":
    main()
