#!/usr/bin/env python3
"""Collect a round's judge labels from the orchestrator's journal, and audit the judges.

The judges of eval/adjudication_prepare.py are language-model agents run by a
workflow of the coding harness this work was done in. That workflow leaves a
directory with `journal.jsonl` (one line per agent, {"type": "result", "agentId",
"result": {"labels": [...]}}) and one transcript per agent (`agent-<id>.jsonl`). This
reads those, and nothing else about the harness:

  labels   each agent's labels go to <key-dir>/labels/<kind>_<NN>.json, the file the
           plan expects, found by the ids alone (an id belongs to one file). A return
           with no labels, or not exactly the file's pairs, is reported and stored
           nowhere. A file that has labels already is never written again.
  audit    every tool call of every judge is read. A judge passes if it opened its
           own file and nothing else and listed no directory; whether its output was
           stopped by a safety classifier is reported beside it.

  python eval/adjudication_collect.py --key-dir <scratch>/key --workflow-dir <dir> [--kind a]
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

LABELS = ("WITHHOLDS_MORE", "SAME", "WITHHOLDS_LESS")


def lines(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def audit(transcript):
    """(own file, tool counts, calls that touch anything else, stopped by a classifier)"""
    rows = lines(transcript)
    first = rows[0].get("message", {}).get("content")
    text = first if isinstance(first, str) else " ".join(b.get("text", "") for b in first if isinstance(b, dict))
    own = re.search(r"FILE: (\S+)", text).group(1)
    names, other, stopped = Counter(), [], False
    for r in rows:
        c = r.get("message", {}).get("content")
        if isinstance(c, str):
            stopped |= "stopped by a safety classifier" in c
            continue
        for b in c or []:
            if b.get("type") == "text":
                stopped |= "safety classifier" in b.get("text", "")
            if b.get("type") != "tool_use":
                continue
            names[b["name"]] += 1
            if b["name"] == "StructuredOutput":
                continue
            s = json.dumps(b.get("input"))
            files = set(re.findall(r"batch_\d+\.jsonl", s)) | set(re.findall(r"[\w/.-]+\.json\b", s))
            if files != {Path(own).name} or re.search(r"\bls\b|glob|listdir|find ", s):
                other.append(s[:200])
    return own, dict(names), other, stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-dir", required=True)
    ap.add_argument("--workflow-dir", required=True)
    ap.add_argument("--workflow", default=None, help="a name for the record; default: the directory's")
    a = ap.parse_args()
    key, wf = Path(a.key_dir), Path(a.workflow_dir)
    plan = json.loads((key / "plan.json").read_text())
    file_of = {plan["ids"][k][p]: (k, n) for k in plan["ids"] for n, f in enumerate(plan["files"][k]) for p in f}
    # an id of a withdrawn file is also in one of its parts: the part is the file to fill
    for k in plan["ids"]:
        for n, f in enumerate(plan["files"][k]):
            if str(n) not in plan.get("withdrawn", {}).get(k, {}):
                for p in f:
                    file_of[plan["ids"][k][p]] = (k, n)
    (key / "labels").mkdir(exist_ok=True)
    wrote, empty, bad = [], [], []
    for r in lines(wf / "journal.jsonl"):
        if r.get("type") != "result":
            continue
        labs = (r.get("result") or {}).get("labels") or []
        if not labs:
            empty.append(r["agentId"])
            continue
        files = {file_of.get(x["id"]) for x in labs}
        if len(files) != 1 or None in files:
            bad.append((r["agentId"], "ids of no one file"))
            continue
        kind, n = files.pop()
        want = {plan["ids"][kind][p] for p in plan["files"][kind][n]}
        got = Counter(x["id"] for x in labs)
        if set(got) != want or max(got.values()) > 1 or any(x["label"] not in LABELS for x in labs):
            bad.append((r["agentId"], f"{kind}:{n} not exactly the file's pairs with the three labels"))
            continue
        f = key / "labels" / f"{kind}_{n:02d}.json"
        if f.exists():
            bad.append((r["agentId"], f"{f.name} has labels already"))
            continue
        f.write_text(json.dumps({"judge_agent": r["agentId"], "workflow": a.workflow or wf.name, "labels": labs},
                                indent=1) + "\n")
        wrote.append((f.name, dict(Counter(x["label"] for x in labs))))
    for name, c in sorted(wrote):
        print(f"[labels] {name}: {c}")
    print(f"[labels] {len(wrote)} files written; returns with no labels: {empty or 'none'}; refused: {bad or 'none'}")
    clean = True
    for t in sorted(wf.glob("agent-*.jsonl")):
        own, names, other, stopped = audit(t)
        clean &= not other
        if other or stopped:
            print(f"[audit] {'/'.join(Path(own).parts[-2:])} {names}"
                  f"{' STOPPED BY A SAFETY CLASSIFIER' if stopped else ''}{' OTHER: ' + str(other) if other else ''}")
    print(f"[audit] {len(list(wf.glob('agent-*.jsonl')))} judges; every one read its own file and nothing else: {clean}")
    missing = [f"{k}:{n}" for k in plan["files"] for n in range(len(plan["files"][k]))
               if not (key / "labels" / f"{k}_{n:02d}.json").exists()
               and str(n) not in plan.get("withdrawn", {}).get(k, {})
               and str(n) not in plan.get("unlabelled", {}).get(k, {})]
    print(f"[labels] files of the plan still without labels: {missing or 'none'}")
    return 0 if clean and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
