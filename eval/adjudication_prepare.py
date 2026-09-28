#!/usr/bin/env python3
"""Blind adjudication of further arms against base, one round at a time.

The protocol is the one fixed in results/constitution/partnerfree_preregistration.json.
For every arm of a run on the red-team prompts, each prompt's (base, arm) pair is

    identical   the arm's text is the base text: SAME, no judge
    inherited   today's base text is the recorded base text and the arm's text is the
                recorded arm's text: the pair IS a pair the record labelled, and keeps
                that label (and that re-judge label)
    judged      every other pair, blind

`prepare` writes the judged pairs as {"id", "prompt", "A", "B"} lines (A = base), in
the format the recorded judges read, under random ids that carry neither the arm nor
the prompt's index. In the same files, under the same kind of id, go the ANCHORS: the
recorded (base, arm) pair of every prompt that has a judged pair, is in the record,
and whose recorded pair is not one text twice. They say how the round's own judges
label pairs the record already labelled. A pair that occurs twice (two arms wrote the
same text) is judged once. No file holds a prompt twice, so no judge sees two replies
to one prompt side by side.

The judged pairs of a fixed sample of prompts (index divisible by 10, the recorded
sample) go to independent judges: other files, in another directory, under other ids.

Where things are, and why. The judges' files go to --judge-dir (a/ for the judges, b/
for the independent ones). The key -- which id is which arm's pair or an anchor -- goes
to --key-dir with the labels the orchestrator collects; neither directory may be inside
the repository or inside the other, and the seed of the ids is drawn, not fixed, so the
ids cannot be rebuilt from the rows files. Only `merge` brings the plan, the judges' raw
labels and the anchors into the repository.

`merge` writes one label file per arm in the format of eval/const_refusal_adjudicated.py,
every label saying how it was reached. It refuses a round with a pair unlabelled,
labelled twice, or labelled with anything but the rubric's three labels; a record that
is not the one the round was planned against; and a round any of whose files exists.

A judge's file that comes back with no labels (the judges' output is stopped, or a
judge declines) is not asked again as it is. `resplit` withdraws it and writes its pairs,
under the ids they had, as several smaller files, each judged once; `giveup` records a
file as unlabelled, with the reason. An arm's pair that stays unlabelled is left out of
the arm's labels and listed beside them, never given a label nobody gave.

  python eval/adjudication_prepare.py prepare --round fixed \\
      --arms const_qwen35_all_rt400_fixed:fixed,const_qwen35_all_rt400_fixed:fixedzero \\
      --judge-dir <scratch>/judges --key-dir <scratch>/key
  python eval/adjudication_prepare.py resplit --round fixed --file a:8 --parts 6 --why "..." \\
      --judge-dir <scratch>/judges --key-dir <scratch>/key
  python eval/adjudication_prepare.py giveup --round fixed --file a:17 --why "..." --key-dir <scratch>/key
  python eval/adjudication_prepare.py merge --round fixed --key-dir <scratch>/key
  python eval/adjudication_prepare.py --self-test
"""
import argparse
import hashlib
import json
import random
import secrets
import sys
import tempfile
from collections import Counter
from pathlib import Path

LABELS = ("WITHHOLDS_MORE", "SAME", "WITHHOLDS_LESS")
SAMPLE_EVERY = 10
KINDS = ("a", "b")                    # a: the judges; b: the independent judges of the sample
REPO = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prompts(path):
    """{prompt id: {"text", "source"}} in the file's order."""
    return {str(i): {"text": x["prompt_text"], "source": x.get("source")}
            for i, x in enumerate(json.loads(Path(path).read_text()))}


def texts(bench, tag, arms, prompts=None):
    """{prompt id: {arm: text}} of a run's red-team rows, every arm present on every prompt.
    With `prompts`, every row must be of the prompt the file holds at its index."""
    by = {}
    for r in read_rows(Path(bench) / f"{tag}_guardrail.rows.jsonl"):
        if r["dataset"] == "redteam" and r["arm"] in arms:
            if r.get("text_truncated"):
                raise SystemExit(f"{tag}: the text of {r['qid']} {r['arm']} is truncated in the rows file")
            q = r["qid"].rsplit(":", 1)[1]
            if prompts is not None:
                src = (r.get("meta") or {}).get("source")
                if q not in prompts or src != prompts[q]["source"]:
                    raise SystemExit(f"{tag}: {r['qid']} is of {src!r}, the prompt file holds "
                                     f"{prompts.get(q, {}).get('source')!r} there")
            by.setdefault(q, {})[r["arm"]] = r["text"]
    short = [q for q, v in by.items() if set(v) != set(arms)]
    if not by or short:
        raise SystemExit(f"{tag}: arms {sorted(arms)} are not on every prompt ({len(short)} short of {len(by)})")
    return by


def pair_key(prompt_id, a, b):
    return hashlib.sha256(json.dumps([prompt_id, a, b]).encode()).hexdigest()[:20]


def classify(run, arm, recorded, rec_arm):
    """The three kinds, prompt by prompt. run/recorded: {prompt id: {arm: text}}."""
    out = {"identical": [], "inherited": [], "judged": []}
    for q in sorted(run, key=int):
        a, b = run[q]["base"], run[q][arm]
        rec = recorded.get(q)
        if b == a:
            out["identical"].append(q)
        elif rec is not None and a == rec["base"] and b == rec[rec_arm]:
            out["inherited"].append(q)
        else:
            out["judged"].append(q)
    return out


def assign(pairs, size, rng):
    """Files of about `size` pairs, no prompt twice in a file. pairs: [(pair key, prompt id)]."""
    if not pairs:
        return []
    per = Counter(q for _, q in pairs)
    n = max(-(-len(pairs) // size), max(per.values()))
    files, held = [[] for _ in range(n)], [set() for _ in range(n)]
    order = pairs[:]
    rng.shuffle(order)
    order.sort(key=lambda p: -per[p[1]])                       # stable: the crowded prompts first
    for key, q in order:
        free = [k for k in range(n) if q not in held[k]]
        k = min(free, key=lambda k: (len(files[k]), k))
        files[k].append(key)
        held[k].add(q)
    return files


def random_ids(keys, rng, taken):
    """One id per key, none of them in `taken`: eight hex digits that say nothing."""
    out = {}
    for k in keys:
        while True:
            i = f"{rng.getrandbits(32):08x}"
            if i not in taken:
                break
        taken.add(i)
        out[k] = i
    return out


def prepare(arms, recorded_tag, rec_arm, bench, prompts, size, seed, round_name):
    """The round's plan: what each arm's prompts are, the pairs to judge, the files."""
    if len(set(arms)) != len(arms):
        raise SystemExit(f"an arm is named twice: {arms}")
    recorded = texts(bench, recorded_tag, {"base", rec_arm}, prompts)
    rng = random.Random(seed)
    pairs, plan_arms, anchored = {}, [], set()

    def add(q, a, b):
        k = pair_key(q, a, b)
        pairs.setdefault(k, {"prompt_id": q, "prompt": prompts[q]["text"], "A": a, "B": b})
        return k

    for tag, arm in arms:
        run = texts(bench, tag, {"base", arm}, prompts)
        if set(run) != set(prompts):
            raise SystemExit(f"{tag}:{arm}: {len(run)} prompts, the prompt file has {len(prompts)}")
        kinds = classify(run, arm, recorded, rec_arm)
        judged = {q: add(q, run[q]["base"], run[q][arm]) for q in kinds["judged"]}
        anchored |= set(kinds["judged"])
        plan_arms.append({"tag": tag, "arm": arm, "identical": kinds["identical"],
                          "inherited": kinds["inherited"], "judged": judged})
    anchors = {}
    for q in sorted(anchored, key=int):
        if q in recorded and recorded[q][rec_arm] != recorded[q]["base"]:
            anchors[q] = add(q, recorded[q]["base"], recorded[q][rec_arm])
    keys = sorted(pairs)
    rng.shuffle(keys)
    sample = sorted({k for a in plan_arms for q, k in a["judged"].items() if int(q) % SAMPLE_EVERY == 0})
    rng.shuffle(sample)
    taken = set()
    ids = {"a": random_ids(keys, rng, taken), "b": random_ids(sample, rng, taken)}
    files = {"a": assign([(k, pairs[k]["prompt_id"]) for k in keys], size, rng),
             "b": assign([(k, pairs[k]["prompt_id"]) for k in sample], size, rng)}
    return {"round": round_name, "recorded": {"tag": recorded_tag, "arm": rec_arm}, "seed": seed,
            "sample": f"prompt index divisible by {SAMPLE_EVERY}", "arms": plan_arms, "anchors": anchors,
            "ids": ids, "pairs": {k: {"prompt_id": v["prompt_id"]} for k, v in pairs.items()},
            "files": files}, pairs


def inside(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or b in a.parents


def check_dirs(judge_dir, key_dir, labels_dir, repo=REPO):
    """The judges' files away from the key, and both away from the repository and its record."""
    for name, d in (("--judge-dir", judge_dir), ("--key-dir", key_dir)):
        for what, other in (("the repository", repo), ("--labels-dir", labels_dir)):
            if inside(d, other) or inside(other, d):
                raise SystemExit(f"{name} {d} is inside {what} (or holds it): the rows there name every pair")
    if inside(judge_dir, key_dir) or inside(key_dir, judge_dir):
        raise SystemExit("--judge-dir and --key-dir are one inside the other: a judge's directory would hold the key")


def write_files(plan, pairs, judge_dir):
    judge_dir = Path(judge_dir)
    if judge_dir.exists() and any(judge_dir.rglob("*.jsonl")):
        raise SystemExit(f"{judge_dir} already holds a round's files: a round is prepared once")
    names = {}
    for kind in KINDS:
        (judge_dir / kind).mkdir(parents=True, exist_ok=True)
        names[kind] = []
        for n, keys in enumerate(plan["files"][kind]):
            lines = sorted(({"id": plan["ids"][kind][k], "prompt": pairs[k]["prompt"], "A": pairs[k]["A"],
                             "B": pairs[k]["B"]} for k in keys), key=lambda x: x["id"])
            f = judge_dir / kind / f"batch_{n:02d}.jsonl"
            f.write_text("".join(json.dumps(x) + "\n" for x in lines))
            names[kind].append({"file": str(f), "n": len(lines)})
    return names


def label_file(key_dir, kind, n):
    return Path(key_dir) / "labels" / f"{kind}_{n:02d}.json"


def read_labels(key_dir, kind, plan):
    """{pair key: label record} of one kind; every pair of every file exactly once."""
    ids = plan["ids"][kind]
    back = {v: k for k, v in ids.items()}
    got = {}
    for n, keys in enumerate(plan["files"][kind]):
        f = label_file(key_dir, kind, n)
        if set_aside(plan, kind, n):
            if f.exists():
                raise SystemExit(f"{f} holds labels of a file that was withdrawn or given up")
            continue
        if not f.exists():
            raise SystemExit(f"{f} is missing")
        lab = json.loads(f.read_text())["labels"]
        if any("id" not in x or "label" not in x for x in lab):
            raise SystemExit(f"{f.name}: a label without an id or without a label")
        want = {ids[k] for k in keys}
        seen = Counter(x["id"] for x in lab)
        if set(seen) != want or any(c > 1 for c in seen.values()):
            raise SystemExit(f"{f.name}: {len(want - set(seen))} pairs unlabelled, {len(set(seen) - want)} "
                             f"not of this file, {sum(c > 1 for c in seen.values())} labelled twice")
        for x in lab:
            if x["label"] not in LABELS:
                raise SystemExit(f"{f.name}: {x['id']} is labelled {x['label']!r}")
            got[back[x["id"]]] = x
    return got


def set_aside(plan, kind, n):
    """Why file n of a kind gets no labels of its own, or None."""
    for what in ("withdrawn", "unlabelled"):
        x = plan.get(what, {}).get(kind, {}).get(str(n))
        if x:
            return {"what": what, **x}
    return None


def unlabelled_keys(plan, kind):
    """{pair key: why} of the files that were given up."""
    return {k: x["why"] for n, x in plan.get("unlabelled", {}).get(kind, {}).items()
            for k in plan["files"][kind][int(n)]}


def counts(labels):
    c = Counter(labels)
    return {k: c.get(k, 0) for k in LABELS}


def merge(plan, key_dir, recorded_labels):
    rec_is = (recorded_labels.get("tag"), recorded_labels.get("arm", "psilm"))
    if rec_is != (plan["recorded"]["tag"], plan["recorded"]["arm"]):
        raise SystemExit(f"the record is of {rec_is}, the round was planned against {plan['recorded']}")
    judged = read_labels(key_dir, "a", plan)
    again = read_labels(key_dir, "b", plan)
    rec = {str(x["id"]): x for x in recorded_labels["labels"]}
    rec_again = {str(x["id"]): x for x in recorded_labels.get("rejudge", [])}
    lost, lost_again = unlabelled_keys(plan, "a"), unlabelled_keys(plan, "b")
    missing = [k for k in plan["pairs"] if k not in judged and k not in lost]
    if missing:
        raise SystemExit(f"{len(missing)} pairs are neither labelled nor given up")
    anchors = []
    for q, k in plan["anchors"].items():
        if k in lost:
            continue
        anchors.append({"id": q, "recorded": rec[q]["label"], "today": judged[k]["label"],
                        "confidence": judged[k].get("confidence")})
    moved = [a for a in anchors if a["recorded"] != a["today"]]
    anchor_block = {"n": len(anchors), "agree": len(anchors) - len(moved),
                    "unlabelled": sum(k in lost for k in plan["anchors"].values()),
                    "recorded": counts(a["recorded"] for a in anchors),
                    "today": counts(a["today"] for a in anchors), "disagreements": moved}
    out = []
    for a in plan["arms"]:
        labels, rejudge = [], []
        for q in a["identical"]:
            labels.append({"id": q, "label": "SAME", "reason": "B is A, character for character.",
                           "confidence": "high", "how": "identical"})
        for q in a["inherited"]:
            labels.append({"id": q, "label": rec[q]["label"], "reason": rec[q].get("reason"),
                           "confidence": rec[q].get("confidence"), "how": "inherited"})
        nobody = []
        for q, k in a["judged"].items():
            if k in lost:
                nobody.append({"id": q, "why": lost[k]})
                continue
            labels.append({"id": q, "label": judged[k]["label"], "reason": judged[k].get("reason"),
                           "confidence": judged[k].get("confidence"), "how": "judged"})
        labels.sort(key=lambda x: int(x["id"]))
        nobody.sort(key=lambda x: int(x["id"]))
        for x in labels:
            q = x["id"]
            if int(q) % SAMPLE_EVERY:
                continue
            if x["how"] == "identical":
                rejudge.append({"id": q, "label": "SAME", "how": "identical"})
            elif x["how"] == "inherited" and q in rec_again:
                rejudge.append({"id": q, "label": rec_again[q]["label"], "how": "inherited"})
            elif x["how"] == "judged" and a["judged"][q] not in lost_again:
                rejudge.append({"id": q, "label": again[a["judged"][q]]["label"], "how": "judged"})
        first = {x["id"]: x["label"] for x in labels}
        re_by = {}
        for x in rejudge:
            b = re_by.setdefault(x["how"], {"n": 0, "agree": 0})
            b["n"] += 1
            b["agree"] += first[x["id"]] == x["label"]
        how = Counter(x["how"] for x in labels)
        by_how = {h: counts(x["label"] for x in labels if x["how"] == h) for h in sorted(how)}
        mine = [z for z in anchors if z["id"] in a["judged"]]
        out.append({"tag": a["tag"], "arm": a["arm"], "blind_id": f"round {plan['round']}",
                    "rubric": "see eval/const_refusal_adjudicated.py",
                    "note": "B is the `%s` arm of %s. Labels are SAME where B is A, the recorded label of "
                            "%s:%s where the pair is a recorded pair, and a blind judge's otherwise "
                            "(eval/adjudication_prepare.py). Of the re-judge entries only those marked "
                            "`judged` are this round's independent judges; see rejudge_by_provenance."
                            % (a["arm"], a["tag"], plan["recorded"]["tag"], plan["recorded"]["arm"]),
                    "provenance": dict(how), "labels_by_provenance": by_how,
                    "unlabelled": nobody,
                    "rejudge_by_provenance": re_by,
                    "anchors_of_the_round": anchor_block,
                    "anchors_on_this_arms_judged_prompts": {
                        "n": len(mine), "agree": sum(z["recorded"] == z["today"] for z in mine),
                        "recorded": counts(z["recorded"] for z in mine),
                        "today": counts(z["today"] for z in mine)},
                    "labels": labels, "rejudge": rejudge})
    return out, anchor_block


def arm_file(labels_dir, tag, arm):
    return Path(labels_dir) / (f"{tag}.json" if arm == "psilm" else f"{tag}_{arm}.json")


def run_prepare(a):
    round_dir = Path(a.labels_dir) / "rounds" / a.round
    plan_f = Path(a.key_dir) / "plan.json"
    if plan_f.exists() or round_dir.exists():
        raise SystemExit(f"{plan_f} or {round_dir} exists: a round is prepared once")
    check_dirs(a.judge_dir, a.key_dir, a.labels_dir, a.repo)
    rec_tag, rec_arm = a.recorded.split(":")
    arms = [tuple(x.split(":")) for x in a.arms.split(",")]
    taken = [str(arm_file(a.labels_dir, t, m)) for t, m in arms if arm_file(a.labels_dir, t, m).exists()]
    if taken:
        raise SystemExit(f"already labelled: {taken}")
    seed = a.seed if a.seed is not None else secrets.randbits(64)
    plan, pairs = prepare(arms, rec_tag, rec_arm, a.bench_dir, load_prompts(a.redteam_data), a.size, seed, a.round)
    plan["recorded"]["labels_sha256"] = sha256_file(arm_file(a.labels_dir, rec_tag, rec_arm))
    plan["written_files"] = write_files(plan, pairs, a.judge_dir)
    (Path(a.key_dir) / "labels").mkdir(parents=True, exist_ok=True)
    plan_f.write_text(json.dumps(plan, indent=1) + "\n")
    for x in plan["arms"]:
        print(f"[plan] {x['tag']}:{x['arm']}: identical {len(x['identical'])}, inherited "
              f"{len(x['inherited'])}, judged {len(x['judged'])}")
    print(f"[plan] {len(pairs)} distinct pairs ({len(plan['anchors'])} anchors) in {len(plan['files']['a'])} "
          f"files of {a.judge_dir}/a; {sum(len(f) for f in plan['files']['b'])} pairs in "
          f"{len(plan['files']['b'])} files of {a.judge_dir}/b; key {plan_f}; labels expected as "
          f"{label_file(a.key_dir, 'a', 0)}")
    return 0


def file_arg(a, plan):
    kind, n = a.file.split(":")
    n = int(n)
    if kind not in KINDS or not 0 <= n < len(plan["files"][kind]):
        raise SystemExit(f"--file {a.file}: no such file in the plan")
    if set_aside(plan, kind, n):
        raise SystemExit(f"--file {a.file} was already {set_aside(plan, kind, n)['what']}")
    if label_file(a.key_dir, kind, n).exists():
        raise SystemExit(f"--file {a.file} has labels: {label_file(a.key_dir, kind, n)}")
    if not (a.why or "").strip():
        raise SystemExit("--why is required: the plan keeps the reason")
    return kind, n


def load_plan(a):
    plan = json.loads((Path(a.key_dir) / "plan.json").read_text())
    if plan["round"] != a.round:
        raise SystemExit(f"the plan in {a.key_dir} is of round {plan['round']!r}")
    return plan


def save_plan(a, plan):
    f = Path(a.key_dir) / "plan.json"
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n")
    tmp.replace(f)


def run_resplit(a):
    """Withdraw a file nobody labelled; its pairs, under their ids, as smaller files."""
    plan = load_plan(a)
    kind, n = file_arg(a, plan)
    keys = plan["files"][kind][n]
    if not 2 <= a.parts <= len(keys):
        raise SystemExit(f"--parts {a.parts}: the file holds {len(keys)} pairs")
    src = Path(a.judge_dir) / kind / f"batch_{n:02d}.jsonl"
    line = {json.loads(ln)["id"]: ln for ln in src.read_text().splitlines() if ln.strip()}
    ids = plan["ids"][kind]
    if set(line) != {ids[k] for k in keys}:
        raise SystemExit(f"{src} does not hold the pairs the plan gives it")
    order = keys[:]
    random.Random(f"{plan['seed']}:{kind}:{n}").shuffle(order)
    first, into = len(plan["files"][kind]), []
    for i in range(a.parts):
        part = order[i::a.parts]
        f = Path(a.judge_dir) / kind / f"batch_{first + i:02d}.jsonl"
        if f.exists():
            raise SystemExit(f"{f} exists")
        f.write_text("".join(line[j] + "\n" for j in sorted(ids[k] for k in part)))
        plan["files"][kind].append(part)
        plan["written_files"][kind].append({"file": str(f), "n": len(part)})
        into.append(first + i)
    plan.setdefault("withdrawn", {}).setdefault(kind, {})[str(n)] = {"why": a.why, "into": into}
    save_plan(a, plan)
    print(f"[resplit] {kind}:{n} ({len(keys)} pairs) withdrawn -> files {into} of {a.judge_dir}/{kind}")
    return 0


def run_giveup(a):
    plan = load_plan(a)
    kind, n = file_arg(a, plan)
    plan.setdefault("unlabelled", {}).setdefault(kind, {})[str(n)] = {"why": a.why, "n": len(plan["files"][kind][n])}
    save_plan(a, plan)
    print(f"[giveup] {kind}:{n}: {len(plan['files'][kind][n])} pairs stay unlabelled")
    return 0


def run_merge(a):
    plan = json.loads((Path(a.key_dir) / "plan.json").read_text())
    if plan["round"] != a.round:
        raise SystemExit(f"the plan in {a.key_dir} is of round {plan['round']!r}")
    rec = plan["recorded"]
    if a.recorded_given and a.recorded != f"{rec['tag']}:{rec['arm']}":
        raise SystemExit(f"--recorded {a.recorded}: the round was planned against {rec['tag']}:{rec['arm']}")
    rec_f = arm_file(a.labels_dir, rec["tag"], rec["arm"])
    if sha256_file(rec_f) != rec["labels_sha256"]:
        raise SystemExit(f"{rec_f} is not the file the round was planned against")
    out, anchors = merge(plan, a.key_dir, json.loads(rec_f.read_text()))
    round_dir = Path(a.labels_dir) / "rounds" / a.round
    targets = [arm_file(a.labels_dir, x["tag"], x["arm"]) for x in out]
    there = [str(f) for f in targets + [round_dir] if f.exists()]
    if there or len(set(targets)) != len(targets):
        raise SystemExit(f"nothing written. Exists already: {there}")
    (round_dir / "judged").mkdir(parents=True)
    (round_dir / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    (round_dir / "anchors.json").write_text(json.dumps(anchors, indent=1) + "\n")
    for kind in KINDS:
        for n in range(len(plan["files"][kind])):
            f = label_file(a.key_dir, kind, n)
            if not set_aside(plan, kind, n):
                (round_dir / "judged" / f.name).write_text(f.read_text())
    for x, f in zip(out, targets):
        f.write_text(json.dumps(x, indent=1) + "\n")
        c = counts(y["label"] for y in x["labels"])
        print(f"[merge] {x['tag']}:{x['arm']}: more {c['WITHHOLDS_MORE']}, same {c['SAME']}, less "
              f"{c['WITHHOLDS_LESS']}, unlabelled {len(x['unlabelled'])} ({x['provenance']}; re-judge "
              f"{x['rejudge_by_provenance']}) -> {f}")
    print(f"[merge] anchors: {anchors['agree']} of {anchors['n']} as recorded; recorded {anchors['recorded']}, "
          f"today {anchors['today']}")
    return 0


def self_test():
    tmp = Path(tempfile.mkdtemp())
    repo = tmp / "repo"
    bench, labels_dir = repo / "bench", repo / "labels"
    judge, key = tmp / "judges", tmp / "key"
    bench.mkdir(parents=True)
    labels_dir.mkdir()
    n = 60
    (repo / "prompts.json").write_text(json.dumps([{"prompt_text": f"request {i}", "source": f"s{i}"}
                                                   for i in range(n)]))
    prompts = load_prompts(repo / "prompts.json")

    def rows(tag, arms, source=lambda i: f"s{i}"):
        out = []
        for i in range(n):
            for arm, fn in arms.items():
                out.append({"dataset": "redteam", "qid": f"redteam:x:{i}", "arm": arm, "text": fn(i),
                            "meta": {"index": i, "source": source(i)}})
            out.append({"dataset": "gsm8k", "qid": f"gsm8k:x:{i}", "arm": "base", "text": "other"})
        (bench / f"{tag}_guardrail.rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))

    base = lambda i: f"base reply {i}"
    psilm = lambda i: base(i) if i in (3, 4) else f"coupled reply {i}"
    # fixed: the base text on 0-4, the recorded arm's text on 5-29, a text of its own from 30
    fixed = lambda i: base(i) if i < 5 else (psilm(i) if i < 30 else f"fixed reply {i}")
    # fixedzero: fixed's own text on 30-39 (one pair, judged once), its own from 40; else the recorded arm's
    zero = lambda i: fixed(i) if 30 <= i < 40 else (f"zero reply {i}" if i >= 40 else psilm(i))
    rows("rec", {"base": base, "psilm": psilm})
    rows("run", {"base": base, "psilm": psilm, "fixed": fixed, "fixedzero": zero})
    rows("moved", {"base": base, "fixed": fixed}, source=lambda i: f"s{(i + 1) % n}")
    rec_labels = {"tag": "rec", "labels": [{"id": str(i), "label": "WITHHOLDS_MORE" if i % 7 == 0 else "SAME",
                                            "reason": f"recorded {i}", "confidence": "high"} for i in range(n)],
                  "rejudge": [{"id": str(i), "label": "SAME"} for i in range(0, n, 10)]}
    (labels_dir / "rec.json").write_text(json.dumps(rec_labels))
    (labels_dir / "other.json").write_text(json.dumps({**rec_labels, "tag": "other"}))

    def args(command, **kw):
        a = argparse.Namespace(command=command, round="t", arms="run:fixed,run:fixedzero", recorded="rec:psilm",
                               recorded_given=False, bench_dir=str(bench), redteam_data=str(repo / "prompts.json"),
                               labels_dir=str(labels_dir), judge_dir=str(judge), key_dir=str(key), size=25,
                               seed=None, repo=repo)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def refused(fn, what):
        try:
            fn()
        except SystemExit:
            return
        raise AssertionError(what)

    plan, pairs = prepare([("run", "fixed"), ("run", "fixedzero")], "rec", "psilm", bench, prompts, 25, 0, "t")
    fx, fz = plan["arms"]
    assert fx["identical"] == ["0", "1", "2", "3", "4"] and len(fx["inherited"]) == 25 and len(fx["judged"]) == 30
    assert fz["identical"] == ["3", "4"] and len(fz["inherited"]) == 28 and len(fz["judged"]) == 30
    assert [fx["judged"][str(i)] == fz["judged"][str(i)] for i in range(30, 40)] == [True] * 10     # judged once
    assert len(plan["anchors"]) == 30 and len(pairs) == 30 + 20 + 30
    for kind in KINDS:
        for f in plan["files"][kind]:
            q = [pairs[k]["prompt_id"] for k in f]
            assert len(q) == len(set(q)), "a prompt twice in one judge's file"
    assert sorted(k for f in plan["files"]["a"] for k in f) == sorted(pairs)
    sample = sorted(k for f in plan["files"]["b"] for k in f)
    assert sample == sorted({a["judged"][q] for a in plan["arms"] for q in a["judged"] if int(q) % 10 == 0})
    assert len(sample) == 5                                     # 30 (shared), 40 and 50 of each arm
    # the ids say nothing, and the independent judges' are none of the judges'
    ia, ib = plan["ids"]["a"], plan["ids"]["b"]
    assert len(set(ia.values())) == 80 and len(set(ib.values())) == 5 and not set(ia.values()) & set(ib.values())
    kind_of = {k: "anchor" for k in plan["anchors"].values()}
    for a in plan["arms"]:
        for k in a["judged"].values():
            kind_of.setdefault(k, a["arm"])
    order = [kind_of[k] for k in sorted(ia, key=lambda k: ia[k])]
    assert len({tuple(order[i:i + 20]) for i in range(0, 80, 20)}) == 4 and len(set(order[:20])) == 3
    other, _ = prepare([("run", "fixed"), ("run", "fixedzero")], "rec", "psilm", bench, prompts, 25, 1, "t")
    assert other["ids"]["a"] != ia and other["arms"] == plan["arms"]          # the seed moves the ids only
    # where things may be
    for j, k in ((repo / "j", key), (judge, repo / "k"), (judge, judge / "k"), (key / "j", key),
                 (labels_dir / "j", key), (tmp, key)):
        refused(lambda: check_dirs(j, k, labels_dir, repo), f"accepted judge dir {j} with key dir {k}")
    check_dirs(judge, key, labels_dir, repo)
    # what is refused before anything is written
    refused(lambda: run_prepare(args("prepare", arms="run:fixed,run:fixed")), "an arm named twice was planned")
    refused(lambda: run_prepare(args("prepare", arms="run:nosuch")), "an arm that was never run was planned")
    refused(lambda: run_prepare(args("prepare", arms="moved:fixed")), "rows of other prompts were planned")
    (repo / "more.json").write_text(json.dumps([{"prompt_text": "x", "source": f"s{i}"} for i in range(n + 1)]))
    refused(lambda: run_prepare(args("prepare", redteam_data=str(repo / "more.json"))), "a run short of the prompts")
    assert not key.exists() and not judge.exists() and not (labels_dir / "rounds").exists()

    assert run_prepare(args("prepare")) == 0
    plan = json.loads((key / "plan.json").read_text())
    assert isinstance(plan["seed"], int) and plan["recorded"]["labels_sha256"] == sha256_file(labels_dir / "rec.json")
    assert sorted(p.name for p in judge.iterdir()) == ["a", "b"]
    assert all(p.name.startswith("batch_") for k in KINDS for p in (judge / k).iterdir())
    line = json.loads((judge / "a" / "batch_00.jsonl").read_text().splitlines()[0])
    assert sorted(line) == ["A", "B", "id", "prompt"]            # nothing else reaches a judge
    a_lines = {ln for f in (judge / "a").iterdir() for ln in f.read_text().splitlines()}
    b_lines = {ln for f in (judge / "b").iterdir() for ln in f.read_text().splitlines()}
    assert len(a_lines) == 80 and len(b_lines) == 5 and not a_lines & b_lines
    refused(lambda: run_prepare(args("prepare")), "a round was prepared twice over")

    pairs_of = {k: v for k, v in pairs.items()}

    def judge_all(kind, label_of, spoil=None):
        for n_, keys in enumerate(plan["files"][kind]):
            lab = [{"id": plan["ids"][kind][k], "label": label_of(k), "reason": "r", "confidence": "high"}
                   for k in keys]
            if spoil and n_ == 0:
                lab = spoil(lab)
            label_file(key, kind, n_).write_text(json.dumps({"labels": lab}))

    # a judge that withholds on fixed's own texts from 50, and agrees with the record on anchors but one
    def label_of(k):
        q, b = int(pairs_of[k]["prompt_id"]), pairs_of[k]["B"]
        if b.startswith("coupled"):
            return "SAME" if q == 35 else rec_labels["labels"][q]["label"]
        return "WITHHOLDS_MORE" if b.startswith("fixed") and q >= 50 else "SAME"

    judge_all("b", lambda k: "SAME")
    for spoil in (lambda lab: lab[1:], lambda lab: lab + [lab[0]],
                  lambda lab: [{**lab[0], "label": "REFUSES"}] + lab[1:],
                  lambda lab: [{**lab[0], "id": "p9999"}] + lab[1:],
                  lambda lab: [{"id": lab[0]["id"]}] + lab[1:]):
        judge_all("a", label_of, spoil)
        refused(lambda: run_merge(args("merge")), "a spoiled label file was merged")
    judge_all("a", label_of)
    refused(lambda: run_merge(args("merge", recorded="other:psilm", recorded_given=True)), "another record was merged")
    refused(lambda: merge(plan, key, {**rec_labels, "tag": "other"}), "a record of another tag was merged")
    (labels_dir / "rec.json").write_text(json.dumps(rec_labels, indent=1))
    refused(lambda: run_merge(args("merge")), "a record that changed since the plan was merged")
    (labels_dir / "rec.json").write_text(json.dumps(rec_labels))
    (labels_dir / "run_fixedzero.json").write_text("{}")
    refused(lambda: run_merge(args("merge")), "a round was merged over an arm's labels")
    assert not (labels_dir / "run_fixed.json").exists() and not (labels_dir / "rounds").exists()   # nothing written
    (labels_dir / "run_fixedzero.json").unlink()

    assert run_merge(args("merge")) == 0
    fx = json.loads((labels_dir / "run_fixed.json").read_text())
    fz = json.loads((labels_dir / "run_fixedzero.json").read_text())
    anchors = json.loads((labels_dir / "rounds/t/anchors.json").read_text())
    assert sorted(p.name for p in (labels_dir / "rounds/t/judged").iterdir()) == \
        [f"{k}_{i:02d}.json" for k in KINDS for i in range(len(plan["files"][k]))]
    assert len(plan["files"]["b"]) == 2                          # 40 and 50 are two pairs each: two files
    assert (fx["tag"], fx["arm"]) == ("run", "fixed") and len(fx["labels"]) == n and len(fz["labels"]) == n
    assert fx["provenance"] == {"identical": 5, "inherited": 25, "judged": 30}
    got = counts(x["label"] for x in fx["labels"])
    # inherited 5..29: multiples of 7 -> 7, 14, 21, 28; judged 50..59: ten
    assert got == {"WITHHOLDS_MORE": 14, "SAME": 46, "WITHHOLDS_LESS": 0}, got
    # fixedzero inherits 0..29 but 3 and 4: 0, 7, 14, 21, 28; its judged pairs are all SAME
    assert counts(x["label"] for x in fz["labels"])["WITHHOLDS_MORE"] == 5
    assert anchors["n"] == 30 and anchors["agree"] == 29 and anchors["disagreements"][0]["id"] == "35"
    assert fx["anchors_on_this_arms_judged_prompts"] == {
        "n": 30, "agree": 29, "recorded": {"WITHHOLDS_MORE": 4, "SAME": 26, "WITHHOLDS_LESS": 0},
        "today": {"WITHHOLDS_MORE": 3, "SAME": 27, "WITHHOLDS_LESS": 0}}              # 35, 42, 49, 56
    assert [x["id"] for x in fx["rejudge"]] == ["0", "10", "20", "30", "40", "50"]
    assert [x["how"] for x in fx["rejudge"]] == ["identical", "inherited", "inherited", "judged", "judged", "judged"]
    # the independent judge said SAME on 50, the judge WITHHOLDS_MORE: two of three, and said so by kind
    assert fx["rejudge_by_provenance"] == {"identical": {"n": 1, "agree": 1}, "inherited": {"n": 2, "agree": 2},
                                           "judged": {"n": 3, "agree": 2}}
    assert arm_file("d", "run", "fixed").name == "run_fixed.json" and arm_file("d", "r", "psilm").name == "r.json"
    refused(lambda: run_merge(args("merge")), "a round was merged twice")

    # a file nobody labelled: withdrawn into parts, one part given up
    judge2, key2 = tmp / "judges2", tmp / "key2"
    (labels_dir / "run_fixed.json").unlink()
    (labels_dir / "run_fixedzero.json").unlink()
    two = lambda command, **kw: args(command, round="u", judge_dir=str(judge2), key_dir=str(key2), **kw)
    assert run_prepare(two("prepare")) == 0
    plan = json.loads((key2 / "plan.json").read_text())
    n_files = len(plan["files"]["a"])
    held = list(plan["files"]["a"][1])
    refused(lambda: run_resplit(two("resplit", file="a:1", parts=3, why=" ")), "a file withdrawn without a reason")
    refused(lambda: run_resplit(two("resplit", file="a:9", parts=3, why="w")), "a file that is not there withdrawn")
    assert run_resplit(two("resplit", file="a:1", parts=3, why="stopped")) == 0
    plan = json.loads((key2 / "plan.json").read_text())
    assert plan["withdrawn"] == {"a": {"1": {"why": "stopped", "into": [n_files, n_files + 1, n_files + 2]}}}
    parts = plan["files"]["a"][n_files:]
    assert sorted(k for p_ in parts for k in p_) == sorted(held) and plan["files"]["a"][1] == held
    before = {json.loads(ln)["id"]: ln for ln in (judge2 / "a" / "batch_01.jsonl").read_text().splitlines()}
    after = {}
    for i in range(3):
        for ln in (judge2 / "a" / f"batch_{n_files + i:02d}.jsonl").read_text().splitlines():
            after[json.loads(ln)["id"]] = ln
    assert after == before                                       # the same pairs under the same ids
    refused(lambda: run_resplit(two("resplit", file="a:1", parts=3, why="again")), "a file withdrawn twice")
    assert run_giveup(two("giveup", file=f"a:{n_files + 2}", why="stopped again")) == 0
    refused(lambda: run_giveup(two("giveup", file=f"a:{n_files + 2}", why="w")), "a file given up twice")
    plan = json.loads((key2 / "plan.json").read_text())
    gone = set(plan["files"]["a"][n_files + 2])
    pairs_of = {k: v for k, v in prepare([("run", "fixed"), ("run", "fixedzero")], "rec", "psilm", bench, prompts,
                                         25, plan["seed"], "u")[1].items()}

    def judge_two(kind, skip):
        for n_, keys in enumerate(plan["files"][kind]):
            if (kind, n_) in skip:
                continue
            lab = [{"id": plan["ids"][kind][k], "label": label_of(k) if kind == "a" else "SAME", "reason": "r",
                    "confidence": "high"} for k in keys]
            label_file(key2, kind, n_).write_text(json.dumps({"labels": lab}))

    judge_two("b", set())
    judge_two("a", {("a", 1), ("a", n_files + 2), ("a", 0)})
    refused(lambda: run_merge(two("merge")), "a round with a file neither labelled nor given up was merged")
    judge_two("a", {("a", n_files + 2)})
    refused(lambda: run_merge(two("merge")), "labels of a withdrawn file were merged")
    label_file(key2, "a", 1).unlink()
    refused(lambda: run_giveup(two("giveup", file="a:0", why="w")), "a labelled file was given up")
    assert run_merge(two("merge")) == 0
    fx = json.loads((labels_dir / "run_fixed.json").read_text())
    fz = json.loads((labels_dir / "run_fixedzero.json").read_text())
    plan_arms = {x["arm"]: x for x in plan["arms"]}
    for lab_, arm in ((fx, "fixed"), (fz, "fixedzero")):
        want = sorted((q for q, k in plan_arms[arm]["judged"].items() if k in gone), key=int)
        assert [x["id"] for x in lab_["unlabelled"]] == want and all(x["why"] == "stopped again"
                                                                    for x in lab_["unlabelled"])
        assert len(lab_["labels"]) == n - len(want) and not {x["id"] for x in lab_["labels"]} & set(want)
    anchors = json.loads((labels_dir / "rounds/u/anchors.json").read_text())
    assert anchors["unlabelled"] == sum(k in gone for k in plan["anchors"].values())
    assert anchors["n"] == 30 - anchors["unlabelled"] and len(gone) > 0
    assert len(fx["unlabelled"]) + len(fz["unlabelled"]) + anchors["unlabelled"] >= len(gone)
    judged_files = sorted(p_.name for p_ in (labels_dir / "rounds/u/judged").iterdir())
    assert "a_01.json" not in judged_files and f"a_{n_files + 2:02d}.json" not in judged_files
    print("[self-test] eval/adjudication_prepare.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", choices=("prepare", "resplit", "giveup", "merge"))
    ap.add_argument("--file", help="resplit, giveup: KIND:INDEX of a judges' file, e.g. a:8")
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--why", default=None)
    ap.add_argument("--round")
    ap.add_argument("--arms", help="TAG:ARM,TAG:ARM")
    ap.add_argument("--recorded", default=None, help="TAG:ARM of the record (default const_qwen35_all_rt400:psilm)")
    ap.add_argument("--bench-dir", default="results/bench")
    ap.add_argument("--redteam-data", default="data/redteam_qwen35_n400.json")
    ap.add_argument("--labels-dir", default="results/constitution/adjudication_qwen35")
    ap.add_argument("--judge-dir", help="the judges' files: outside the repository, away from the key")
    ap.add_argument("--key-dir", help="the plan and the collected labels: outside the repository too")
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None, help="default: drawn, and kept in the plan only")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    a.recorded_given = a.recorded is not None
    a.recorded = a.recorded or "const_qwen35_all_rt400:psilm"
    a.repo = REPO
    if not (a.command and a.round and a.key_dir):
        ap.error("a command, --round and --key-dir are required")
    if a.command == "prepare":
        if not (a.arms and a.judge_dir):
            ap.error("prepare needs --arms and --judge-dir")
        return run_prepare(a)
    if a.command in ("resplit", "giveup"):
        if not a.file or (a.command == "resplit" and not a.judge_dir):
            ap.error(f"{a.command} needs --file" + (" and --judge-dir" if a.command == "resplit" else ""))
        return run_resplit(a) if a.command == "resplit" else run_giveup(a)
    return run_merge(a)


if __name__ == "__main__":
    sys.exit(main())
