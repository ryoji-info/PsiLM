#!/usr/bin/env python3
"""The AI generation disclosure of four published cards names Claude, not a model.

  python results/hf_export/update_card_wording.py --check     # the hub against git; uploads nothing
  python results/hf_export/update_card_wording.py --upload    # ONE commit a repository, then --verify
  python results/hf_export/update_card_wording.py --verify    # the hub against what was uploaded

The repositories are PUBLIC. In each, README.md is replaced by its source in git, and in the one
whose MANIFEST.md lists the card's size and sha256, the manifest with that row rewritten.

Every safeguard is a refusal:
- the published card must be the source as it was before the wording changed, to the byte, with
  only this difference: every line that differs named a model on the hub and names none here,
  and nothing else in it moved. A card that differs in any other way is left alone;
- the sources must be on origin/main as they are here;
- the upload is made onto the commit the check was made against (`parent_commit`);
- after the upload every file but the ones uploaded must be the blob it was.

Visibility is not touched. Uses the ambient `hf auth login`; no credentials pass through.
"""
import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_psilm2_tokens import REPO, pushed   # noqa: E402

CARDS = {"ryoji-info/Gemma-4-12B-PsiLM": {"README.md": "release/gemma-4-12b-psilm/README.md"},
         "ryoji-info/Qwen3.5-9B-PsiLM": {"README.md": "release/qwen3.5-9b-psilm/README.md",
                                         "MANIFEST.md": "release/qwen3.5-9b-psilm/MANIFEST.md"},
         "ryoji-info/PsiLM-bridges": {"README.md": "results/hf_export/bridges/README.md"},
         "ryoji-info/PsiLM-physics": {"README.md": "results/hf_export/physics/README.md"}}
MODEL = re.compile(r"Fable|Opus|Sonnet|Haiku")
BASE = REPO / "results/hf_export/card_wording_base.json"
TITLE = "Card: the AI generation disclosure names Claude, not a model"


def only_the_wording(old: str, new: str, what: str):
    """The lines of `old` that `new` replaces, each by one line; SystemExit if anything else differs."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    changed = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or i2 - i1 != j2 - j1:
            raise SystemExit(f"{what}: lines were added or removed, the first of them:\n  " + (a[i1:i2] or b[j1:j2])[0][:200])
        for x, y in zip(a[i1:i2], b[j1:j2]):
            if not MODEL.search(x) or MODEL.search(y):
                raise SystemExit(f"{what}: a line changed that is not the wording:\n  {x[:200]}\n  {y[:200]}")
            changed.append((x, y))
    return changed


def only_the_row(old: str, new: str, card: bytes, what: str):
    """The manifest differs in the card's row alone, and the new row is the card's."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    if len(a) != len(b):
        raise SystemExit(f"{what}: lines were added or removed")
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    want = f"{len(card):,} B | `{hashlib.sha256(card).hexdigest()}` |"
    if len(diff) != 1 or not diff[0][0].startswith("| `README.md` |") or not diff[0][1].startswith("| `README.md` |") \
            or not diff[0][1].rstrip().endswith(want):
        raise SystemExit(f"{what}: it differs from the published one in more than the card's row, or the row is not the card's")
    return diff


def state(api, hub, revision=None):
    info = api.repo_info(hub, repo_type="model", files_metadata=True, revision=revision)
    return info, {s.rfilename: s.blob_id for s in info.siblings}


def check(write=True):
    from huggingface_hub import HfApi, hf_hub_download
    api, base = HfApi(), {}
    for hub, files in CARDS.items():
        info, blobs = state(api, hub)
        pub = {p: Path(hf_hub_download(hub, p, repo_type="model", revision=info.sha, force_download=True)).read_bytes()
               for p in files}
        src = {p: (REPO / f).read_bytes() for p, f in files.items()}
        for f in files.values():
            if not pushed(REPO / f):
                raise SystemExit(f"{f} is not on origin/main as it is here: commit and push it first")
        lines = only_the_wording(pub["README.md"].decode("utf-8"), src["README.md"].decode("utf-8"), f"{hub} README.md")
        if not lines:
            raise SystemExit(f"{hub}: the published card is the source already")
        if "MANIFEST.md" in files:
            only_the_row(pub["MANIFEST.md"].decode("utf-8"), src["MANIFEST.md"].decode("utf-8"), src["README.md"],
                         f"{hub} MANIFEST.md")
        base[hub] = {"commit": info.sha, "private": info.private, "blobs": blobs,
                     "sha256": {p: hashlib.sha256(v).hexdigest() for p, v in src.items()}}
        print(f"{hub}@{info.sha[:8]} (private {info.private}, {len(blobs)} files): README.md changes {len(lines)} lines"
              + (", MANIFEST.md its row" if "MANIFEST.md" in files else ""))
        for x, y in lines:
            print("   - " + x.strip()[:150] + "\n   + " + y.strip()[:150])
    if write:
        BASE.write_text(json.dumps(base, indent=1) + "\n")
    return base


def upload():
    from huggingface_hub import CommitOperationAdd, HfApi
    api, base = HfApi(), check()
    for hub, files in CARDS.items():
        b = base[hub]
        got = api.create_commit(
            repo_id=hub, repo_type="model", parent_commit=b["commit"], commit_message=TITLE,
            commit_description="README.md is its source in git with the model names taken out of the disclosure; "
                               "nothing else in it changes" + ("; MANIFEST.md lists the card's new size and sha256"
                                                                if "MANIFEST.md" in files else "") + ".",
            operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(REPO / f)) for p, f in files.items()])
        print(f"committed {got.oid[:8]} on {hub}", flush=True)
    return verify()


def verify():
    from huggingface_hub import HfApi, hf_hub_download
    api, base, bad = HfApi(), json.loads(BASE.read_text()), []
    for hub, files in CARDS.items():
        b = base[hub]
        info, blobs = state(api, hub)
        for p in files:
            got = hashlib.sha256(Path(hf_hub_download(hub, p, repo_type="model", revision=info.sha,
                                                      force_download=True)).read_bytes()).hexdigest()
            if got != b["sha256"][p] or got != hashlib.sha256((REPO / files[p]).read_bytes()).hexdigest():
                bad.append(f"{hub} {p}: not the source")
        other = {p: v for p, v in blobs.items() if p not in files}
        was = {p: v for p, v in b["blobs"].items() if p not in files}
        if other != was:
            bad.append(f"{hub}: a file other than {sorted(files)} changed: "
                       + str(sorted(set(other.items()) ^ set(was.items()))[:4]))
        if info.private != b["private"]:
            bad.append(f"{hub}: its visibility changed")
        print(f"{hub}@{info.sha[:8]}: private {info.private}, {len(blobs)} files, {len(files)} as the sources, "
              f"the other {len(other)} the blobs they were")
    if bad:
        raise SystemExit("\n".join(bad))
    return 0


def self_test():
    old = "A card.\n\nGenerated by Claude Fable 5 (Anthropic).\n\nEnd.\n"
    new = old.replace("Claude Fable 5", "Claude")
    assert len(only_the_wording(old, new, "t")) == 1 and only_the_wording(old, old, "t") == []
    for bad in (new.replace("End.", "Ended."),                      # another line changed
                new + "More.\n",                                     # a line added
                old.replace("A card.\n\n", ""),                      # lines removed
                old.replace("Fable 5", "Opus 5")):                   # a model named still
        try:
            only_the_wording(old, bad, "t")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"accepted: {bad!r}")
    card = new.encode()
    row = lambda c: f"| `README.md` | the card | {len(c):,} B | `{hashlib.sha256(c).hexdigest()}` |\n"   # noqa: E731
    m_old, m_new = "# M\n\n" + row(old.encode()) + "| `x` | y |\n", "# M\n\n" + row(card) + "| `x` | y |\n"
    assert len(only_the_row(m_old, m_new, card, "t")) == 1
    for bad in (m_new.replace("| `x` | y |", "| `x` | z |"), m_old, m_new + "\n"):
        try:
            only_the_row(m_old, bad, card, "t")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"accepted: {bad!r}")
    print("[self-test] results/hf_export/update_card_wording.py: all assertions passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--check", "--upload", "--verify", "--self-test"):
        g.add_argument(f, action="store_true")
    a = ap.parse_args()
    sys.exit(self_test() if a.self_test else upload() if a.upload else verify() if a.verify
             else (check(write=False) and 0))
