#!/usr/bin/env python3
"""Publish what the cards of ryoji-info/PsiLM-2 have gained, and nothing else.

  python results/hf_export/upload_psilm2_card.py --stage --date YYYY-MM-DD \\
      --title "<the commit's title>" --what "<one sentence: what the cards gained>"
  python results/hf_export/upload_psilm2_card.py --upload     # ONE commit on the hub, then --verify
  python results/hf_export/upload_psilm2_card.py --verify     # the hub against what was staged

The repository is PUBLIC. One commit rewrites the cards that gained text, and the manifest:

  README.md                       the release's card:  results/hf_export/psilm2_card.md
  constitution_model/README.md    the partner's card:  results/hf_export/psilm2_partner_card.md
  MANIFEST.md                     the published one with those cards' rows written and a dated
                                  line under its head

Both sources are in git, and the manifest's line names them. A card whose source is the
published card is left out of the commit.

Every safeguard is a refusal:
- a card may only GAIN lines: a line of the published card that is missing from the new one, or
  changed, stops the stage (a correction of published text is another tool's work, and a
  person's decision). The gained lines must stand as blocks of their own: lines put INSIDE a
  published paragraph reword it while keeping its lines, and a comment mark or a code fence
  can hide published lines without removing them; both are refused. This is checked at the
  stage and again, against the hub's own card, before the upload;
- the staged manifest is, to the byte, the published one with the staged cards' rows written
  and this stage's note;
- the stage is built on ONE commit of the hub and the upload is made onto that commit only
  (`parent_commit`): if the hub moved, stage again;
- the cards' sources and the uploader, which the manifest names, must be on origin/main as they
  are here;
- after the upload every file but the ones staged must be the blob it was.

Visibility is not touched here. Uses the ambient `hf auth login`; no credentials pass through.
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_psilm2_tokens import HUB, REPO, hub_state, manifest_rows, pushed, sha256   # noqa: E402

CARDS = {"README.md": REPO / "results/hf_export/psilm2_card.md",
         "constitution_model/README.md": REPO / "results/hf_export/psilm2_partner_card.md"}
STAGE = REPO / "results/hf_export/psilm2_card_stage"
BASE = REPO / "results/hf_export/psilm2_card_base.json"
TABLE = "\n| path | bytes | sha256 |\n"


def text(path) -> str:
    """A file's text exactly as its bytes are: UTF-8 and no newline translation (read_text()
    turns \\r\\n and \\r into \\n, and would let a change of line ends pass as no change)."""
    return Path(path).read_bytes().decode("utf-8")


def changed(a, b):
    """The opcodes of a line diff that are not `equal`."""
    return [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]


BLOCK = re.compile(r"^(>?\s*$|#|\s*[-*+] |\s*\d+\. |\|)")      # a line a new block may be put before
SETEXT = re.compile(r"^ {0,3}(=+|-+)\s*$")          # under a paragraph's last line it makes the paragraph a heading
HIDES = re.compile(r"<!--|-->|```|~~~|<details|</details|^\s*\[[^\]]+\]:\s|^\s*</?[A-Za-z]")


def only_gains(old: str, new: str):
    """The lines the new text adds; SystemExit if it drops or changes a line of the old one,
    puts lines inside a published paragraph, or adds a line that can hide published ones.
    Lines are compared whole and with their ends: a word added inside a published line, a
    trailing blank, a lost final newline are changes."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    ops = changed(a, b)
    lost = [ln for tag, i1, i2, _, _ in ops if tag != "insert" for ln in a[i1:i2]]
    if lost:
        raise SystemExit("the new card drops or changes published lines, the first of them:\n  " + lost[0][:200])
    blank = lambda ln: ln.strip() in ("", ">")                                       # noqa: E731
    front = (a.index("---\n", 1) + 1) if a[:1] == ["---\n"] and "---\n" in a[1:] else 0   # the card's metadata
    for _, i1, _, j1, j2 in ops:
        if i1 < front:
            raise SystemExit("a line is added to the card's front matter:\n  " + b[j1][:200])
        before, after = (a[i1 - 1] if i1 else "\n"), (a[i1] if i1 < len(a) else "\n")
        # above: a block of its own, or a sentence added where a published block ends: what follows
        # is a blank line or a block of its own, and the line before ends a sentence
        closes = blank(after) or bool(BLOCK.match(after))
        top = blank(before) or (closes and (blank(b[j1]) or before.rstrip().endswith((".", ":", "|", ".*", ")"))))
        joined = b[j1:next((k for k in range(j1, j2) if blank(b[k])), j2)]   # the gained lines the published block runs into
        if not blank(before) and any(SETEXT.match(ln) for ln in joined):
            raise SystemExit("a gained line would make published text a heading:\n  " + b[j1][:200])
        # below: what follows is a block of its own
        bottom = blank(b[j2 - 1]) or bool(BLOCK.match(after))
        if not (top and bottom):
            raise SystemExit("lines are put inside a published paragraph, the first of them:\n  " + b[j1][:200])
        hid = [ln for ln in b[j1:j2] if HIDES.search(ln)]
        if hid:
            raise SystemExit("a gained line could hide published text:\n  " + hid[0][:200])
    return [ln for tag, _, _, j1, j2 in ops for ln in b[j1:j2]]


def row(path, size, sha) -> str:
    return f"| `{path}` | {size:,} | `{sha}` |"


def new_manifest(old: str, cards: dict, what: str, date: str) -> str:
    """The published manifest with the rows of `cards` ({path: file}) written and one line
    under its head."""
    head, sep, table = old.partition(TABLE)
    rows = manifest_rows(old)
    if not sep or not cards or any(p not in rows for p in cards):
        raise SystemExit("the published MANIFEST.md is not the one this was written for")
    for p, f in cards.items():
        was = row(p, *rows[p])
        if table.count(was + "\n") != 1:
            raise SystemExit(f"{p}: its row is not in the published manifest as the manifest's own rows say")
        table = table.replace(was + "\n", row(p, Path(f).stat().st_size, sha256(f)) + "\n")
    names = " and ".join(f"`{p}`" for p in cards)
    srcs = " and ".join(f"`{CARDS[p].relative_to(REPO).as_posix()}`" for p in cards)
    its = "Its row was" if len(cards) == 1 else "Their rows were"
    note = (f"\n{names} gained text on {date} and lost none (`results/hf_export/upload_psilm2_card.py`, from\n"
            f"{srcs} of the ΨLM checkout):\n{what}\n"
            f"{its} rewritten; no other row and no other file changed.\n")
    if note in old:
        raise SystemExit("this change is already in the published manifest")
    return head.rstrip("\n") + "\n" + note + TABLE + table


def stage(what: str, title: str, date: str):
    from huggingface_hub import HfApi, hf_hub_download
    if STAGE.exists():
        raise SystemExit(f"{STAGE} exists: remove it to stage again")
    for name, v in (("--what", what), ("--title", title), ("--date", date)):
        if not v or "\n" in v or "\r" in v:
            raise SystemExit(f"{name}: one line")
    if not what.rstrip().endswith("."):
        raise SystemExit("--what: a sentence, ending with a full stop")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise SystemExit("--date: YYYY-MM-DD")
    api = HfApi()
    info, files = hub_state(api)
    pub = {p: Path(hf_hub_download(HUB, p, repo_type="model", revision=info.sha, force_download=True))
           for p in ["MANIFEST.md", *CARDS]}
    man = text(pub["MANIFEST.md"])
    rows = manifest_rows(man)
    gained, checked, lines = {}, {}, {}
    for p, src in CARDS.items():
        if rows[p] != (pub[p].stat().st_size, sha256(pub[p])):
            raise SystemExit(f"{p}: the published card is not the one the published manifest lists")
        new = src.read_bytes()
        got = only_gains(text(pub[p]), new.decode("utf-8"))
        if got:
            gained[p], checked[p], lines[p] = len(got), new, got
    if not gained:
        raise SystemExit("the cards in git are the published ones: nothing to publish")
    for p in gained:
        (STAGE / p).parent.mkdir(parents=True, exist_ok=True)
        (STAGE / p).write_bytes(checked[p])                       # the bytes that were checked
    (STAGE / "MANIFEST.md").write_bytes(
        new_manifest(man, {p: STAGE / p for p in gained}, what, date).encode("utf-8"))
    BASE.write_text(json.dumps({"hub": HUB, "commit": info.sha, "what": what, "title": title, "date": date,
                                "cards": gained, "files": files, "published_manifest": man,
                                "published_cards": {p: text(pub[p]) for p in gained}}, indent=1,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    verify_stage()
    for p, got in lines.items():                                  # what goes is read before it goes
        print(f"\n==== {p} gains {len(got)} lines and loses none ====\n" + "".join(got), end="")
    print(f"\nstaged on {HUB}@{info.sha[:8]}: " + "; ".join(f"{p} +{n} lines" for p, n in gained.items()))


def verify_stage():
    """Everything that can be checked without the hub: what is staged is the cards in git, and the
    published manifest with those cards' rows replaced and one note added."""
    if not BASE.exists() or not STAGE.exists():
        raise SystemExit("nothing is staged (--stage)")
    b = json.loads(BASE.read_text(encoding="utf-8"))
    want = sorted(["MANIFEST.md", *b["cards"]])
    have = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file() and p.name != ".DS_Store")
    if have != want:
        raise SystemExit(f"staged files are not the ones the stage was built with: {sorted(set(have) ^ set(want))}")
    for p in b["cards"]:
        if (STAGE / p).read_bytes() != CARDS[p].read_bytes():
            raise SystemExit(f"the staged {p} is not {CARDS[p].relative_to(REPO)}: remove "
                             f"{STAGE.relative_to(REPO)} and run --stage again")
        if len(only_gains(b["published_cards"][p], text(STAGE / p))) != b["cards"][p]:
            raise SystemExit(f"{p}: not the lines that were staged")
    me = Path.home().name
    for f, body in [(f, text(STAGE / f)) for f in have] + [("--title", b["title"]), ("--what", b["what"])]:
        for what in (str(Path.home()), f"/{me}/", f"Users/{me}", "~/", "/private/", "/tmp/", me):
            if what in body:
                raise SystemExit(f"{f}: something of this machine is in it ({what!r})")
    staged = text(STAGE / "MANIFEST.md")
    old, new = manifest_rows(b["published_manifest"]), manifest_rows(staged)
    if list(old) != list(new) or any(old[p] != new[p] for p in old if p not in b["cards"]):
        raise SystemExit("the staged manifest changes a row that is not a staged card's")
    for p in b["cards"]:
        if new[p] != ((STAGE / p).stat().st_size, sha256(STAGE / p)):
            raise SystemExit(f"{p}: the staged manifest's row is not the staged card's")
    if staged != new_manifest(b["published_manifest"], {p: STAGE / p for p in b["cards"]}, b["what"], b["date"]):
        raise SystemExit("the staged MANIFEST.md is not the published one with the cards' rows and this stage's note")
    return b


def upload():
    from huggingface_hub import CommitOperationAdd, HfApi
    b = verify_stage()
    named = [CARDS[p] for p in b["cards"]] + [Path(__file__), Path(__file__).with_name("upload_psilm2_tokens.py")]
    for p in named:
        if not pushed(p):
            raise SystemExit(f"{p.resolve().relative_to(REPO)}: not on origin/main as it is here, and the "
                             "release names it or is made by it: commit and push it, then upload")
    from huggingface_hub import hf_hub_download
    api = HfApi()
    info, _ = hub_state(api)
    if info.sha != b["commit"]:
        raise SystemExit(f"{HUB} is at {info.sha[:8]}, the stage was built on {b['commit'][:8]}: if this stage was "
                         "already uploaded, run --verify (do not remove the stage); otherwise remove it and stage again")
    # the base file is a local file: what it says was published is held to the hub once more
    for p, was in {"MANIFEST.md": b["published_manifest"], **b["published_cards"]}.items():
        there = Path(hf_hub_download(HUB, p, repo_type="model", revision=b["commit"], force_download=True))
        if text(there) != was:
            raise SystemExit(f"{BASE.name}: its copy of the published {p} is not what the hub holds at {b['commit'][:8]}")
    files = sorted(["MANIFEST.md", *b["cards"]])
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(STAGE / p)) for p in files]
    done = api.create_commit(HUB, repo_type="model", operations=ops, parent_commit=b["commit"],
                             commit_message=b["title"],
                             commit_description=f"{b['what']} " + " and ".join(b["cards"]) + (
                                 " gains text and loses none; MANIFEST.md gains a dated note and lists its new size "
                                 "and sha256." if len(b["cards"]) == 1 else
                                 " gain text and lose none; MANIFEST.md gains a dated note and lists their new "
                                 "sizes and sha256.") + " No other file changes.")
    print(f"committed {done.oid[:8]} on {HUB}", flush=True)
    verify()


def verify():
    from huggingface_hub import HfApi, hf_hub_download
    b = verify_stage()
    files = sorted(["MANIFEST.md", *b["cards"]])
    api = HfApi()
    info, hub = hub_state(api)
    print(f"{HUB}@{info.sha[:8]}: private {info.private}, {len(hub)} files", flush=True)
    bad = []
    if set(hub) != set(b["files"]):
        bad.append(f"the file list changed: {sorted(set(hub) ^ set(b['files']))}")
    for p, was in b["files"].items():
        if p not in files and hub.get(p, {}).get("blob") != was["blob"]:
            bad.append(f"{p}: not the blob it was")
    for p in files:
        hub_sha = sha256(hf_hub_download(HUB, p, repo_type="model", revision=info.sha, force_download=True))
        ok = hub_sha == sha256(STAGE / p)
        print(f"  {'ok ' if ok else 'BAD'} {p}  {hub_sha[:16]}", flush=True)
        if not ok:
            bad.append(f"{p}: not as staged")
    if bad:
        raise SystemExit("\n".join(bad))
    print(f"the {len(files)} files are the staged ones and the other {len(b['files']) - len(files)} are the "
          f"blobs they were: https://huggingface.co/{HUB}", flush=True)


def self_test():
    import tempfile
    old = "# Card\n\nOne.\n\n1. first.\n2. second.\n\n## End\n"
    gained = only_gains(old, old.replace("2. second.\n", "2. second.\n   added\n").replace("## End", "## New\n\ntext\n\n## End"))
    assert sorted(gained) == ["\n", "\n", "   added\n", "## New\n", "text\n"], gained   # which blank is "new" is the differ's choice
    assert only_gains(old, old) == []
    para = "A sentence that runs\nover two lines.\n\nNext.\n"
    for spliced in (para.replace("runs\n", "runs\nnot\n"),                       # inside a paragraph
                    para.replace("\nNext.", "\n<!--\n\nNext.\n\n-->"),          # a comment around a published line
                    para.replace("\nNext.", "\n```\n\nNext.")):                # a fence opened before one
        try:
            only_gains(para, spliced)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"a gain that rewords or hides published text was accepted: {spliced!r}")
    for spliced in (para.replace("runs\n", "runs\n\n"),                         # a published paragraph split
                    para.replace("runs\n", "runs\n\nnot\n\n"),                  # a block put inside it
                    para.replace("two lines.\n", "two lines.\n---\n"),          # it made a heading
                    para.replace("two lines.\n", "two lines.\n===\n"),
                    para.replace("two lines.\n", "two lines.\nmore.\n---\n"),
                    para.replace("\nNext.", "\n<div hidden>\n\nNext.\n\n</div>")):     # HTML around a published line
        try:
            only_gains(para, spliced)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"a gain that rewords or hides published text was accepted: {spliced!r}")
    try:                                                                         # an unfinished last line finished
        only_gains("A\n\nruns into\n\nB\n", "A\n\nruns into\nnothing.\n\nB\n")
    except SystemExit:
        pass
    else:
        raise AssertionError("a gained line that finishes a published sentence was accepted")
    assert sorted(only_gains(para, para.replace("two lines.\n", "two lines.\n\n---\n"))) == ["\n", "---\n"]   # a rule
    assert sorted(only_gains(para, para.replace("\nNext.", "\nNew\n---\n\nNext."))) == ["\n", "---\n", "New\n"]  # a heading of its own
    assert only_gains(para, para.replace("\nNext.", "\nNew.\n\nNext.")) == ["New.\n", "\n"]
    card = "---\nlicense: x\n---\n\n# Card\n\nOne.\n"
    try:
        only_gains(card, card.replace("license: x\n", "license: x\ntags: y\n"))
    except SystemExit:
        pass
    else:
        raise AssertionError("a line added to the front matter was accepted")
    assert only_gains(card, card + "\nTwo.\n") == ["\n", "Two.\n"]
    tmpf = Path(tempfile.mkdtemp())
    for name, data in (("crlf", old.replace("\n", "\r\n")), ("cr-end", old[:-1] + "\r"),
                       ("one-crlf", old.replace("One.\n", "One.\r\n"))):
        (tmpf / name).write_bytes(data.encode("utf-8"))
        assert "\r" in text(tmpf / name) and "\r" not in (tmpf / name).read_text()
        try:
            only_gains(old, text(tmpf / name))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"a card whose line ends changed was accepted through the file path: {name}")
    quote = "> One.\n> Two.\n\nText.\n"
    assert len(only_gains(quote, quote.replace("> Two.\n", "> Two.\n>\n> Three.\n"))) == 2   # a block quote gains a paragraph
    for worse in (old.replace("One.", "Two."), old.replace("1. first.\n", ""), old.replace("second", "second, more"),
                  old.replace("1. first.\n2. second.\n", "2. second.\n1. first.\n"), old.replace("One.\n", "One. \n"),
                  old[:-1], old.replace("## End\n", "## End")):
        try:
            only_gains(old, worse)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"a card that changes published text was accepted: {worse!r}")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a").write_text("the new card\n")
    (tmp / "b").write_text("the partner's new card\n")
    zero = "0" * 64
    man = (f"# MANIFEST\n\nEvery file.\n\nAn earlier note.\n{TABLE}|---|---:|---|\n| `README.md` | 1 | `{zero}` |\n"
           f"| `a/README.md` | 1 | `{zero}` |\n| `b/x` | 3 | `{'3' * 64}` |\n| `constitution_model/README.md` | 1 | `{zero}` |\n")
    for cards in ({"README.md": tmp / "a"}, {"README.md": tmp / "a", "constitution_model/README.md": tmp / "b"}):
        new = new_manifest(man, cards, "The cards gained a caveat.", "2026-09-30")
        rows, was = manifest_rows(new), manifest_rows(man)
        assert list(rows) == list(was)
        for p in was:
            assert rows[p] == ((Path(cards[p]).stat().st_size, sha256(cards[p])) if p in cards else was[p]), p
        assert "An earlier note." in new and "The cards gained a caveat." in new and new.count(TABLE) == 1
        a, n = man.splitlines(keepends=True), new.splitlines(keepends=True)
        gone = [ln for tag, i1, i2, _, _ in changed(a, n) if tag != "insert" for ln in a[i1:i2]]
        assert sorted(gone) == sorted(row(p, 1, zero) + "\n" for p in cards), gone
        assert ("`README.md` and `constitution_model/README.md` gained text" in new) == (len(cards) == 2)
        assert "gained text on 2026-09-30 and lost none" in new and "rewritten on" not in new
        assert all(CARDS[q].relative_to(REPO).as_posix() in new for q in cards)
        for worse in (new.replace("Every file.", "Every file. And one more sentence."),
                      new.replace("|---|---:|---|\n", "|---|---:|---|\n| extra | line |\n"),
                      new.replace("gained a caveat", "gained two caveats")):
            assert worse != new_manifest(man, cards, "The cards gained a caveat.", "2026-09-30")
        try:
            new_manifest(new, cards, "The cards gained a caveat.", "2026-09-30")
        except SystemExit:
            pass
        else:
            raise AssertionError("the same change was written into the manifest twice")
    try:                                               # a card the manifest does not list
        new_manifest(man, {"c/README.md": tmp / "a"}, "x.", "2026-09-30")
    except SystemExit:
        pass
    else:
        raise AssertionError("a row was written for a file the manifest does not list")
    print("[self-test] results/hf_export/upload_psilm2_card.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--stage", "--upload", "--verify", "--self-test"):
        g.add_argument(flag, action="store_true")
    ap.add_argument("--what", default=None, help="with --stage: one sentence, what the cards gained")
    ap.add_argument("--title", default=None, help="with --stage: the title of the commit on the hub")
    ap.add_argument("--date", default=None, help="with --stage: the date the manifest's line carries (YYYY-MM-DD)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.stage:
        stage(a.what, a.title, a.date)
    else:
        (upload if a.upload else verify)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
