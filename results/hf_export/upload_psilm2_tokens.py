#!/usr/bin/env python3
"""Add the stored tokens of the Qwen3.5 9B full-width bridge to ryoji-info/PsiLM-2.

  python results/hf_export/upload_psilm2_tokens.py --stage     # build what would go; nothing leaves the machine
  python results/hf_export/upload_psilm2_tokens.py --upload    # ONE commit on the hub, then --verify
  python results/hf_export/upload_psilm2_tokens.py --verify    # the hub against the staged manifest

The repository is PUBLIC and holds the release of the PsiLM-2 paper. This adds two
files and rewrites three, in one commit, and touches nothing else:

  bridges/qwen3.5-9b/all/tokens.safetensors   new: the stored tokens (eval/export_constitution_bridge.py --tokens)
  bridges/qwen3.5-9b/all/tokens.json          new: their record, with the verdict on using them in the partner's place
  bridges/qwen3.5-9b/all/config.json          the published one plus a `stored_tokens` block, nothing else
  README.md                                   the card (results/hf_export/psilm2_card.md; --upload refuses
                                              unless it, and this file, are on origin/main as they are here)
  MANIFEST.md                                 the published one with those four rows written, no other row changed

Every safeguard is a refusal:
- the bridges exported now must be, byte for byte, the bridges the hub holds: the
  tokens' record names them by their sha256;
- the config.json written now must be the hub's with `stored_tokens` added;
- the stage is built on ONE commit of the hub and the upload is made onto that
  commit only (`parent_commit`): if the hub moved, stage again;
- the card and this uploader, which the release names, must be pushed before the upload;
- after the upload every file but the five must be the blob it was.

Visibility is not touched here. Uses the ambient `hf auth login`; no credentials
pass through.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_HOME = "/Users/rxiii/Documents/huggingface"
if os.path.isfile(os.path.join(_HOME, "token")):
    os.environ.setdefault("HF_HOME", _HOME)

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results/stage2c_qwen35_all"
STAGE = REPO / "results/hf_export/psilm2_tokens"
CARD = REPO / "results/hf_export/psilm2_card.md"
HUB = "ryoji-info/PsiLM-2"
BACKBONE = "ryoji-info/Qwen3.5-9B-PsiLM"
DIR = "bridges/qwen3.5-9b/all"
NEW = [f"{DIR}/tokens.json", f"{DIR}/tokens.safetensors"]
REWRITTEN = ["MANIFEST.md", "README.md", f"{DIR}/config.json"]
FILES = sorted(NEW + REWRITTEN)
NOTE = ("\n`bridges/qwen3.5-9b/all/tokens.safetensors` and `tokens.json` were added on 2026-09-30 "
        "(`results/hf_export/upload_psilm2_tokens.py`):\nthe stored tokens of that bridge, from "
        "`eval/export_constitution_bridge.py --tokens`. Their rows were written then, and the\nrows of "
        "`README.md` and of that bridge's `config.json` (which gained a `stored_tokens` block) rewritten; "
        "no other row\nand no other file changed.\n")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_state(api, revision=None):
    info = api.repo_info(HUB, repo_type="model", files_metadata=True, revision=revision)
    files = {}
    for s in info.siblings:
        lfs = (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
        files[s.rfilename] = {"size": s.size, "blob": s.blob_id, "lfs": lfs}
    return info, files


def manifest_rows(text):
    """{path: (bytes, sha256)} of a MANIFEST.md, and the lines of its table in order."""
    rows = {}
    for ln in text.splitlines():
        m = re.fullmatch(r"\| `([^`]+)` \| ([0-9,]+) \| `([0-9a-f]{64})` \|", ln)
        if m:
            rows[m.group(1)] = (int(m.group(2).replace(",", "")), m.group(3))
    return rows


def row(path, file) -> str:
    return f"| `{path}` | {Path(file).stat().st_size:,} | `{sha256(file)}` |"


def new_manifest(old: str, stage: Path) -> str:
    """The published manifest with the rows of the five files' four listed ones written (the
    manifest does not list itself), in the table's own order, and a note under its head."""
    want = {p: row(p, stage / p) for p in FILES if p != "MANIFEST.md"}
    head, _, table = old.partition("\n| path | bytes | sha256 |\n")
    if not table or NOTE in old:
        raise SystemExit("the published MANIFEST.md is not the one this was written for")
    lines = table.splitlines()
    rule, body = lines[0], lines[1:]
    paths = [re.match(r"\| `([^`]+)` ", ln).group(1) for ln in body]
    if paths != sorted(paths):
        raise SystemExit("the published manifest's rows are not in sorted order: written by hand?")
    kept = {p: ln for p, ln in zip(paths, body) if p not in want}
    every = {**kept, **want}
    out = head.rstrip("\n") + "\n" + NOTE + "\n| path | bytes | sha256 |\n" + rule + "\n"
    return out + "\n".join(every[p] for p in sorted(every)) + "\n"


def stage():
    from huggingface_hub import HfApi, hf_hub_download
    if not CARD.exists():
        raise SystemExit(f"{CARD}: the card is not written")
    if STAGE.exists():
        raise SystemExit(f"{STAGE} exists: remove it to stage again")
    rec = json.loads((RUN / "fixed_tokens_mean.json").read_text())
    if (rec.get("verdict") or {}).get("verdict") != "stands_in":
        raise SystemExit(f"{RUN}/fixed_tokens_mean.json: these tokens were not shown to stand in; nothing is staged")
    api = HfApi()
    info, files = hub_state(api)
    if any(p in files for p in NEW):
        raise SystemExit(f"{HUB} already holds {[p for p in NEW if p in files]}")
    published = {p: Path(hf_hub_download(HUB, p, repo_type="model", revision=info.sha, force_download=True))
                 for p in ("MANIFEST.md", f"{DIR}/config.json")}
    listed = manifest_rows(published["MANIFEST.md"].read_text())
    # the manifest the hub publishes is true of the hub, for every file stored with its hash
    for p, (size, sha) in listed.items():
        if p not in files or files[p]["size"] != size or (files[p]["lfs"] and files[p]["lfs"] != sha):
            raise SystemExit(f"{p}: the hub's file is not the one its manifest lists")
    tmp = STAGE / ".export"
    subprocess.run([sys.executable, str(REPO / "eval/export_constitution_bridge.py"), "--run",
                    str(RUN.relative_to(REPO)), "--out", str(tmp), "--tokens",
                    str((RUN / "fixed_tokens_mean.npz").relative_to(REPO)), "--backbone-name", BACKBONE],
                   check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)})
    for name in ("bridges.safetensors", "bridges.safetensors.meta"):
        if (tmp / name).stat().st_size != listed[f"{DIR}/{name}"][0] or sha256(tmp / name) != listed[f"{DIR}/{name}"][1]:
            raise SystemExit(f"{name}: exported now, it is not the file {HUB} holds; the tokens would name other bridges")
    cfg, was = json.loads((tmp / "config.json").read_text()), json.loads(published[f"{DIR}/config.json"].read_text())
    if {k: v for k, v in cfg.items() if k != "stored_tokens"} != was or "stored_tokens" in was:
        raise SystemExit("config.json: more than the `stored_tokens` block differs from the published one")
    (STAGE / DIR).mkdir(parents=True)
    for name in ("tokens.safetensors", "tokens.json", "config.json"):
        (tmp / name).replace(STAGE / DIR / name)
    for leftover in tmp.iterdir():
        leftover.unlink()
    tmp.rmdir()
    (STAGE / "README.md").write_bytes(CARD.read_bytes())
    (STAGE / "MANIFEST.md").write_text(new_manifest(published["MANIFEST.md"].read_text(), STAGE))
    (STAGE.parent / "psilm2_tokens_base.json").write_text(json.dumps(
        {"hub": HUB, "commit": info.sha, "files": files}, indent=1) + "\n")
    check()
    print(f"staged on {HUB}@{info.sha[:8]}: {', '.join(FILES)}")


def base():
    f = STAGE.parent / "psilm2_tokens_base.json"
    if not f.exists():
        raise SystemExit("nothing is staged (--stage)")
    return json.loads(f.read_text())


def check():
    """What is staged is the five files, says nothing of this machine, and loads."""
    if (STAGE / "README.md").read_bytes() != CARD.read_bytes():
        raise SystemExit(f"the staged README.md is not {CARD.relative_to(REPO)}: remove "
                         f"{STAGE.relative_to(REPO)} and run --stage again")
    have = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file() and p.name != ".DS_Store")
    if have != FILES:
        raise SystemExit(f"staged files are not the five: {sorted(set(have) ^ set(FILES))}")
    me = Path.home().name
    for f in have:
        if f.endswith((".json", ".md")):
            text = (STAGE / f).read_text()
            for what in (str(Path.home()), f"/{me}/", f"Users/{me}", '"~/', "/private/", "/tmp/"):
                if what in text:
                    raise SystemExit(f"{f}: a local path is in it ({what!r})")
    b = base()
    rows = manifest_rows((STAGE / "MANIFEST.md").read_text())
    if set(rows) != (set(b["files"]) | set(NEW)) - {"MANIFEST.md", ".gitattributes"}:
        raise SystemExit("the staged manifest does not list the hub's files and the two new ones")
    for p in FILES:
        if p != "MANIFEST.md" and rows[p] != ((STAGE / p).stat().st_size, sha256(STAGE / p)):
            raise SystemExit(f"{p}: not the file the staged manifest lists")
    tok = json.loads((STAGE / DIR / "tokens.json").read_text())
    cfg = json.loads((STAGE / DIR / "config.json").read_text())
    assert tok["bridges_sha256"] == rows[f"{DIR}/bridges.safetensors"][1] == cfg["sha256_safetensors"]
    assert tok["tokens_sha256"] == sha256(STAGE / DIR / "tokens.safetensors") == cfg["stored_tokens"]["sha256"]
    assert tok["model"] == cfg["backbone"] == BACKBONE and tok["verdict"]["verdict"] == "stands_in"
    assert tok["bridges"] == "bridges.safetensors" and tok["ckpt"] == str((RUN / "bridges.npz").relative_to(REPO))
    return b


def pushed(p: Path) -> bool:
    """The file here is, byte for byte, the blob origin/main holds at its path."""
    rel = p.resolve().relative_to(REPO).as_posix()
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)   # noqa: E731
    f = git("fetch", "--quiet", "origin", "main")
    if f.returncode != 0:                      # a stale origin/main would let a file pass that is not there
        raise SystemExit("git fetch origin main failed; what origin/main holds cannot be told: "
                         + f.stderr.strip()[:200])
    held = git("rev-parse", "--verify", "--quiet", f"origin/main:{rel}")
    return held.returncode == 0 and held.stdout.strip() == git("hash-object", rel).stdout.strip()


def upload():
    from huggingface_hub import CommitOperationAdd, HfApi
    b = check()
    for p in (CARD, Path(__file__)):
        if not pushed(p):
            raise SystemExit(f"{p.resolve().relative_to(REPO)}: not on origin/main as it is here, and the "
                             "release names it: commit and push it, then upload")
    api = HfApi()
    info, files = hub_state(api)
    if info.sha != b["commit"]:
        raise SystemExit(f"{HUB} is at {info.sha[:8]}, the stage was built on {b['commit'][:8]}: stage again")
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(STAGE / p)) for p in FILES]
    done = api.create_commit(
        HUB, repo_type="model", operations=ops, parent_commit=b["commit"],
        commit_message="Stored tokens for the 9B full-width bridge: it runs with no partner model",
        commit_description="Adds bridges/qwen3.5-9b/all/tokens.safetensors and tokens.json; config.json of that "
                           "bridge gains a `stored_tokens` block; the card says what was measured; MANIFEST.md "
                           "lists the two new files. No other file changes.")
    print(f"committed {done.oid[:8]} on {HUB}", flush=True)
    verify()


def verify():
    from huggingface_hub import HfApi, hf_hub_download
    b = check()
    api = HfApi()
    info, files = hub_state(api)
    print(f"{HUB}@{info.sha[:8]}: private {info.private}, {len(files)} files", flush=True)
    bad = []
    if set(files) != set(b["files"]) | set(NEW):
        bad.append(f"the file list changed by more than the two new files: {sorted(set(files) ^ (set(b['files']) | set(NEW)))}")
    for p, was in b["files"].items():
        if p not in REWRITTEN and files.get(p, {}).get("blob") != was["blob"]:
            bad.append(f"{p}: not the blob it was")
    for p in FILES:
        if p not in files:
            bad.append(f"{p}: not on the hub")
            continue
        hub_sha = files[p]["lfs"] or sha256(hf_hub_download(HUB, p, repo_type="model", revision=info.sha,
                                                            force_download=True))
        ok = hub_sha == sha256(STAGE / p)
        print(f"  {'ok ' if ok else 'BAD'} {p}  {hub_sha[:16]}", flush=True)
        if not ok:
            bad.append(f"{p}: not as staged")
    if bad:
        raise SystemExit("\n".join(bad))
    print(f"the five files are the staged ones and the other {len(b['files']) - len(REWRITTEN)} are the blobs they "
          f"were: https://huggingface.co/{HUB}", flush=True)


def self_test():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / DIR).mkdir(parents=True)
    for p in FILES:
        if p != "MANIFEST.md":
            (tmp / p).write_text(p)
    old = ("# MANIFEST\n\nEvery file.\n\n| path | bytes | sha256 |\n|---|---:|---|\n"
           f"| `README.md` | 1 | `{'0' * 64}` |\n| `{DIR}/bridges.safetensors` | 7 | `{'1' * 64}` |\n"
           f"| `{DIR}/config.json` | 2 | `{'2' * 64}` |\n| `physics/x` | 3 | `{'3' * 64}` |\n")
    new = new_manifest(old, tmp)
    rows, was = manifest_rows(new), manifest_rows(old)
    assert list(rows) == sorted(rows) and set(rows) == set(was) | set(NEW)
    assert rows[f"{DIR}/bridges.safetensors"] == was[f"{DIR}/bridges.safetensors"] and rows["physics/x"] == was["physics/x"]
    for p in FILES:
        if p != "MANIFEST.md":
            assert rows[p] == (len(p), hashlib.sha256(p.encode()).hexdigest()), p
    assert NOTE in new and new.startswith("# MANIFEST\n\nEvery file.\n")
    # every line of the old table that was not one of the four is in the new one, unchanged
    assert [ln for ln in old.splitlines() if "bridges.safetensors`" in ln or "physics" in ln] == \
           [ln for ln in new.splitlines() if "bridges.safetensors`" in ln or "physics" in ln]
    for broken in (new, old.replace("| `physics/x`", "| `a/x`")):        # staged twice; rows out of order
        try:
            new_manifest(broken, tmp)
        except SystemExit:
            pass
        else:
            raise AssertionError("a manifest this was not written for was rewritten")
    print("[self-test] results/hf_export/upload_psilm2_tokens.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--stage", "--upload", "--verify", "--self-test"):
        g.add_argument(flag, action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    (stage if a.stage else upload if a.upload else verify)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
