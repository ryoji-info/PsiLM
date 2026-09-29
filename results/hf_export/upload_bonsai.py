#!/usr/bin/env python3
"""Stage and upload the Ternary Bonsai 2 27B constitution bridge, to a PRIVATE repo.

  python results/hf_export/upload_bonsai.py --stage            # export, manifest, checks; nothing leaves the machine
  python results/hf_export/upload_bonsai.py --upload           # create the repo PRIVATE and upload the staged files
  python results/hf_export/upload_bonsai.py --verify           # the hub's files against the manifest

Staged under results/hf_export/bonsai27b/ (not in git):

  README.md                           the card (results/hf_export/bonsai27b_card.md, which IS in git)
  MANIFEST.md                         every file, its size and sha256
  bridges/all/bridges.safetensors     the trained bridges, every tensor, fp32 as trained
  bridges/all/bridges.safetensors.meta, config.json
  bridges/all/tokens.safetensors      the stored tokens the write attends to, and tokens.json, their record
                                      (what they were made from, and the verdict on using them with no partner)

The backbone is not redistributed (prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) and the
partner model is in ryoji-info/PsiLM-2 (constitution_model/).

Visibility is deliberately NOT handled here: the repository is created private, and
it is refused if it already exists and is not. The maintainer makes it public by
hand. Uses the ambient `hf auth login`; no credentials pass through.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_HOME = "/Users/rxiii/Documents/huggingface"
if os.path.isfile(os.path.join(_HOME, "token")):
    os.environ.setdefault("HF_HOME", _HOME)

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results/stage2c_bonsai27b_all"
STAGE = REPO / "results/hf_export/bonsai27b"
CARD = REPO / "results/hf_export/bonsai27b_card.md"
BACKBONE = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
HUB = "ryoji-info/Ternary-Bonsai-2-27B-PsiLM"
FILES = ["README.md", "MANIFEST.md", "bridges/all/bridges.safetensors", "bridges/all/bridges.safetensors.meta",
         "bridges/all/config.json", "bridges/all/tokens.safetensors", "bridges/all/tokens.json"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def staged():
    return sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file() and p.name != ".DS_Store")


def stage():
    verdict = json.loads((RUN / "fixed_tokens_mean.json").read_text()).get("verdict") or {}
    if verdict.get("verdict") not in ("stands_in", "does_not"):
        raise SystemExit(f"{RUN}/fixed_tokens_mean.json carries no verdict: run results/bonsai/stored_tokens.sh first")
    if not CARD.exists():
        raise SystemExit(f"{CARD}: the card is not written")
    if STAGE.exists():
        raise SystemExit(f"{STAGE} exists: a release is staged once (remove it to stage again)")
    out = STAGE / "bridges/all"
    subprocess.run([sys.executable, str(REPO / "eval/export_constitution_bridge.py"), "--run",
                    str(RUN.relative_to(REPO)), "--out", str(out), "--tokens",
                    str((RUN / "fixed_tokens_mean.npz").relative_to(REPO)), "--backbone-name", BACKBONE],
                   check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)})
    (STAGE / "README.md").write_bytes(CARD.read_bytes())
    rows = [f"| `{f}` | {(STAGE / f).stat().st_size:,} | `{sha256(STAGE / f)}` |"
            for f in staged() if f != "MANIFEST.md"]
    (STAGE / "MANIFEST.md").write_text(
        "# MANIFEST\n\nEvery file in this repository but this one, its size in bytes and its sha256, written when "
        "the release was staged\nfrom the ΨLM checkout (`results/hf_export/upload_bonsai.py --stage`). The bridges "
        "and the tokens come from\n`eval/export_constitution_bridge.py`, which reads each file back and compares it "
        "bit for bit with the training\ncheckpoint.\n\n| path | bytes | sha256 |\n|---|---:|---|\n"
        + "\n".join(rows) + "\n")
    check()
    print(f"staged {len(staged())} files under {STAGE} ({sum((STAGE / f).stat().st_size for f in staged()) / 1e6:.1f} MB); "
          f"the tokens' verdict: {verdict['verdict']}")


def same_card():
    """The staged card is the one in git: a card corrected after staging is not left behind."""
    if (STAGE / "README.md").read_bytes() != CARD.read_bytes():
        raise SystemExit(f"the staged README.md is not {CARD.relative_to(REPO)}: remove "
                         f"{STAGE.relative_to(REPO)} and run --stage again")


def check():
    """What is staged is what is meant to go, and says nothing of this machine."""
    same_card()
    have = staged()
    if have != sorted(FILES):
        raise SystemExit(f"staged files are not the release's: {sorted(set(have) ^ set(FILES))}")
    me = Path.home().name
    for f in have:
        if f.endswith((".json", ".md", ".meta")):
            text = (STAGE / f).read_text()
            for what in (str(Path.home()), f"/{me}/", f"Users/{me}", '"~/', "/private/", "/tmp/"):
                if what in text:
                    raise SystemExit(f"{f}: a local path is in it ({what!r})")
    cfg = json.loads((STAGE / "bridges/all/config.json").read_text())
    tok = json.loads((STAGE / "bridges/all/tokens.json").read_text())
    assert cfg["sha256_safetensors"] == sha256(STAGE / "bridges/all/bridges.safetensors")
    assert tok["bridges_sha256"] == cfg["sha256_safetensors"]
    assert tok["tokens_sha256"] == sha256(STAGE / "bridges/all/tokens.safetensors") == cfg["stored_tokens"]["sha256"]
    assert cfg["backbone"] == BACKBONE and tok["model"] == BACKBONE
    manifest = (STAGE / "MANIFEST.md").read_text()
    for f in have:
        if f != "MANIFEST.md" and sha256(STAGE / f) not in manifest:
            raise SystemExit(f"{f}: not the file the manifest lists")
    sys.path.insert(0, str(REPO))
    from psilm.mlx.constitution import load_stored_stack, stored_verdict
    coupler, meta, rec = load_stored_stack(STAGE / "bridges/all/bridges.safetensors")
    print(f"[check] the staged bridge loads with no partner: tokens {coupler.ident}, "
          f"{tuple(coupler.stored.shape)}, verdict {stored_verdict(rec)}, write at layer {meta['l_rev']}")


def upload():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    check()
    api = HfApi()
    try:
        info = api.repo_info(HUB, repo_type="model")
        if not info.private:
            raise SystemExit(f"{HUB} exists and is PUBLIC: nothing is uploaded to a public repository from here")
        print(f"{HUB} exists (private): uploading into it", flush=True)
    except RepositoryNotFoundError:
        api.create_repo(HUB, repo_type="model", private=True, exist_ok=False)
        print(f"created {HUB} (private)", flush=True)
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            api.upload_folder(folder_path=str(STAGE), repo_id=HUB, repo_type="model",
                              ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
                              commit_message="the constitution bridge on Ternary Bonsai 2 27B: bridges, stored tokens, card")
            print(f"uploaded in {(time.time() - t0) / 60:.1f} min", flush=True)
            break
        except Exception as e:                                         # noqa: BLE001
            print(f"attempt {attempt} failed: {e!s:.200}", flush=True)
            if attempt == 5:
                raise SystemExit("the upload did not go through")
            time.sleep(30)
    verify(require_private=True)


def verify(require_private=False):
    """The hub's files against the staged ones. After an upload the repository must still be
    private; asked for alone (--verify) this also checks a repository its maintainer made public."""
    from huggingface_hub import HfApi, hf_hub_download
    same_card()
    api = HfApi()
    info = api.repo_info(HUB, repo_type="model", files_metadata=True)
    there = {s.rfilename: s for s in info.siblings if s.rfilename != ".gitattributes"}
    print(f"{HUB}: private {info.private}, {len(there)} files", flush=True)
    if require_private and not info.private:
        raise SystemExit("the repository is not private")
    bad = sorted(set(there) ^ set(FILES))
    for f in FILES:
        if f not in there:
            continue
        s = there[f]
        hub_sha = (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
        if hub_sha is None:                                           # a small file: download and hash it
            hub_sha = sha256(hf_hub_download(HUB, f, repo_type="model", force_download=True))
        ok = hub_sha == sha256(STAGE / f)
        print(f"  {'ok ' if ok else 'BAD'} {f}  {hub_sha[:16]}", flush=True)
        if not ok:
            bad.append(f)
    if bad:
        raise SystemExit(f"not as staged: {bad}")
    print(f"every file on the hub is the staged one: https://huggingface.co/{HUB}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", action="store_true")
    g.add_argument("--upload", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    (stage if a.stage else upload if a.upload else verify)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
