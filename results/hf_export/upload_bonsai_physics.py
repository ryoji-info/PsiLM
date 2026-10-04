#!/usr/bin/env python3
"""Stage and upload the Ternary Bonsai 2 27B physics bridge, beside the constitution bridge,
into the PRIVATE repository that upload_bonsai.py created.

  python results/hf_export/upload_bonsai_physics.py --stage    # export, card, manifest, checks; nothing leaves the machine
  python results/hf_export/upload_bonsai_physics.py --upload   # upload the new and changed files into the private repo
  python results/hf_export/upload_bonsai_physics.py --verify   # every file on the hub against the stage

Added under results/hf_export/bonsai27b/ (not in git), beside the constitution bridge's files:

  bridges/physics/bridges.safetensors   the trained physics bridge, every tensor the inference path uses, fp32
                                        as trained (the unused learned-pointer tensors are dropped by the exporter)
  bridges/physics/config.json           how to build it: coupling depths, construction, physics model, training
  physics/fno_burgers_singlemode.safetensors
                                        the frozen 1D Burgers FNO, byte for byte the file in ryoji-info/Qwen3.5-9B-PsiLM
  README.md                             the card (results/hf_export/bonsai27b_card.md, which IS in git), rewritten
  MANIFEST.md                           every file, its size and sha256, rewritten

The constitution bridge's five files are left as they are; the stage refuses to go on unless the staged copies
are the ones on the hub. Visibility is not handled here: the upload refuses a repository that is not private, and
the maintainer makes it public by hand. Uses the ambient `hf auth login`; no credentials pass through.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_HOME = "/Users/rxiii/Documents/huggingface"
if os.path.isfile(os.path.join(_HOME, "token")):
    os.environ.setdefault("HF_HOME", _HOME)

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results/stage2_bonsai27b"
VERDICT = REPO / "results/bonsai/physics_verdict.json"
STAGE = REPO / "results/hf_export/bonsai27b"
CARD = REPO / "results/hf_export/bonsai27b_card.md"
FNO_SRC = REPO / "results/hf_export/physics/fno_burgers_singlemode.safetensors"
FNO_SHA = "7bb0076c85cdcf2505a9079c05964e3eb77ac4a776eccf34216953f3c37bfcdd"   # ryoji-info/Qwen3.5-9B-PsiLM's file
BACKBONE = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
HUB = "ryoji-info/Ternary-Bonsai-2-27B-PsiLM"
CONST = ["bridges/all/bridges.safetensors", "bridges/all/bridges.safetensors.meta", "bridges/all/config.json",
         "bridges/all/tokens.safetensors", "bridges/all/tokens.json"]
PHYS = ["bridges/physics/bridges.safetensors", "bridges/physics/config.json",
        "physics/fno_burgers_singlemode.safetensors"]
NEW = ["README.md", "MANIFEST.md"] + PHYS                 # what this script uploads
FILES = sorted(CONST + NEW)
DROPPED = {"fwd.x0_query", "fwd.x0_key.weight", "fwd.x0_key.bias"}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def staged():
    return sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file() and p.name != ".DS_Store")


def hub_files():
    """{path: sha256} of the repository as it is now (a small file is downloaded and hashed)."""
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().repo_info(HUB, repo_type="model", files_metadata=True)
    out = {}
    for s in info.siblings:
        if s.rfilename == ".gitattributes":
            continue
        sha = (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
        out[s.rfilename] = sha or sha256(hf_hub_download(HUB, s.rfilename, repo_type="model", force_download=True))
    return info, out


def stage():
    verdict = json.loads(VERDICT.read_text())
    if verdict.get("verdict") != "works" or verdict.get("checkpoint_step") != 11500:
        raise SystemExit(f"{VERDICT}: the verdict is not 'works' at step 11500; nothing to release")
    meta = json.loads((RUN / "bridges.npz.meta").read_text())
    if meta["step"] != verdict["checkpoint_step"] or [meta["l_fwd"], meta["l_rev"]] != verdict["couple"][:2]:
        raise SystemExit("the run's checkpoint is not the one the verdict is about")
    if (STAGE / "bridges/physics").exists() or (STAGE / "physics").exists():
        raise SystemExit(f"{STAGE}: the physics bridge is staged already (remove bridges/physics and physics/ to stage again)")
    _, hub = hub_files()
    for f in CONST:                                   # the constitution files stay what the hub has
        if sha256(STAGE / f) != hub.get(f):
            raise SystemExit(f"{f}: the staged copy is not the hub's; run upload_bonsai.py --verify first")
    subprocess.run([sys.executable, str(REPO / "eval/export_bridges.py"), "--run", str(RUN.relative_to(REPO)),
                    "--out", str(STAGE / "bridges/physics"), "--n-layers", "64", "--backbone-name", BACKBONE],
                   check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, stdout=subprocess.DEVNULL)
    fix_config(STAGE / "bridges/physics/config.json")
    (STAGE / "physics").mkdir()
    shutil.copyfile(FNO_SRC, STAGE / "physics/fno_burgers_singlemode.safetensors")
    write_card_and_manifest()
    check()


def fix_config(path):
    """Three fields the exporter fills from the last chunk's arguments or from a constant, said here as they were for
    this run: the warm-up ran at batch 8 (results/bonsai/physics_bonsai.sh, `train_to $A_END 8 3e-4`), every in-run
    evaluation scored 8 rollouts (physics_bonsai.log; there is no coupled.log), and the FNO has 72,033 parameters
    counting each complex spectral weight once (psilm_infer.py prints 72K)."""
    cfg = json.loads(path.read_text())
    assert cfg["training"]["batch"] == 2 and cfg["training"]["phase_A_readout_only"] == 2000
    cfg["training"]["batch"] = "8 (warm-up), 2 (coupled, no-harm phase)"
    cfg["held_out_n_source"] = ("8 rollouts on the first 8 validation questions at the end of every chunk, "
                                "throughout the run (results/bonsai/physics_bonsai.log in the ΨLM repository)")
    assert "70K params" in cfg["physics"]["source"]
    cfg["physics"]["source"] = cfg["physics"]["source"].replace(
        "70K params", "72K parameters, each complex spectral weight counted once")
    path.write_text(json.dumps(cfg, indent=1) + "\n")


def write_card_and_manifest():
    (STAGE / "README.md").write_bytes(CARD.read_bytes())
    rows = [f"| `{f}` | {(STAGE / f).stat().st_size:,} | `{sha256(STAGE / f)}` |" for f in FILES if f != "MANIFEST.md"]
    (STAGE / "MANIFEST.md").write_text(
        "# MANIFEST\n\nEvery file in this repository but this one, its size in bytes and its sha256, written when "
        "the release was staged\nfrom the ΨLM checkout (`results/hf_export/upload_bonsai.py --stage` for the "
        "constitution bridge, then\n`results/hf_export/upload_bonsai_physics.py --stage` for the physics bridge). "
        "The constitution bridge's files\ncome from `eval/export_constitution_bridge.py`, which reads each file back "
        "and compares it bit for bit with\nthe training checkpoint; the physics bridge's from `eval/export_bridges.py`, "
        "whose every tensor the stage compared\nbit for bit with the checkpoint. The physics model is byte for byte "
        "the file in ryoji-info/Qwen3.5-9B-PsiLM.\n\n| path | bytes | sha256 |\n|---|---:|---|\n" + "\n".join(rows) + "\n")


def same_card():
    if (STAGE / "README.md").read_bytes() != CARD.read_bytes():
        raise SystemExit(f"the staged README.md is not {CARD.relative_to(REPO)}: run --restage-card")


def check():
    """What is staged is what is meant to go, says nothing of this machine, and is the checkpoint the verdict judged."""
    import mlx.core as mx
    import numpy as np
    same_card()
    have = staged()
    if have != FILES:
        raise SystemExit(f"staged files are not the release's: {sorted(set(have) ^ set(FILES))}")
    me = Path.home().name
    for f in have:
        if f.endswith((".json", ".md", ".meta")):
            text = (STAGE / f).read_text()
            for what in (str(Path.home()), f"/{me}/", f"Users/{me}", '"~/', "/private/", "/tmp/"):
                if what in text:
                    raise SystemExit(f"{f}: a local path is in it ({what!r})")
    cfg = json.loads((STAGE / "bridges/physics/config.json").read_text())
    assert cfg["backbone"] == BACKBONE and cfg["hf_tokenizer"] == BACKBONE
    assert cfg["coupling"] == {"l_fwd": 26, "l_rev": 48, "n_layers": 64}, cfg["coupling"]
    assert cfg["physics"]["file"] == "physics/fno_burgers_singlemode.safetensors"
    assert cfg["training"]["steps_total"] == 11500
    assert cfg["training"]["batch"] == "8 (warm-up), 2 (coupled, no-harm phase)" and "72K" in cfg["physics"]["source"]
    assert sha256(STAGE / "physics/fno_burgers_singlemode.safetensors") == FNO_SHA
    z = np.load(RUN / "bridges.npz")
    w = mx.load(str(STAGE / "bridges/physics/bridges.safetensors"))
    if set(z.files) - set(w) != DROPPED or set(w) - set(z.files):
        raise SystemExit(f"tensor sets differ: {sorted(set(z.files) ^ set(w))}")
    for k, v in w.items():
        a = np.array(v)
        if a.dtype != z[k].dtype or a.shape != z[k].shape or not np.array_equal(a, z[k]):
            raise SystemExit(f"{k}: not the checkpoint's tensor")
    manifest = (STAGE / "MANIFEST.md").read_text()
    for f in have:
        if f != "MANIFEST.md" and sha256(STAGE / f) not in manifest:
            raise SystemExit(f"{f}: not the file the manifest lists")
    n = sum(int(np.prod(v.shape)) for v in w.values())
    print(f"[check] {len(have)} files staged; the physics bridge is the step-11500 checkpoint bit for bit, "
          f"{len(w)} tensors, {n:,} values; reads {cfg['coupling']['l_fwd']}, writes {cfg['coupling']['l_rev']} "
          f"of {cfg['coupling']['n_layers']}; physics model {FNO_SHA[:12]}")


def upload():
    from huggingface_hub import HfApi
    check()
    api = HfApi()
    info, hub = hub_files()
    if not info.private:
        raise SystemExit(f"{HUB} is PUBLIC: nothing is uploaded to a public repository from here")
    for f in CONST:
        if hub.get(f) != sha256(STAGE / f):
            raise SystemExit(f"{f}: the hub's copy is not the staged one")
    print(f"{HUB} is private: uploading {len(NEW)} files into it", flush=True)
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            api.upload_folder(folder_path=str(STAGE), repo_id=HUB, repo_type="model", allow_patterns=NEW,
                              commit_message="the physics bridge on Ternary Bonsai 2 27B: bridges, physics model, card")
            print(f"uploaded in {(time.time() - t0) / 60:.1f} min", flush=True)
            break
        except Exception as e:                                         # noqa: BLE001
            print(f"attempt {attempt} failed: {e!s:.200}", flush=True)
            if attempt == 5:
                raise SystemExit("the upload did not go through")
            time.sleep(30)
    verify(require_private=True)


def verify(require_private=False):
    same_card()
    info, hub = hub_files()
    print(f"{HUB}: private {info.private}, {len(hub)} files, head {info.sha[:12]}", flush=True)
    if require_private and not info.private:
        raise SystemExit("the repository is not private")
    bad = sorted(set(hub) ^ set(FILES))
    for f in FILES:
        if f in hub:
            ok = hub[f] == sha256(STAGE / f)
            print(f"  {'ok ' if ok else 'BAD'} {f}  {hub[f][:16]}", flush=True)
            if not ok:
                bad.append(f)
    if bad:
        raise SystemExit(f"not as staged: {bad}")
    print(f"every file on the hub is the staged one: https://huggingface.co/{HUB}", flush=True)


def restage_card():
    """A card corrected after staging: copy it in again and rewrite the manifest, then check."""
    write_card_and_manifest()
    check()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", action="store_true")
    g.add_argument("--restage-card", action="store_true")
    g.add_argument("--upload", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    (stage if a.stage else restage_card if a.restage_card else upload if a.upload else verify)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
