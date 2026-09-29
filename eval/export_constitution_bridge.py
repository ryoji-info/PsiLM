#!/usr/bin/env python3
"""Export a bridge checkpoint (constitution or physics) to the Hugging Face layout.

Reads ``<run>/bridges.npz`` + ``bridges.npz.meta`` and writes
``<out>/bridges.safetensors`` (every tensor, unchanged, fp32 as trained) and
``<out>/config.json`` (the meta, plus the export's provenance: source run,
step, tensor count, parameter count, sha256 of the safetensors). Then reads
the safetensors back with ``mx.load`` -- the same call
``psilm.mlx.constitution.load_constitution_bridge_weights`` makes -- and
refuses to finish unless every array is bit-identical to the npz.

With ``--tokens <fixed_tokens_mean.npz>`` the bridge's stored tokens go beside it
as ``tokens.safetensors`` and ``tokens.json`` (the record of what they were made
from, which now also names the exported bridges by hash), so that
``psilm.mlx.constitution.load_stored_stack(<out>/bridges.safetensors)`` runs the
bridge with no partner model. The export is then loaded back that way and the
tokens compared bit for bit.

  python3 eval/export_constitution_bridge.py --run results/stage2c_qwen35_all \
      --out results/hf_export/psilm2/bridges/qwen3.5-9b/all
  python3 eval/export_constitution_bridge.py --run results/stage2c_bonsai27b_all \
      --out results/hf_export/bonsai27b/bridges/all --tokens results/stage2c_bonsai27b_all/fixed_tokens_mean.npz \
      --backbone-name prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
  python3 eval/export_constitution_bridge.py --all   # every stage2c_* run with a step-1000/2000 checkpoint
"""
import argparse, glob, hashlib, json, os
from pathlib import Path

import mlx.core as mx
import numpy as np

BACKBONE_DIR = {"qwen35": "qwen3.5-9b", "qwen0.5b": "qwen2.5-0.5b"}
# the published name of each backbone, written in place of the local path the
# training meta records (config.json "backbone", and every meta field that held it)
BACKBONE_NAME = {"qwen35": "ryoji-info/Qwen3.5-9B-PsiLM",
                 "qwen0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"}


def scrub_meta(meta: dict, backbone_name: str | None) -> dict:
    """The meta is copied into the export verbatim except for local paths: the
    backbone's path (meta["model"], and every field equal to it, such as
    hf_tokenizer and args.model) becomes the published backbone name, a path
    inside this checkout is written relative to it, and any other path under
    the exporting user's home directory is written with `~`.
    Nothing a loader reads (shapes, mask, cap, const_model, step) changes."""
    home = str(Path.home())
    repo = str(Path(__file__).resolve().parents[1])
    local_backbone = meta.get("model")

    def fix(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                fix(v)
            elif isinstance(v, str):
                if backbone_name and local_backbone and v == local_backbone:
                    d[k] = backbone_name
                elif v.startswith(repo + "/"):          # as a run started inside the checkout records it
                    d[k] = v[len(repo) + 1:]
                elif v.startswith(home + "/"):
                    d[k] = "~" + v[len(home):]
    out = json.loads(json.dumps(meta))
    fix(out)
    return out


REPO = Path(__file__).resolve().parents[1]
RECORD_FIELDS = ("kind", "n", "tokens_sha256", "ckpt", "ckpt_step", "ckpt_sha256")


def shown(path) -> str:
    """A path as an export may print it: relative to the repository, or its last name."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return Path(path).name


def check_tokens(src: Path, npz: Path, meta: dict):
    """The tokens and their record, or SystemExit: they must be THIS run's. The
    export names the exported bridges in the record by hash, and a loader then takes
    that hash as the tokens' origin, so it is written only for tokens that were
    made from the checkpoint being exported."""
    from psilm.mlx.constitution import load_stored_tokens
    rec_f = src.with_suffix(".json")
    if not rec_f.exists():
        raise SystemExit(f"{src}: no record beside it ({rec_f.name})")
    rec = json.loads(rec_f.read_text())
    missing = [k for k in RECORD_FIELDS if k not in rec]
    if missing:
        raise SystemExit(f"{rec_f}: the record lacks {missing}")
    if rec["ckpt_sha256"] != hashlib.sha256(npz.read_bytes()).hexdigest():
        raise SystemExit(f"{src}: made from {rec['ckpt']} (step {rec['ckpt_step']}), which is not {npz}")
    try:
        load_stored_tokens(src, npz, meta)              # the file, its record, the step, the shape
    except ValueError as e:
        raise SystemExit(str(e))
    return rec


def export_tokens(src: Path, out: Path, bridges_sha: str, rec: dict, backbone_name=None):
    """The stored tokens of a bridge, in the Hugging Face layout, with their record."""
    t = np.load(src)["tokens"]
    st = out / "tokens.safetensors"
    mx.save_safetensors(str(st), {"tokens": mx.array(t)})
    b = np.array(mx.load(str(st))["tokens"])
    assert b.dtype == t.dtype and b.shape == t.shape and np.array_equal(b, t), "tokens: round-trip differs"
    new = scrub_meta(rec, backbone_name)
    if isinstance(new.get("verdict"), dict):             # the verdict travels with the tokens
        new["verdict"] = {**new["verdict"], **{k: shown(new["verdict"][k]) for k in ("report", "criteria")
                                               if k in new["verdict"]}}
    # each file is named beside the hash that is its own: `ckpt` is the training checkpoint the
    # tokens were made from, `bridges` its export, which is the file they are published with
    new.update(ckpt=shown(rec["ckpt"]), bridges="bridges.safetensors", bridges_sha256=bridges_sha,
               source_tokens_sha256=rec["tokens_sha256"],
               tokens_sha256=hashlib.sha256(st.read_bytes()).hexdigest(),
               format="psilm2-constitution-tokens/v1",
               load_with="psilm.mlx.constitution.load_stored_stack(<this dir>/bridges.safetensors)")
    clean(new, "tokens.json")
    (out / "tokens.json").write_text(json.dumps(new, indent=1) + "\n")
    from psilm.mlx.constitution import load_stored_stack
    coupler, _, _ = load_stored_stack(out / "bridges.safetensors")
    assert np.array_equal(np.array(coupler.stored), t.astype(np.float32)), "the loaded tokens differ"
    return new


def clean(payload: dict, name: str):
    """Nothing of this machine in what an export writes: no home directory, no user name."""
    text = json.dumps(payload)
    for what in (str(Path.home()), f"/{Path.home().name}/"):
        if what in text:
            raise SystemExit(f"{name}: a local path survived ({what!r}); nothing of it was kept")


def export(run: Path, out: Path, backbone_name=None, tokens=None):
    npz, meta_f = run / "bridges.npz", run / "bridges.npz.meta"
    rec = check_tokens(Path(tokens), npz, json.loads(meta_f.read_text())) if tokens is not None else None
    z = np.load(npz)
    arrays = {k: z[k] for k in z.files}
    out.mkdir(parents=True, exist_ok=True)
    st = out / "bridges.safetensors"
    mx.save_safetensors(str(st), {k: mx.array(v) for k, v in arrays.items()})
    back = dict(mx.load(str(st)))
    assert set(back) == set(arrays), (sorted(set(back) ^ set(arrays)))
    for k, v in arrays.items():
        b = np.array(back[k])
        assert b.dtype == v.dtype and b.shape == v.shape and np.array_equal(b, v), f"{k}: round-trip differs"
    meta = scrub_meta(json.loads(meta_f.read_text()), backbone_name)
    clean(meta, "the meta")
    # Every reader in this project resolves the meta as "<checkpoint> + .meta",
    # so the same payload sits beside the safetensors under that name too.
    (out / "bridges.safetensors.meta").write_text(json.dumps(meta, indent=1) + "\n")
    n_params = int(sum(v.size for v in arrays.values()))
    kind = "constitution" if "const_model" in meta else "physics"
    cfg = {"format": f"psilm2-{kind}-bridge/v1",
           "bridges_class": ("psilm.mlx.constitution.ConstitutionBridgesMLX" if kind == "constitution"
                             else "psilm.mlx.bridges.PsiBridgesMLX"),
           "load_with": ("psilm.mlx.constitution.load_constitution_stack(<this dir>/bridges.safetensors, <partner dir>)"
                         if kind == "constitution" else
                         "eval/bench_guardrail.py --phys-ckpt <this dir>/bridges.safetensors, or psilm2.dual.load_dual_stack"),
           "source_run": shown(run), "step": meta.get("step"),
           "n_tensors": len(arrays), "n_params": n_params,
           "backbone": backbone_name or meta.get("model"),
           "sha256_safetensors": hashlib.sha256(st.read_bytes()).hexdigest(),
           "meta": meta}
    clean(cfg, "config.json")
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    if tokens is not None:
        if kind != "constitution":
            raise SystemExit("stored tokens belong to a constitution bridge")
        new = export_tokens(Path(tokens), out, cfg["sha256_safetensors"], rec, backbone_name)
        cfg["stored_tokens"] = {"file": "tokens.safetensors", "record": "tokens.json",
                                "sha256": new["tokens_sha256"], "kind": new["kind"], "prompts": new["n"],
                                "verdict": (new.get("verdict") or {}).get("verdict", "unverified"),
                                "load_with": new["load_with"]}
        clean(cfg, "config.json")
        (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return n_params, st.stat().st_size, cfg["sha256_safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--out")
    ap.add_argument("--all", action="store_true", help="every results/stage2c_* run into results/hf_export/psilm2/bridges/<backbone>/<variant>")
    ap.add_argument("--backbone-name", default=None)
    ap.add_argument("--tokens", default=None, help="the bridge's stored tokens (fixed_tokens_mean.npz), with --run")
    a = ap.parse_args()
    if a.tokens and a.all:
        ap.error("--tokens goes with one --run")
    jobs = []
    if a.all:
        for d in sorted(glob.glob("results/stage2c_*")):
            run = Path(d)
            if not (run / "bridges.npz").exists():
                continue
            tag = run.name[len("stage2c_"):]
            if "_" not in tag or not (run / "bridges.npz.meta").exists():
                print(f"skip {run.name}: not a <backbone>_<variant> constitution run with a meta")
                continue
            bk, var = tag.split("_", 1)
            if bk not in BACKBONE_DIR:
                # the release covers the paper's backbones; Bonsai's bridges (and its
                # smoke runs) belong to the chat app and are exported with --run/--out
                print(f"skip {run.name}: backbone {bk!r} is not part of the ΨLM-2 release")
                continue
            jobs.append((run, Path("results/hf_export/psilm2/bridges") / BACKBONE_DIR.get(bk, bk) / var,
                         a.backbone_name or BACKBONE_NAME.get(bk), None))
    else:
        jobs.append((Path(a.run), Path(a.out), a.backbone_name, a.tokens))
    for run, out, name, tokens in jobs:
        n, size, sha = export(run, out, name, tokens)
        print(f"{run.name:34s} -> {out}  {n/1e6:.2f}M params, {size/1e6:.1f} MB, sha256 {sha[:12]}...  round-trip OK"
              + ("; stored tokens beside it, loaded back with no partner" if tokens else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
