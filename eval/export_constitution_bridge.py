#!/usr/bin/env python3
"""Export a bridge checkpoint (constitution or physics) to the Hugging Face layout.

Reads ``<run>/bridges.npz`` + ``bridges.npz.meta`` and writes
``<out>/bridges.safetensors`` (every tensor, unchanged, fp32 as trained) and
``<out>/config.json`` (the meta, plus the export's provenance: source run,
step, tensor count, parameter count, sha256 of the safetensors). Then reads
the safetensors back with ``mx.load`` -- the same call
``psilm.mlx.constitution.load_constitution_bridge_weights`` makes -- and
refuses to finish unless every array is bit-identical to the npz.

  python3 eval/export_constitution_bridge.py --run results/stage2c_qwen35_all \
      --out results/hf_export/psilm2/bridges/qwen3.5-9b/all
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
    hf_tokenizer and args.model) becomes the published backbone name, and any
    other path under the exporting user's home directory is written with `~`.
    Nothing a loader reads (shapes, mask, cap, const_model, step) changes."""
    home = str(Path.home())
    local_backbone = meta.get("model")

    def fix(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                fix(v)
            elif isinstance(v, str):
                if backbone_name and local_backbone and v == local_backbone:
                    d[k] = backbone_name
                elif v.startswith(home + "/"):
                    d[k] = "~" + v[len(home):]
    out = json.loads(json.dumps(meta))
    fix(out)
    return out


def export(run: Path, out: Path, backbone_name=None):
    npz, meta_f = run / "bridges.npz", run / "bridges.npz.meta"
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
    assert str(Path.home()) not in json.dumps(meta), "a local path survived the scrub"
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
           "source_run": str(run), "step": meta.get("step"),
           "n_tensors": len(arrays), "n_params": n_params,
           "backbone": backbone_name or meta.get("model"),
           "sha256_safetensors": hashlib.sha256(st.read_bytes()).hexdigest(),
           "meta": meta}
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return n_params, st.stat().st_size, cfg["sha256_safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--out")
    ap.add_argument("--all", action="store_true", help="every results/stage2c_* run into results/hf_export/psilm2/bridges/<backbone>/<variant>")
    ap.add_argument("--backbone-name", default=None)
    a = ap.parse_args()
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
                         a.backbone_name or BACKBONE_NAME.get(bk)))
    else:
        jobs.append((Path(a.run), Path(a.out), a.backbone_name))
    for run, out, name in jobs:
        n, size, sha = export(run, out, name)
        print(f"{run.name:34s} -> {out}  {n/1e6:.2f}M params, {size/1e6:.1f} MB, sha256 {sha[:12]}...  round-trip OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
