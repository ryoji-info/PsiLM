#!/usr/bin/env python3
"""Reassemble an Ollama MLX model into a directory mlx-lm can load.

Ollama stores an `-mlx` model as one single-tensor safetensors file per weight,
addressed by digest through a manifest, plus the tokenizer and config as
separate JSON blobs. mlx-lm wants a normal Hugging Face-style directory. This
walks the manifest and writes one.

Three things it has to fix along the way:

  * naming. Ollama writes the nvfp4 block scales as `<name>.weight.scale`;
    mlx.nn.QuantizedLinear expects `<name>.scales`.
  * the quantization stanza. The config Ollama ships has `quantization: null`
    even though the tensors carry `{"group_size": "16", "quant_type": "nvfp4"}`
    in their own metadata, so mlx-lm would build dense layers and fail to load
    the packed weights. We read the metadata and write the stanza.
  * the vision tower. PsiLM drives the text decoder only, and `--text-only`
    (the default) drops `vision_tower.*`, which is most of what is not needed.

  python eval/ollama_to_mlx.py qwen3.5:9b-mlx --out models/qwen3.5-9b-mlx
"""
import argparse
import json
import os
import shutil
import struct
import tempfile
from collections import Counter
from pathlib import Path

OLLAMA = Path.home() / ".ollama/models"


def manifest_path(tag: str) -> Path:
    name, _, version = tag.partition(":")
    return OLLAMA / "manifests/registry.ollama.ai/library" / name / (version or "latest")


def blob(digest: str) -> Path:
    return OLLAMA / "blobs" / digest.replace(":", "-")


def load_blob(digest: str, tmp: Path):
    """mx.load dispatches on the file extension and Ollama's blobs have none,
    so read each through a symlink rather than copying 9 GB to rename it."""
    import mlx.core as mx
    link = tmp / (digest.replace(":", "-") + ".safetensors")
    if not link.exists():
        os.symlink(blob(digest), link)
    return mx.load(str(link))


def read_one_tensor(path: Path):
    """(header dict, metadata dict, raw payload) of a single-tensor safetensors blob."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        payload = f.read()
    meta = header.pop("__metadata__", {})
    return header, meta, payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="Ollama tag, e.g. qwen3.5:9b-mlx")
    ap.add_argument("--out", required=True, help="directory to write")
    ap.add_argument("--keep-vision", action="store_true",
                    help="keep vision_tower.* (PsiLM does not need it)")
    args = ap.parse_args()

    mpath = manifest_path(args.tag)
    if not mpath.is_file():
        raise SystemExit(f"{mpath}: no such manifest (ollama list to see tags)")
    man = json.loads(mpath.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tensors, jsons = [], []
    for layer in man["layers"]:
        (tensors if layer["mediaType"].endswith("tensor") else jsons).append(layer)

    for layer in jsons:
        name = layer.get("name")
        if name and name.endswith(".json"):
            shutil.copyfile(blob(layer["digest"]), out / name)
            print(f"  {name}")

    import mlx.core as mx

    tmp = Path(tempfile.mkdtemp(prefix="ollama-mlx-"))
    weights, quant, dropped, kinds = {}, None, 0, Counter()
    for layer in tensors:
        name = layer["name"]
        if not args.keep_vision and name.startswith("vision_tower."):
            dropped += 1
            continue
        arrays = load_blob(layer["digest"], tmp)
        _, meta, _ = read_one_tensor(blob(layer["digest"]))
        if meta.get("quant_type"):
            quant = quant or {"group_size": int(meta["group_size"]), "bits": 4,
                              "mode": meta["quant_type"]}
            kinds[meta["quant_type"]] += 1
        for key, value in arrays.items():
            # <name>.weight.scale  ->  <name>.scales   (mlx.nn.QuantizedLinear)
            weights[key[: -len(".weight.scale")] + ".scales"
                    if key.endswith(".weight.scale") else key] = value

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(weights)} tensors kept, {dropped} vision tensors dropped, "
          f"quantized: {dict(kinds)}")
    mx.save_safetensors(str(out / "model.safetensors"), weights,
                        metadata={"format": "mlx"})

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if quant:
        cfg["quantization"] = quant
        cfg.setdefault("quantization_config", quant)
    if not args.keep_vision:
        cfg.pop("vision_config", None)
    cfg_path.write_text(json.dumps(cfg, indent=1))
    print(f"config.json: quantization = {quant}")

    size = (out / "model.safetensors").stat().st_size / 1e9
    print(f"\nwrote {out}  ({size:.1f} GB)")


if __name__ == "__main__":
    main()
