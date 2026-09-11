#!/usr/bin/env python3
"""Upload a locally converted backbone to a PRIVATE Hub repo, with its card.

Visibility is deliberately NOT handled here: the maintainer flips the repo
public by hand once the card reads right live. Uses the ambient
`hf auth login`; no credentials pass through here.

  python results/hf_export/upload_backbone.py \
      --dir /Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx \
      --repo ryoji-info/Qwen3.5-9B-nvfp4-mlx \
      --card results/hf_export/backbones/qwen3.5-9b-nvfp4-mlx/README.md
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_HOME = "/Users/rxiii/Documents/huggingface"
if os.path.isfile(os.path.join(_HOME, "token")):
    os.environ.setdefault("HF_HOME", _HOME)
from huggingface_hub import HfApi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--card", required=True)
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    print("repo ready (private):", args.repo, flush=True)
    t0 = time.time()
    api.upload_folder(folder_path=args.dir, repo_id=args.repo, repo_type="model",
                      ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
                      commit_message="Qwen3.5 9B text tower, NVFP4, reassembled from Ollama qwen3.5:9b-mlx")
    print(f"weights uploaded in {(time.time()-t0)/60:.1f} min", flush=True)
    api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md", repo_id=args.repo,
                    repo_type="model", commit_message="Card")
    info = api.repo_info(args.repo, repo_type="model")
    print("card uploaded; still private:", info.private, "| files:", len(info.siblings), flush=True)


if __name__ == "__main__":
    main()
