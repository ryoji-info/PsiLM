#!/usr/bin/env python3
"""Resumable upload of the converted backbone (multi-part, retried, resumes from
its own .cache/huggingface bookkeeping inside the folder). Same ambient login and
same private-first policy as upload_backbone.py; use this when a plain
upload_folder of the 8 GB file drops mid-way.

  python results/hf_export/upload_backbone_large.py \
      --dir /Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx --repo ryoji-info/Qwen3.5-9B-nvfp4-mlx
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
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    t0 = time.time()
    api.upload_large_folder(repo_id=args.repo, folder_path=args.dir, repo_type="model",
                            num_workers=args.workers, print_report_every=60)
    print(f"upload_large_folder finished in {(time.time()-t0)/60:.1f} min", flush=True)
    info = api.repo_info(args.repo, repo_type="model")
    print("private:", info.private, "| files:", sorted(s.rfilename for s in info.siblings), flush=True)


if __name__ == "__main__":
    main()
