#!/usr/bin/env python3
"""Create the PRIVATE release repo and upload release/<dir> as-is (minus scratch).

Visibility is deliberately NOT handled here: the maintainer flips the repo
public by hand. Uses the ambient `hf auth login`; no credentials pass through.

  python results/hf_export/upload_release.py --dir release/qwen3.5-9b-psilm --repo ryoji-info/Qwen3.5-9B-PsiLM
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
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    print("repo ready (private):", args.repo, flush=True)
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            api.upload_folder(folder_path=args.dir, repo_id=args.repo, repo_type="model",
                              ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store", "smoke_run.log"],
                              commit_message="Qwen3.5-9B-PsiLM: bridges, FNO, inference script, card")
            print(f"uploaded in {(time.time()-t0)/60:.1f} min", flush=True)
            break
        except Exception as e:
            print(f"attempt {attempt} failed: {e!s:.200}", flush=True)
            time.sleep(30)
    info = api.repo_info(args.repo, repo_type="model")
    print("private:", info.private, "| files:", sorted(s.rfilename for s in info.siblings), flush=True)


if __name__ == "__main__":
    main()
