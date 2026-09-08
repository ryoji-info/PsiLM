#!/usr/bin/env python3
"""Final publish step: upload the current cards, then flip the three repos public.

Run only after every number on the cards is final. Uses the ambient
`hf auth login`; no credentials pass through here.
"""
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi

CARDS = [("results/hf_export/bridges/README.md", "ryoji-info/PsiLM-bridges"),
         ("results/hf_export/physics/README.md", "ryoji-info/PsiLM-physics"),
         ("release/gemma-4-12b-psilm/README.md", "ryoji-info/Gemma-4-12B-PsiLM")]


def main():
    api = HfApi()
    for path, repo in CARDS:
        api.upload_file(path_or_fileobj=path, path_in_repo="README.md", repo_id=repo,
                        repo_type="model", commit_message="Card: final numbers before publishing")
        print("card uploaded", repo)
    if "--public" not in sys.argv:
        print("cards uploaded; pass --public to flip visibility")
        return
    for _, repo in CARDS:
        api.update_repo_settings(repo_id=repo, repo_type="model", private=False)
        print("now public:", repo, "private =", api.repo_info(repo, repo_type="model").private)


if __name__ == "__main__":
    main()
