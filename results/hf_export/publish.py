#!/usr/bin/env python3
"""Upload the current cards to the three (private) repos.

Visibility is deliberately NOT handled here: the maintainer flips the repos
public by hand once the cards read right live. Uses the ambient
`hf auth login`; no credentials pass through here.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# the cache (and the login token with it) was relocated on 2026-09-10; a
# non-interactive shell does not read ~/.zshrc, so point at it explicitly
_HOME = "/Users/rxiii/Documents/huggingface"
if os.path.isfile(os.path.join(_HOME, "token")):
    os.environ.setdefault("HF_HOME", _HOME)
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
    for _, repo in CARDS:
        print("still private:", repo, "->", api.repo_info(repo, repo_type="model").private)


if __name__ == "__main__":
    main()
