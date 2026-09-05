# MANIFEST — Hugging Face repo `ryoji-info/Gemma-4-12B-PsiLM`

Everything in this directory (`release/gemma-4-12b-psilm/`) is uploaded as-is to the
model repo; nothing outside it is needed. The backbone (`mlx-community/gemma-4-12B-it-4bit`)
is **not** uploaded — the script downloads it at first run.

## Files to upload (and where each one came from)

| path in the HF repo | source in the GitHub checkout | size | sha256 |
|---|---|---:|---|
| `README.md` | written for this release (the HF model card, YAML front matter) | 15 KB | — |
| `psilm_infer.py` | written for this release | 18 KB | — |
| `requirements.txt` | written for this release | 1 KB | — |
| `MANIFEST.md` | this file (harmless to upload; drop it if you prefer) | — | — |
| `psilm-banner.png` | `assets/psilm-banner.png` (1600 px wide) | 1.35 MB | `4cd65f32fe4aab5b49b66681a841b0128628e8d94c75494ae84f941b49c48089` |
| `bridges/gemma-4-12b-4bit-mlx-1d-value-selective/bridges.safetensors` | `results/hf_export/bridges/gemma-4-12b-4bit-mlx-1d-value-selective/bridges.safetensors` (= `results/stage2_gemma12b/bridges.npz`, step 7000, learned-pointer tensors dropped) | 102,068,660 B | `f6ef8946c41b3cfa17df7c22bcab2c5cffbd856c2dd608970a364cc14d5e8f7d` |
| `bridges/gemma-4-12b-4bit-mlx-1d-value-selective/config.json` | `results/hf_export/bridges/gemma-4-12b-4bit-mlx-1d-value-selective/config.json` | 1,287 B | `8a5add382beb65a0214f5e8b0b640e5d5df6f19e9a6caf1945400d2fe1e6bba5` |
| `physics/fno_burgers_singlemode.safetensors` | `results/hf_export/physics/fno_burgers_singlemode.safetensors` (= `results/stage2/fno.pt`; loaders verified identical, max weight and field difference 0.0) | 552,076 B | `7bb0076c85cdcf2505a9079c05964e3eb77ac4a776eccf34216953f3c37bfcdd` |

The copies were made with `cp` on 2026-09-06 and the hashes match the sources
(`shasum -a 256`). Not uploaded: `__pycache__/`, `.DS_Store`.

### To add later (the two runs in progress)

When the Gemma multi-mode and 2D runs finish, export them the same way and add:

| path in the HF repo | source |
|---|---|
| `bridges/gemma-4-12b-4bit-mlx-multimode/{bridges.safetensors,config.json}` | the stage-2b Gemma checkpoint (`results/stage2b_gemma12b*/bridges.npz` + a `config.json` in the format of the 1D one) |
| `physics/fno_burgers_multimode.safetensors` | `results/hf_export/physics/fno_burgers_multimode.safetensors` (loads with `load_fno_safetensors`, verified) |
| `bridges/gemma-4-12b-4bit-mlx-2d-dpot/{bridges.safetensors,config.json}` | the stage-2d Gemma checkpoint |
| `physics/dpot_tiny_fisher2d_finetuned.safetensors` | `results/hf_export/physics/dpot_tiny_fisher2d_finetuned.safetensors` |

and fill the two *in progress* rows of `README.md` ("Bridges in this repository") from their
`final_eval.json`; `psilm_infer.py` will need a `--task` switch for those (it is 1D-only today).

## Upload (maintainer runs this)

From the GitHub checkout root, with `HF_TOKEN` set (or `huggingface-cli login` done):

```bash
cd /Users/rxiii/Documents/GitHub/PsiLM
HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
repo = "ryoji-info/Gemma-4-12B-PsiLM"
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    folder_path="release/gemma-4-12b-psilm",
    repo_id=repo,
    repo_type="model",
    commit_message="Gemma-4-12B-PsiLM: bridges, Burgers FNO, one-command CLI, model card",
    ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
)
print("UPLOAD DONE")
EOF
```

`HF_HUB_DISABLE_XET=1` keeps the upload on the classic LFS path (the Xet backend has
stalled on this machine's earlier uploads). Make the repo public afterwards with
`api.update_repo_settings(repo, private=False)` or from the repo's settings page, together
with `ryoji-info/PsiLM-bridges` and `ryoji-info/PsiLM-physics`, which the card links to.

## After upload: the one-command check

```bash
huggingface-cli download ryoji-info/Gemma-4-12B-PsiLM --local-dir /tmp/g4psilm && cd /tmp/g4psilm
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python psilm_infer.py
```

Note for the `pip install` line: the `psilm @ git+...` requirement needs the GitHub
`pyproject.toml` to use package discovery (`[tool.setuptools.packages.find] include = ["psilm*"]`,
changed on 2026-09-06 in the working tree from `packages = ["psilm"]`, which shipped only the
top-level package without `psilm.mlx`, `psilm.stage2`, `psilm.physics`). That change must be
pushed before the requirement installs a usable package; until then the script says so, and the
`PSILM_REPO=/path/to/clone` fallback works regardless.
