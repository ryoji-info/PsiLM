# MANIFEST — Hugging Face repo `ryoji-info/Gemma-4-12B-PsiLM`

Everything in this directory (`release/gemma-4-12b-psilm/`) is uploaded as-is to the
model repo; nothing outside it is needed. The backbone (`mlx-community/gemma-4-12B-it-4bit`)
is **not** uploaded — the script downloads it at first run.

## Files to upload (and where each one came from)

| path in the HF repo | source in the GitHub checkout | size | sha256 |
|---|---|---:|---|
| `README.md` | written for this release (the HF model card, YAML front matter) | 17 KB | — |
| `psilm_infer.py` | written for this release | 42 KB | — |
| `requirements.txt` | written for this release | 1 KB | — |
| `MANIFEST.md` | this file (harmless to upload; drop it if you prefer) | — | — |
| `psilm-banner.png` | `assets/psilm-banner.png` (1600 px wide) | 1.35 MB | `4cd65f32fe4aab5b49b66681a841b0128628e8d94c75494ae84f941b49c48089` |
| `bridges/gemma-4-12b-4bit-mlx-1d-value-selective/bridges.safetensors` | `results/hf_export/bridges/gemma-4-12b-4bit-mlx-1d-value-selective/bridges.safetensors` (= `results/stage2_gemma12b/bridges.npz`, step 7000, learned-pointer tensors dropped) | 102,068,660 B | `f6ef8946c41b3cfa17df7c22bcab2c5cffbd856c2dd608970a364cc14d5e8f7d` |
| `bridges/gemma-4-12b-4bit-mlx-1d-value-selective/config.json` | `results/hf_export/bridges/gemma-4-12b-4bit-mlx-1d-value-selective/config.json` | 1,287 B | `8a5add382beb65a0214f5e8b0b640e5d5df6f19e9a6caf1945400d2fe1e6bba5` |
| `physics/fno_burgers_singlemode.safetensors` | `results/hf_export/physics/fno_burgers_singlemode.safetensors` (= `results/stage2/fno.pt`; loaders verified identical, max weight and field difference 0.0) | 552,076 B | `7bb0076c85cdcf2505a9079c05964e3eb77ac4a776eccf34216953f3c37bfcdd` |

The copies were made with `cp` on 2026-09-06 and the hashes match the sources
(`shasum -a 256`). Not uploaded: `__pycache__/`, `.DS_Store`.

### To add later (the two runs in progress)

`psilm_infer.py` has the `--task {1d,multimode,2d}` switch (2026-09-06; the 1D path is
unchanged: on the 0.5B the previous script and this one print the same numbers and text,
`--question-only` byte-for-byte, full runs differing only in the timings). When the Gemma
multi-mode and 2D runs finish, export them and add:

| path in the HF repo | source in the GitHub checkout | size | sha256 |
|---|---|---:|---|
| `bridges/gemma-4-12b-4bit-mlx-multimode/bridges.safetensors` | the stage-2b Gemma checkpoint `results/stage2b_gemma12b_2b/bridges.npz` (see the export recipe below) | — | — |
| `bridges/gemma-4-12b-4bit-mlx-multimode/config.json` | written from `results/stage2b_gemma12b_2b/bridges.npz.meta` in the schema below | — | — |
| `physics/fno_burgers_multimode.safetensors` | `results/hf_export/physics/fno_burgers_multimode.safetensors` (= `results/stage2b/fno.pt`; loads with `load_fno_safetensors`, verified against the torch FNO1d to 4e-7 in the smoke below) | 552,076 B | `8cebd7a82d0f5da13ae1cfb74848d74d43b98b98cd09767655f6c4ac069f879b` |
| `bridges/gemma-4-12b-4bit-mlx-2d-dpot/bridges.safetensors` | the stage-2d Gemma checkpoint `results/stage2d_gemma12b_2d/bridges.npz` (tag as in `eval/mlx_stage2d_train.py`'s docstring) | — | — |
| `bridges/gemma-4-12b-4bit-mlx-2d-dpot/config.json` | written from its `bridges.npz.meta` in the schema below | — | — |
| `physics/dpot_tiny_fisher2d_finetuned.safetensors` | `results/hf_export/physics/dpot_tiny_fisher2d_finetuned.safetensors` (= `results/stage2d/dpot_ft.pt`, bare DPOT-Tiny state-dict keys; verified identical to the .pt in the smoke below) | 30,144,524 B | `88472c199fed489c05ac5ceaf9113c607dd28d95e45385f62e52fac6494b15d8` |
| `physics/model_Ti.pth` | `results/stage2d/model_Ti.pth` -- the upstream DPOT-Tiny base checkpoint (hzk17/DPOT, Apache-2.0). **Ship it, or rely on the download:** `psilm_infer.py --task 2d` looks for it at `--dpot-base` (default `physics/model_Ti.pth`) and, when absent, downloads `model_Ti.pth` from the Hugging Face repo `hzk17/DPOT` (the repo `psilm/physics/dpot_wrapper.py` names; the file is listed there, download verified 2026-09-06) into the HF cache. `DPOTPhysics` loads the base first and the fine-tuned safetensors then replaces every one of its 67 tensors (`strict=True`), so the base only satisfies the wrapper's constructor. | 90,475,962 B | `074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f` |

Then fill the two *in progress* rows of `README.md` ("Bridges in this repository") from their
`final_eval.json`, and add `einops>=0.8` to `requirements.txt` (the vendored DPOT definition
imports it; only the 2D task needs it).

**The 2D task needs a clone.** `psilm/physics/dpot_wrapper.py` imports `dpot_model` from
`<repo>/vendor/`, which the pip package (`[tool.setuptools.packages.find] include = ["psilm*"]`)
does not ship. `psilm_infer.py` adds `$PSILM_REPO/vendor` to `sys.path` when `PSILM_REPO`
points at a clone (even with the package installed) and exits with a message naming this
otherwise; the README's usage section says so. Packaging `vendor/dpot_model.py` inside `psilm`
would lift the requirement.

#### `config.json` schema (multimode and 2d)

`psilm_infer.py` requires every key below for `--task multimode` / `--task 2d` and exits naming
the missing one (e.g. `config.json: missing key(s) ['coupling.l_rev', 'physics']`). The 1D
directory keeps its existing, more lenient config (defaults inferred from the tensors). `task`
is also what the script auto-detects when `--bridges` is given without `--task` (falling back to
the directory name: `multimode`, `2d`/`dpot`).

```json
{
 "task": "multimode",                                   // "multimode" | "2d"  (required)
 "backbone": "mlx-community/gemma-4-12B-it-4bit",       // informational
 "hf_tokenizer": "mlx-community/gemma-4-12B-it-4bit",   // used for the chat template (else --backbone)
 "bridges_class": "psilm.mlx.multimode.make_bridges_multi (PsiBridgesMLX, n_params=6)",
                                                        // 2d: "psilm.mlx.bridges2d.PsiBridges2DMLX"
 "construct": {                                         // required, all five keys; passed verbatim to the
  "d_model": 3840,                                      //   constructor the trainer used:
  "channel": "value",                                   //   make_bridges_multi(**construct)  (eval/mlx_stage2b_train.py)
  "inj_cap": 0.2,                                       //   PsiBridges2DMLX(**construct)     (eval/mlx_stage2d_train.py)
  "gate_bias": 0.0,                                     //   = meta["args"]{channel, inj_cap, gate_bias, readout_norm}
  "readout_norm": "dim"
 },
 "coupling": {"l_fwd": 20, "l_rev": 30, "n_layers": 48},   // required; meta["l_fwd"], meta["l_rev"]; n_layers is checked
 "physics": {                                           // required; "file" is relative to the repo root and is the
  "file": "physics/fno_burgers_multimode.safetensors",  //   default for --physics
  "dpot_base": "physics/model_Ti.pth"                   //   2d only, informational (the CLI flag --dpot-base decides)
 },
 "training": {"...": "free-form record, as in the 1D config"},
 "license": "apache-2.0"
}
```

(JSON has no comments; the file itself carries none.) Values must match how the checkpoint was
trained: the script constructs the bridges from `construct`, then checks every tensor shape in
`bridges.safetensors` against the module and refuses on a mismatch or an unknown tensor; missing
tensors are allowed (`strict=False`) and reported as unused (the multimode export may drop the
retired learned pointer `fwd.x0_query`, `fwd.x0_key.*`, as the 1D export did; the 2D bridges
have no such tensors). `fwd.dim_mu` / `fwd.dim_sigma` (the `readout_norm: "dim"` calibration)
are ordinary tensors of the checkpoint and must be included.

#### Export recipe (maintainer, from the checkout root)

```python
import json, mlx.core as mx
from pathlib import Path
task, src, dst = "multimode", Path("results/stage2b_gemma12b_2b"), Path("release/gemma-4-12b-psilm/bridges/gemma-4-12b-4bit-mlx-multimode")
# task, src, dst = "2d", Path("results/stage2d_gemma12b_2d"), Path("release/gemma-4-12b-psilm/bridges/gemma-4-12b-4bit-mlx-2d-dpot")
meta = json.loads((src / "bridges.npz.meta").read_text()); a = meta["args"]
w = {k: v for k, v in mx.load(str(src / "bridges.npz")).items()
     if not k.startswith(("fwd.x0_query", "fwd.x0_key."))}          # unused learned pointer (multimode only)
dst.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(dst / "bridges.safetensors"), w)
cfg = {"task": task, "backbone": meta["model"], "hf_tokenizer": a["hf_tokenizer"],
       "bridges_class": {"multimode": "psilm.mlx.multimode.make_bridges_multi (PsiBridgesMLX, n_params=6)",
                         "2d": "psilm.mlx.bridges2d.PsiBridges2DMLX"}[task],
       "construct": {"d_model": 3840, "channel": a["channel"], "inj_cap": a["inj_cap"],
                     "gate_bias": a["gate_bias"], "readout_norm": a["readout_norm"]},
       "coupling": {"l_fwd": meta["l_fwd"], "l_rev": meta["l_rev"], "n_layers": 48},
       "physics": {"multimode": {"file": "physics/fno_burgers_multimode.safetensors"},
                   "2d": {"file": "physics/dpot_tiny_fisher2d_finetuned.safetensors", "dpot_base": "physics/model_Ti.pth"}}[task],
       "training": {"step": meta["step"], "args": a}, "license": "apache-2.0"}
(dst / "config.json").write_text(json.dumps(cfg, indent=1))
```

Then `python psilm_infer.py --task multimode` / `--task 2d` from `release/gemma-4-12b-psilm`
must print the three sections with a `PsiLM match` verdict on the defaults before upload.

#### Smoke record (2026-09-06, mechanics only)

Run with randomly initialised bridges of the right shapes for `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
(d_model 896, 24 layers, coupling 10/15), built with the trainers' own constructors
(`make_bridges_multi(896, gate_bias=0.0, inj_cap=0.2, channel="value", readout_norm="dim")`,
`PsiBridges2DMLX(d_model=896, ...)` with the same arguments), saved as `bridges.safetensors` +
`config.json` in the schema above, with the physics files from `results/hf_export/physics/`:

- `--task multimode` (auto-detected from the directory name in one run), single-mode default and
  the two-mode combination, with and without the baseline arm: three sections printed; the
  `[3] physics model (FNO)` value agreed with an independent torch computation
  (`psilm.stage2.bridges.build_ic_multi` + `psilm.physics.fno.FNO1d` loaded from
  `results/stage2b/fno.pt` + `fourier_interp`) to 5e-8 (single) and 4e-7 (combo).
- `--task 2d` on MPS with `--dpot-base results/stage2d/model_Ti.pth`, and on CPU with the base
  absent (downloaded from `hzk17/DPOT`): three sections printed; `[3] physics model (DPOT)`
  agreed with the torch stage-2d path (`DPOTPhysics` from `model_Ti.pth` + `results/stage2d/dpot_ft.pt`,
  `psilm.stage2d.bridges2d.build_ic_2d`, `features_and_field`, `fisher2d.bilinear_periodic`, CPU)
  to 7e-7 (MPS) and 0.0 (CPU).
- Error paths: a config missing `coupling.l_rev` and `physics` exits naming both keys; `--task 2d`
  on a multimode directory exits on the `task` mismatch; malformed `--modes` exits.
- The coupled answers were garbage, as expected from random bridges (the backbone's own reply
  template, no `equals`), which the script reports as "reply left the trained template".

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
