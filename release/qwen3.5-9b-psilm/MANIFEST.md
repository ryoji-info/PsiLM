# MANIFEST — Hugging Face repo `ryoji-info/Qwen3.5-9B-PsiLM`

Everything in this directory (`release/qwen3.5-9b-psilm/`) is uploaded as-is to the
model repo, and the backbone is uploaded to the **root of the same repo** from
`/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx` (the Qwen3.5 9B text tower, NVFP4,
reassembled from Ollama's `qwen3.5:9b-mlx` by `eval/ollama_to_mlx.py`), so the repo is
both the PsiLM release and a loadable `mlx-lm` model. `psilm_infer.py` uses the directory
it sits in as the backbone when `config.json` is beside it, else the Hub id.

## Files to upload (and where each one came from)

| path in the HF repo | source in the GitHub checkout | size | sha256 |
|---|---|---:|---|
| `README.md` | written for this release (the HF model card, YAML front matter) | 23,569 B | `a619d59a6dac0645caed60c40b15300422e0a30a6641f906fedf1d8e6099816d` |
| `psilm_infer.py` | `release/gemma-4-12b-psilm/psilm_infer.py` with the default backbone and the 1d bridges directory changed (docstring, `DEFAULT_BACKBONE`, `TASKS["1d"]`); the multimode/2d code paths are untouched and unused here | 43,401 B | `d9e047e2e460ffb73aa298ca26763827511072f2b28e1723bd2da3a94e72e161` |
| `requirements.txt` | written for this release (`mlx>=0.32.2` for native NVFP4; `mlx-lm==0.31.3` for `qwen3_5.py`) | 1,011 B | `9f14b111f7058a42724181601f38d2e04d96073d23786297148a98085d118266` |
| `MANIFEST.md` | this file (harmless to upload; drop it if you prefer) | — | — |
| `psilm-banner.png` | `assets/psilm-banner.png` (1600 px wide) | 1,352,073 B | `4cd65f32fe4aab5b49b66681a841b0128628e8d94c75494ae84f941b49c48089` |
| `bridges/qwen3.5-9b-nvfp4-mlx-1d-value-selective/bridges.safetensors` | `results/hf_export/bridges/qwen3.5-9b-nvfp4-mlx-1d-value-selective/bridges.safetensors` (= `results/stage2_qwen35/bridges.npz`, step 11,500 — the end of the no-harm phase; learned-pointer tensors dropped; 28.39M params in 41 fp32 tensors) | 113,579,339 B | `a45e3b4aa1bebff75c672d64fbd665668802e4774d1d59777df89386e4c80762` |
| `bridges/qwen3.5-9b-nvfp4-mlx-1d-value-selective/config.json` | `results/hf_export/bridges/qwen3.5-9b-nvfp4-mlx-1d-value-selective/config.json` (`eval/export_bridges.py --backbone-name ryoji-info/Qwen3.5-9B-PsiLM`; coupling 13/26 of 32; per-chunk held-out scores, n=16 except the first coupled chunk at n=48) | 1,514 B | `e0cd84435be1465295141c276c1d1da27ed06fc5a5e8c9c8ee7e380a698ce1ae` |
| `physics/fno_burgers_singlemode.safetensors` | `results/hf_export/physics/fno_burgers_singlemode.safetensors` (= `results/stage2/fno.pt`; identical to the file in the Gemma release) | 552,076 B | `7bb0076c85cdcf2505a9079c05964e3eb77ac4a776eccf34216953f3c37bfcdd` |

| `config.json` | `/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx/config.json` (the converted backbone; not in the GitHub checkout — weights never enter git) | 2,189 B | `31426fc674d7e58150cac520a4e4cb320fc051d697e3bafb82640801b43ddb30` |
| `model.safetensors` | `/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx/model.safetensors` (the converted backbone; not in the GitHub checkout — weights never enter git) | 7,971,382,795 B | `89c1eb2f0901a5e3fd2bcd43fffbea15752fed62fcecd5a046777a1e7682eba7` |
| `tokenizer.json` | `/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx/tokenizer.json` (the converted backbone; not in the GitHub checkout — weights never enter git) | 12,807,982 B | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | `/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx/tokenizer_config.json` (the converted backbone; not in the GitHub checkout — weights never enter git) | 16,710 B | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| `vocab.json` | `/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx/vocab.json` (the converted backbone; not in the GitHub checkout — weights never enter git) | 6,722,759 B | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `LICENSE` | `/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx/LICENSE` (the converted backbone; not in the GitHub checkout — weights never enter git) | 11,357 B | `5f3a3c817e78f5b8a4ad2d2c458a3e4b2cce470d6c12642c4eeb12cb8a9bf51d` |

The copies were made with `cp` on 2026-09-11 and the hashes match the sources
(`shasum -a 256`). Not uploaded: `__pycache__/`, `.DS_Store`, the release smoke-run log (kept at `results/qwen35/release_smoke_run.log`), and the converted directory's `preprocessor_config.json` / `video_preprocessor_config.json` (vision-side files the text tower does not use).

## Records the card cites

| claim on the card | record |
|---|---|
| held-out 100% / MAE 0.0147, oracle 98.3% / 0.100, backbone 3.3% / 0.570, zero 1.7% | `results/stage2_qwen35/final_eval.json` (kernel path) |
| same 60 answers on the training numerics (58 identical, 2 within 0.01) | `results/stage2_qwen35/final_eval_ops.json` (`--ops-path`) |
| parity 0.0, padded-row parity 0.0, ops-vs-kernel 1.4e-2 | `results/qwen35/setup_summary.txt` (`eval/mlx_qwen35_setup.py`) |
| coupling-depth probes (16.0 GB at 26, 19.1 GB at 24, 23.5 GB at 20) | `results/qwen35/probes.txt` |
| training trajectory, gate on negatives 0.001 / physics 0.94 | `results/stage2_qwen35/train_log.jsonl`, `supervisor.log`, `results/qwen35/coupled.log` |
| guard-rail table | `results/bench/guardrail_qwen35_summary.json` (fill in when the run completes) |
