---
license: apache-2.0
tags:
  - psilm
  - latent-coupling
  - physics
  - neural-operator
  - pde
  - qwen
  - gemma
  - mlx
  - safetensors
  - research
---

<!-- Maintainer: upload assets/psilm-banner.png from the GitHub repository to the root of this HF repo so the image below resolves. -->
![PsiLM banner](psilm-banner.png)

# PsiLM bridges

**Trained bridge checkpoints for [PsiLM](https://github.com/ryoji-info/PsiLM) (ΨLM), one directory per (backbone, task) pair.**

## What is PsiLM, and what is in this repo?

PsiLM couples a **frozen** language model to a **frozen** physics model through small trainable *latent bridges*; no text crosses the interface. A forward bridge reads the physics model's inputs out of the prompt's hidden states (deterministic span pooling over each number's tokens plus 100-bin classifiers). The physics model (a 70K-parameter Burgers FNO, or DPOT-Tiny for 2D) runs on those inputs. A reverse channel turns the value at the queried position into Fourier features and eight soft tokens, injected back into the language model through gated cross-attention at a later layer; a no-harm training arm makes the gate open only on physics prompts. This repo holds **only the bridge weights**: backbones and physics models are loaded separately (see below), and bridges do not transfer between backbones.

**Easiest entry point:** the standalone repo [`ryoji-info/Gemma-4-12B-PsiLM`](https://huggingface.co/ryoji-info/Gemma-4-12B-PsiLM) packages the Gemma 4 12B bridges with the physics model and a runnable example. Use this repo when you want a specific backbone or task.

Links: [GitHub](https://github.com/ryoji-info/PsiLM) · [paper (PDF)](https://github.com/ryoji-info/PsiLM/blob/main/paper/psilm.pdf) · physics models: [`ryoji-info/PsiLM-physics`](https://huggingface.co/ryoji-info/PsiLM-physics)

## What the bridges cost, and what they buy

What the 25.5M trained parameters in one of these directories are worth, on the
largest backbone measured (Gemma 4 12B-it 4-bit, Apple M2 24 GB).

| component | parameters | on disk | trained? |
|---|---:|---:|---|
| Gemma 4 12B-it, 4-bit MLX (language tower) | 12.28B | 6.3 GB | **frozen** |
| PsiLM bridges, one (backbone, task) pair | **25.5M** | 102 MB | **trained** |
| Burgers FNO, the physics hemisphere | 0.07M | 0.55 MB | frozen, pretrained |
| DPOT-Tiny, 2D task only | 7.5M | 30 MB | frozen, fine-tuned |

The trained part is **0.21% of the backbone's parameters** and 1.6% of its
checkpoint size. The 12.28B never move.

| Gemma 4 12B, n = 100 per dataset | alone | **+ PsiLM** | injection zeroed | gate σ |
|---|---:|---:|---:|---:|
| physics QA | 0% | **97%** | 10% | 0.144 |
| MMLU, 5 subjects | 53% | 55% | 53% | 0.008 |
| GSM8K | 84% | 84% | 84% | 0.004 |
| physics, seconds per question | 77.0 (768 tokens) | **3.14** (16.9 tokens) | — | — |
| GSM8K, seconds per question | 25.07 | 25.06 | — | — |

**+0.21% parameters and +103 MB turn 0% into 97%, at 24× lower latency on the
task, with nothing measurable lost elsewhere.** The last column is why the
middle rows do not move: the gate's σ is 0.14 on physics against 0.004–0.008 on
everything else, so the channel is shut when physics is irrelevant. Zeroing the
injection collapses physics to 10% — the accuracy arrives through the bridge,
not through the prompt.

Sources: `results/bench/gemma12b_guardrail_summary.json` (accuracy, gate σ and seconds per question), `results/stage2_gemma12b/final_eval.json` (n=60 held-out: PsiLM 96.7%, oracle 98.3%). Parameter counts are read from the checkpoint headers, not from the model names. The backbone's 0% is its own text protocol: it spends the whole 768-token budget deriving and never commits to an answer line; forced to answer (n=60, `final_eval_baseline_forced.json`) it scores 6.7%, four near-constant guesses (0.41 / 0.51 / 0.54 / 0.11, sd 0.37 against truths spanning [-0.56, +0.60]) landing inside the tolerance; the best single constant would score 13.3%. The latency gap has the same cause — PsiLM answers in one line.

## Which directory do I want?

- **Best result on a large backbone with a selective gate:** `gemma-4-12b-4bit-mlx-1d-value-selective` or `qwen3-8b-4bit-mlx-1d-value-selective` (MLX, Apple Silicon).
- **Smallest working system (PyTorch or MLX, runs on a laptop in minutes):** `qwen2.5-0.5b-1d` or `qwen2.5-0.5b-4bit-mlx-1d`.
- **2D physics with a pretrained physics foundation model:** `qwen2.5-0.5b-2d-dpot`.
- **Generalization / loop-coupling studies:** `qwen2.5-0.5b-multimode`, `qwen2.5-0.5b-loop2` (and the refuted `-v2` kept for the record).
- **The Bicameral-Model reproduction (twin LLMs + calculator, no physics):** `qwen2.5-0.5b-bicameral`.

Held-out accuracy is within ±0.05 of the ground truth unless the row says otherwise. Every number below is copied from the evaluation records committed in the GitHub repository (`results/**/final_eval*.json`, `results/bench/*_summary.json`).

| directory | backbone | task | trained params | held-out result |
|---|---|---|---|---|
| `qwen2.5-0.5b-bicameral` | Qwen2.5-0.5B-Instruct ×2 (fp16) | Stage 1: twin-LLM + calculator (Bicameral reproduction) | 6.2M | 100% exact tool recall; digit-sim 0.689 vs 0.371 |
| `qwen2.5-0.5b-1d` | Qwen2.5-0.5B-Instruct (fp16) | 1D Burgers field QA (single-mode) | 3.5M | 100% @±0.05, MAE 0.0138 |
| `qwen2.5-0.5b-multimode` | Qwen2.5-0.5B-Instruct | 1D multi-mode + generalization study | 3.5M | iid 97.9%; unseen combination 31.3%; extrapolation 50.0% |
| `qwen2.5-0.5b-multimode-v2-refuted` | Qwen2.5-0.5B-Instruct | v2 interface (refuted hypothesis — kept for the record) | 3.8M | worse on every family |
| `qwen2.5-0.5b-loop2` | Qwen2.5-0.5B-Instruct | two-pass loop coupling | 3.5M | combination 47.9% (+16.7 over single-pass), iid 100% |
| `qwen2.5-0.5b-2d-dpot` | Qwen2.5-0.5B-Instruct | 2D Fisher–KPP with fine-tuned DPOT-Tiny | 3.7M | 95.0% @±0.05, MAE 0.0168 |
| `qwen3-1.7b-1d` | Qwen3-1.7B (fp16) | 1D Burgers field QA | 12.6M | 93.3% @±0.05, MAE 0.0221 |
| `qwen2.5-0.5b-4bit-mlx-1d` | Qwen2.5-0.5B-Instruct-4bit (MLX) | 1D Burgers field QA, MLX stack (step 5000; learned pointer, field channel, trained with the pre-readout-norm `psilm.mlx` code — loads with `strict=False` only and does not reproduce under the current code; kept for the record) | 3.5M | 1.00 rollouts, MAE 0.0136 (at training time) |
| `qwen3-8b-4bit-mlx-1d-value` | Qwen3-8B-4bit (MLX) | 1D Burgers field QA; deterministic span pointer, value-token channel (8 soft tokens from the looked-up u(x0)), inject @ layer 22/36, injection cap 0.2 | 28.4M | **98.3% @±0.05, MAE 0.0135** (n=60; oracle 100%; backbone alone 0% forced to answer, 3.3% with four solved examples, 1,536 tokens and thinking — `final_eval_baseline_{forced,strong}.json`) |
| `qwen3-8b-4bit-mlx-1d-value-selective` | Qwen3-8B-4bit (MLX) | as above + gate-selectivity training (no-harm arm, 500 steps) | 28.4M | physics **95%** (n=100) / 93.8% (n=48); **GSM8K 89% = backbone, MMLU 61% vs 60%**; gate 0.79 on physics, 0.002 elsewhere |
| `gemma-4-12b-4bit-mlx-1d-value-selective` | Gemma 4 12B-it-4bit (MLX) | 1D Burgers field QA; calibrated per-dimension readout (`readout_norm: dim`, buffers included), value-token channel, inject @ layer 30/48, cap 0.2, gate-selectivity training | 25.5M | **96.7% @±0.05, MAE 0.017** (n=60; oracle 98.3%); GSM8K 84% = backbone, MMLU 55% vs 53%; gate 0.14 on physics, 0.004 elsewhere |
| `gemma-4-12b-4bit-mlx-multimode` | Gemma 4 12B-it-4bit (MLX) | 1D multi-mode + generalization study (run tag `stage2b_gemma12b_2b`) | 25.5M | **iid 100%** @±0.05, MAE 0.009 (n=48; backbone 20.8%, oracle 100%); held-out mode combination 25.0%, amplitude extrapolation 52.1% |
| `gemma-4-12b-4bit-mlx-2d-dpot` | Gemma 4 12B-it-4bit (MLX) | 2D Fisher–KPP with fine-tuned DPOT-Tiny (run tag `stage2d_gemma12b_2d`) | 13.8M | **100%** @±0.05, MAE 0.0096 (n=60; backbone 10.0%, oracle 96.7% — PsiLM above the text ceiling) |

Which physics model each directory needs: the `-1d` / `-value` directories use `fno_burgers_singlemode`, the `-multimode`, `-v2-refuted` and `-loop2` directories use `fno_burgers_multimode`, and `-2d-dpot` uses `dpot_tiny_fisher2d_finetuned`, all from `ryoji-info/PsiLM-physics`. The bicameral interface uses no physics model. The backbones are the public checkpoints named in the table (`mlx-community/*` for MLX rows); the two 8B directories and the Gemma directory carry a `config.json` with the exact backbone id, coupling layers and training recipe.

## Loading

Clone the GitHub repository and `pip install -e .` first; the loaders live in the `psilm` package. Fetch one directory with `huggingface_hub.snapshot_download("ryoji-info/PsiLM-bridges", allow_patterns="<directory>/*")`.

### Loading the Qwen3-8B / Gemma 4 bridges (any of the three directories; Gemma loads through `psilm.mlx.gemma_loader.load_backbone_any`)

```python
import json, mlx.core as mx
from psilm.mlx.bridges import PsiBridgesMLX
cfg = json.load(open("qwen3-8b-4bit-mlx-1d-value/config.json"))
c = dict(cfg["construct"]); bridges = PsiBridgesMLX(**c)
bridges.load_weights("qwen3-8b-4bit-mlx-1d-value/bridges.safetensors", strict=False)  # the retired learned-pointer tensors are omitted
# Backbone (works for Qwen and Gemma 4; Gemma checkpoints are wrapped in a text tower):
import psilm.mlx.gemma_loader
tower, stock, tok = psilm.mlx.gemma_loader.load_backbone_any(cfg["backbone"])
```
Couple at layers 15 (read) / 22 (inject) of 36 and pass the QA builder's `x0_span` to the forward bridge (see `eval/mlx_stage2_eval.py`). For Gemma the coupling layers are 20 (read) / 30 (inject) of 48, as recorded in its `config.json`; the calibrated readout buffers (`fwd.dim_mu`, `fwd.dim_sigma`) are inside `bridges.safetensors`.

The PyTorch directories (`qwen2.5-0.5b-*`, `qwen3-1.7b-1d`) hold `bridges.safetensors` (or `interface.safetensors` for the bicameral run) in the `psilm.stage2` / `psilm.bicameral` state-dict layout; the matching evaluation scripts (`eval/stage2_eval.py`, `eval/stage2b_eval.py`, `eval/stage2d_eval.py`, `eval/stage1_eval.py`) show how each is instantiated. Training scripts, evaluation records and the paper are in the [GitHub repository](https://github.com/ryoji-info/PsiLM). All results are reproducible on a single Apple M2 (24 GB).

## Is the answer really coming through the channel?

Two controls, and the second is decisive. **Zeroing the injection** while running
everything else — readout, FNO, value tokens, gate — removes the physics result
(0% for Qwen3-8B, 10% for Gemma, which is what the reply template alone
recovers). **Corrupting only the number** — feeding the value encoder another
question's answer at matched magnitude, with prompt, readout, gate, reply length
and parsing untouched — makes the frozen model report the corruption: the spoken
answer lands within ±0.05 of the *injected* value on **99 of 100** held-out
questions and within ±0.05 of the truth on 9. Accuracy falls 98% → 9% while the
KL to the base model is unchanged (0.222 either way): the output distribution
travels just as far, to a different number.

Run on non-physics prompts the same swap changes nothing (GSM8K 0.88 both ways,
MMLU 0.66 both ways, p = 1.00), which separates what the channel does by its
**presence** from what it does by its **content**. Full sweep and records:
[`results/bench/leaky_8b_shuf_guardrail_summary.json`](https://github.com/ryoji-info/PsiLM/blob/main/results/bench) and §9.7 of the [paper](https://github.com/ryoji-info/PsiLM/blob/main/paper/psilm.pdf).

## Limitations

- **Narrow task.** Every bridge was trained and evaluated on one synthetic field-value question family (1D Burgers or 2D Fisher–KPP) with exact solver ground truth, in-distribution test sets, and greedy decoding. Held-out families in the multi-mode study drop to 31–50%.
- **Pointer supplied by the task at 8B and 12B.** The large-backbone readouts pool over the question builder's `x0` token span; the learned attention pointer did not train at 4096 dimensions (paper, Section 9). Prompts must follow the builder's template.
- **Backbone-specific.** Bridges are sized from the backbone config and do not transfer across backbones or quantizations.
- **Guard-rail is measured, not guaranteed.** Selectivity was checked on n=100 GSM8K / MMLU slices and the physics set; gate behaviour on other prompt types is untested. The non-selective `qwen3-8b-4bit-mlx-1d-value` directory loses 55 GSM8K points (89% → 34%) and is kept for the record.
- **Hardware.** MLX directories need Apple Silicon; the 8B and 12B rows were trained and evaluated on a 24 GB M2.

## Beyond physics

The bridges here couple a frozen language model to a frozen *physics* model, but the recipe (read a fixed set of quantities from text; let a frozen quantitative model compute; return one value through a selective gate) is not specific to PDEs. A calibrated market or event-probability model in the physics model's seat would be the same architecture, and the appeal is the same: a language model's forecast grounded in a model that can be validated separately, with a gate that stays shut when the model does not apply. Nothing in this repository has been trained or tested on financial data; the physics results relied on exact oracles, deterministic targets and no distribution shift, none of which markets provide. This is a research direction, not a capability, and not a basis for investment decisions.

## Support

PsiLM is independent research run on a single Apple M2. If it is useful to you, you can support the work at [ko-fi.com/ryojifurui](https://ko-fi.com/ryojifurui).

## Citation

```bibtex
@misc{furui2026psilm,
  title  = {PsiLM: Coupling Frozen Language and Physics Models through Trainable Latent Bridges},
  author = {Furui, Ryoji},
  year   = {2026},
  note   = {Research generated by Claude Fable 5 (Anthropic) under the author's direction},
  url    = {https://github.com/ryoji-info/PsiLM}
}
```

*Research generated by Claude Fable 5 (Anthropic) under the direction of Ryoji Furui; see the repository's AI generation disclosure.* License: Apache-2.0.
