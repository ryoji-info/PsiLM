---
license: apache-2.0
tags:
  - psilm
  - neural-operator
  - fno
  - dpot
  - pde
  - physics
  - burgers
  - fisher-kpp
  - safetensors
  - research
---

<!-- Maintainer: upload assets/psilm-banner.png from the GitHub repository to the root of this HF repo so the image below resolves. -->
![PsiLM banner](psilm-banner.png)

# PsiLM physics hemispheres

**The three frozen physics models that [PsiLM](https://github.com/ryoji-info/PsiLM) (ΨLM) couples to a language model.**

## What is PsiLM, and what is in this repo?

PsiLM couples a **frozen** language model to a **frozen** physics model through small trainable *latent bridges*; no text crosses the interface. The language side reads the physics model's inputs out of the prompt's hidden states, the physics model evolves the field, and the value at the queried position returns as eight soft tokens through a gated cross-attention layer. This repo contains the **physics side only**: two 70K-parameter 1D Burgers FNOs trained here from spectral-solver data, and a DPOT-Tiny checkpoint fine-tuned for 2D Fisher–KPP. They are ordinary neural operators and can be used without PsiLM. The trained bridges live in [`ryoji-info/PsiLM-bridges`](https://huggingface.co/ryoji-info/PsiLM-bridges); the language backbones are public checkpoints loaded separately.

**Easiest entry point:** the standalone repo [`ryoji-info/Gemma-4-12B-PsiLM`](https://huggingface.co/ryoji-info/Gemma-4-12B-PsiLM) packages the Gemma 4 12B bridges together with the physics model and a runnable example.

Links: [GitHub](https://github.com/ryoji-info/PsiLM) · [paper (PDF)](https://github.com/ryoji-info/PsiLM/blob/main/paper/psilm.pdf) · bridges: [`ryoji-info/PsiLM-bridges`](https://huggingface.co/ryoji-info/PsiLM-bridges)

## What the physics hemisphere costs, and what it buys

These files are the frozen half that does the computing — 0.07M parameters for
the Burgers FNO, 7.5M for DPOT-Tiny — sitting beside a 12.28B language model
that is also frozen. Measured on one Apple M2 (24 GB).

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

Sources: `results/bench/gemma12b_guardrail_summary.json` (accuracy, gate σ and seconds per question), `results/stage2_gemma12b/final_eval.json` (n=60 held-out: PsiLM 96.7%, oracle 98.3%). Parameter counts are read from the checkpoint headers, not from the model names. The backbone's 0% is its own text protocol: it spends the whole 768-token budget deriving and never commits to an answer line; forcing it to answer (n=8 probe) also gives 0%, with its predictions clustered at 0.41/0.51 while the truths span [-0.56, +0.58]. The latency gap has the same cause — PsiLM answers in one line.

## Which file do I want?

| file | model | task | val relL2 |
|---|---|---|---|
| `fno_burgers_singlemode.safetensors` | FNO1d (70K params) | 1D Burgers u0→u(0.5), ν=0.02, single-mode ICs | 0.28% |
| `fno_burgers_multimode.safetensors` | FNO1d (70K params) | 1D Burgers, broad multi-mode ICs | 2.76% |
| `dpot_tiny_fisher2d_finetuned.safetensors` | DPOT-Tiny (7.5M), fine-tuned 1200 steps | 2D Fisher–KPP replicated-IC → u(0.4) | 1.64% |

Validation relative-L2 errors are recorded in `results/physics_models.json` in the GitHub repository; the training data facts below come from the committed training scripts (`eval/stage2_pretrain_fno.py`, `eval/stage2b_pretrain_fno.py`, `eval/stage2d_prepare.py`).

| file | architecture | training data | size on disk | used by these bridges (in `PsiLM-bridges`) |
|---|---|---|---|---|
| `fno_burgers_singlemode.safetensors` | FNO1d, width 32, 16 Fourier modes, 4 layers (70K params); trained from scratch | 4096 train + 256 val trajectories from the repository's spectral Burgers solver (128-point periodic grid, ν=0.02, T=0.5), single-sinusoid ICs `a·sin(2πx+φ)`; 1500 Adam steps, batch 64 | 552 KB | `qwen2.5-0.5b-1d`, `qwen2.5-0.5b-4bit-mlx-1d`, `qwen3-1.7b-1d`, `qwen3-8b-4bit-mlx-1d-value`, `qwen3-8b-4bit-mlx-1d-value-selective`, `gemma-4-12b-4bit-mlx-1d-value-selective` |
| `fno_burgers_multimode.safetensors` | same FNO1d architecture (70K params); trained from scratch | 6144 train + 256 val trajectories, same solver; random Fourier ICs with 1–3 active modes drawn from modes 1–4 and mixed amplitudes, deliberately broader than any QA family; 2500 Adam steps, batch 64 | 552 KB | `qwen2.5-0.5b-multimode`, `qwen2.5-0.5b-multimode-v2-refuted`, `qwen2.5-0.5b-loop2`; Gemma multi-mode run *(in progress)* |
| `dpot_tiny_fisher2d_finetuned.safetensors` | DPOT-Tiny AFNO operator (7.5M params; embed 512, 4 blocks, patch 8, 128×128 input, 10-step history), from the published `model_Ti.pth` | 3072 train + 256 val fields from the repository's spectral Fisher–KPP solver (`u_t = D∇²u + r·u(1−u)`, D=0.001, r=6, periodic unit square, 128×128, T=0.4), Gaussian-bump ICs replicated across the 10-step history; 1200 AdamW steps, batch 8 | 30.1 MB | `qwen2.5-0.5b-2d-dpot`; Gemma 2D run *(in progress)* |

## Loading

All files are safetensors. The FNO spectral weights are complex64, which
safetensors does not support, so they are stored as `<key>.real` /
`<key>.imag` pairs; reassemble with
`sd[k] = torch.complex(f[k + ".real"], f[k + ".imag"])` before
`FNO1d.load_state_dict`. DPOT-Tiny derives from
[hzk17/DPOT](https://huggingface.co/hzk17/DPOT) (Apache-2.0). The FNOs load with `psilm.physics.fno.FNO1d` (PyTorch) or
`psilm.mlx.fno.convert_from_torch` (MLX, numerically identical to 1e-7).

Clone the GitHub repository and `pip install -e .` first; the model classes live in the `psilm` package.

```python
import torch
from safetensors.torch import load_file
from psilm.physics.fno import FNO1d

f = load_file("fno_burgers_singlemode.safetensors")
sd = {}
for k, v in f.items():
    if k.endswith(".real"):
        sd[k[:-5]] = torch.complex(v, f[k[:-5] + ".imag"])
    elif not k.endswith(".imag"):
        sd[k] = v
fno = FNO1d(width=32, modes=16, layers=4)
fno.load_state_dict(sd)
fno.eval()
u_T = fno(u0)  # u0: (batch, 128) initial condition on the periodic unit interval -> u(x, 0.5)
```

For DPOT-Tiny, construct `psilm.physics.dpot_wrapper.DPOTPhysics` (which builds the DPOT-Tiny network and expects the published `model_Ti.pth` from hzk17/DPOT to initialize) and then overwrite its weights with the fine-tuned state: `phys.net.load_state_dict(load_file("dpot_tiny_fisher2d_finetuned.safetensors"), strict=True)`. The keys are the bare DPOT-Tiny state-dict keys (`blocks.*`, `pos_embed`, `time_agg_layer.*`, ...). `phys(u0)` maps a `(batch, 128, 128)` initial field to `u(x, y, 0.4)`; `phys.features_and_field(u0)` also returns the 256 latent patch tokens the PsiLM reverse bridge attends over.

## Limitations

- **One PDE setting each.** The Burgers FNOs are trained at a single viscosity (ν=0.02), horizon (T=0.5) and 128-point resolution; the multimode model covers ICs up to mode 4 only. DPOT-Tiny is fine-tuned for one Fisher–KPP parameter pair (D=0.001, r=6) and horizon (T=0.4) with Gaussian-bump ICs.
- **One-shot operators.** Each model maps the initial condition straight to the field at the final time; they are not time-steppers and were not validated at other horizons.
- **Small validation sets.** The relL2 figures are on 256 held-out trajectories/fields from the same generator as the training data; no out-of-distribution physics evaluation was run.
- **DPOT provenance.** The fine-tuned DPOT-Tiny weights are derived from hzk17/DPOT (Apache-2.0); of DPOT's 4 input channels only the first carries the field; the other three are zero in this task.

## Beyond physics

These files are the *physics* seat of PsiLM, but the recipe (read a fixed set of quantities from text; let a frozen quantitative model compute; return one value through a selective gate) is not specific to PDEs. A calibrated market or event-probability model in the same seat would be the same architecture, and the appeal is the same: a language model's forecast grounded in a model that can be validated separately, with a gate that stays shut when the model does not apply. Nothing in this repository has been trained or tested on financial data; the physics results relied on exact oracles, deterministic targets and no distribution shift, none of which markets provide. This is a research direction, not a capability, and not a basis for investment decisions.

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

*Research generated by Claude Fable 5 (Anthropic) under the direction of
Ryoji Furui; see the repository's AI generation disclosure.* License: Apache-2.0 (DPOT-Tiny upstream: Apache-2.0).
