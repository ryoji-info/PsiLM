# ΨLM — PsiLM

**A language model coupled with a physics model: two hemispheres, one output.**

PsiLM pairs a pretrained language model with a pretrained physics model and runs
them *together* at inference, communicating while a single answer is produced —
the way the brain's hemispheres cooperate across the corpus callosum. The name
is the wave function Ψ: language as the observed, particle-like reading of a
continuous physical process evolved by the physics side (Bohr's
*complementarity* — two descriptions of one reality).

As of August 2026, no published work couples a pretrained LLM to a
neural-operator physics model through a bidirectional latent channel at
inference. The nearest neighbors are the Bicameral Model
([arXiv:2605.11167](https://arxiv.org/abs/2605.11167), two frozen LLMs coupled
through hidden states — no code released), the Global Latent Workspace line
([shimmer](https://github.com/ruflab/shimmer)), and CALM
([arXiv:2401.02412](https://arxiv.org/abs/2401.02412)). PsiLM builds toward
that gap, on consumer hardware first (Apple Silicon, 24 GB unified memory).

## Roadmap

| Stage | What | Status |
|---|---|---|
| **0** | Loop-level coupling: LLM extracts parameters → simulator computes → result returns to context (the Mind's Eye pattern, local). Establishes the eval harness and the baseline arms. | **done — first results below** |
| **1** | Reproduce the Bicameral Model at 0.5B: two frozen Qwen2.5-0.5B twins, trainable gated hidden-state interface, PyTorch-MPS. | next |
| **2** | Swap the twin for a physics hemisphere (DPOT / FNO / Poseidon): physics→language P-Former first, then bidirectional gated coupling. | planned |
| **3** | Port the loop to MLX; ship as a Mac app. | planned |

## Stage 0

A 60-item physics QA benchmark (`eval/generate_qa.py`, seeded) in the style of
UTOPIA (Mind's Eye, ICLR 2023): five rigid-body scene types — free fall,
projectile, friction slide, elastic collision, inclined plane — asked as
three-way comparisons (A / B / about the same), scored against a closed-form
simulator (`psilm/simulator.py`). A third of the items are physics traps whose
compared quantities are equal despite different surface parameters (mass in
free fall, complementary launch angles).

Two arms (`psilm/arms.py`):

- **alone** — the LLM answers directly.
- **tool** — the LLM extracts scene parameters as JSON, the simulator runs,
  and the numeric result is appended to the prompt before answering. Failed
  extractions fall back to the alone arm and are recorded
  (`tool_call_success`).

### Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python eval/generate_qa.py
.venv/bin/python eval/run_eval.py            # default: Qwen2.5-0.5B-Instruct-4bit
.venv/bin/python eval/run_eval.py --model mlx-community/Qwen3-4B-4bit
```

### First results

Apple M2, 24 GB, MLX, greedy decoding, 60 items, seed 0. Chance is ~33%.

| model | alone | + simulator (tool loop) | Δ | tool-call success |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct-4bit | 35.0% | 31.7% | −3.3 | 100% |
| Qwen2.5-3B-Instruct-4bit | 38.3% | **66.7%** | **+28.4** | 100% |

Two findings, both matching the literature. **(1) The capability-gap effect is
real and local:** at 3B the simulator adds +28.4 points — the same magnitude
as Mind's Eye's published +27.9 zero-shot average — while at 0.5B the same
pipeline *hurts* (−3.3): the small model extracts parameters perfectly (100%
tool success at both sizes) but cannot reliably compare two numbers handed to
it, echoing the Bicameral Model's finding that coupling injects noise when the
receiving model cannot exploit the channel. **(2) Extraction is not the
bottleneck; evidence-use is** — which is precisely the step a *trained*
interface (Stage 1) is supposed to absorb.

## Repository layout

```
psilm/            the package
  simulator.py    Stage-0 physics hemisphere (closed-form, deterministic)
  llm.py          language hemisphere (mlx-lm wrapper, greedy decoding)
  arms.py         alone / tool-loop evaluation arms
eval/
  generate_qa.py  seeded QA generation
  run_eval.py     runs arms, scores, writes results/
data/             generated QA sets (committed for reproducibility)
results/          eval outputs (committed as the experimental record)
```

## License

Apache-2.0.
