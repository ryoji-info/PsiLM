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
| **1** | Reproduce the Bicameral Model at 0.5B: two frozen Qwen2.5-0.5B twins, trainable gated hidden-state interface, PyTorch-MPS. | **mechanism reproduced — results below** |
| **2** | Swap the twin for a physics hemisphere: frozen LLM ⇄ frozen FNO through bidirectional latent bridges. | **works — results below** |
| **2b–2d** | Harden: multi-mode ICs, held-out families, 2D physics via pretrained DPOT-Tiny. | **done — results below** |
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

## Stage 1

A reproduction of the Bicameral Model (arXiv:2605.11167) — to our knowledge
the first public one. Two frozen Qwen2.5-0.5B-Instruct streams generate in
lockstep on Apple Silicon (PyTorch MPS, fp16 backbones), coupled by a 6.2M
trainable interface (`psilm/bicameral/`): forward coupling p→a at layer 10,
reverse a→p at layer 15, each an fp32 MLP translation network plus a
suppression gate that reads the receiver ("pull" design). The auxiliary
stream drives a calculator; tool output is forced into the auxiliary stream
only — the primary receives results purely through the hidden-state channel.
Trained with dual-target SFT on procedurally generated multiplication
(log-uniform operands, causality-aligned aux traces), 12,000 steps × batch 16
= 192k samples, ~3.5 s/step on an M2 24 GB.

**The paper's phase transition reproduces, in its causal order.** Forward
coupling strengthens first; exact tool recall then jumps 0.00 → 1.00 in one
chunk (~112k samples); answer accuracy onsets after. Teacher-forced
diagnostics at 96k samples: the auxiliary spells `calc(A*B)` at 92% token
accuracy reading the operands purely through the channel; the primary's
result digits reach 61.5% (chance 10%).

Held-out eval (n=40, operands ∈ [10³, 10⁵], products 7–10 digits, greedy):

| arm | exact answer | exact tool call | digit similarity |
|---|---:|---:|---:|
| Qwen2.5-0.5B alone | 0.0% | — | 0.371 |
| Bicameral (coupled) | 2.5% | **100%** | **0.689** |

The forward channel is essentially solved: every held-out rollout emits the
exact `calc(A*B)`. The reverse channel carries the leading 4–6 result digits
reliably and degrades toward the tail (`77298035 → 77299015`), so exact match
understates it badly; on shorter (6–7 digit) products it reaches ~38–50%
exact. Loss plateaued for the last 5k steps, so the remaining fidelity gap is
an optimization/capacity question — the paper's arithmetic configuration used
a 16M interface and its multiplication study ran to 320k samples. Two
reproduction lessons worth recording: (1) the first auxiliary window token is
predicted from the *uncoupled* prompt tail, which training can never
influence — free-running generation derails without a protocol-level
bootstrap (we force two initial wait tokens; the paper does not discuss
this); (2) coupling injects noise below the capability threshold, confirmed
independently in Stage 0.

Reproduce: `python eval/stage1_train.py --steps 1000 --batch 16` (chunked,
resumable), then `python eval/stage1_eval.py`.

## Stage 2 — PsiLM proper

The configuration the literature survey found unclaimed: a **frozen** LLM
(Qwen2.5-0.5B-Instruct) and a **frozen** physics model (a 70K-param FNO that
solves 1D viscous Burgers to 0.28% relative error in one shot) running as one
system, coupled through 3.5M parameters of trainable latent bridges
(`psilm/stage2/`) — **no text anywhere at the interface**:

- **forward (language→physics):** attention pooling over the LLM's layer-10
  prompt states regresses the initial-condition parameters and reads the
  query position x0 as a 100-way classification (softmax-expectation); a
  differentiable sinusoid feeds the FNO, so answer-loss gradients flow from
  the LLM's words back into the physics input.
- **reverse (physics→language):** K=16 learned queries compress the
  operator's latent field into soft tokens, plus a position-lookup token — a
  learnable-sharpness periodic kernel sampling the field at the x0 readout —
  injected into the primary at layer 15 behind a suppression gate.

Task: field-value QA — "u(x,0) = a·sin(2πx+φ) evolves by Burgers' equation
(ν=0.02) until t=0.5; what is u at x0?" — with spectral-solver ground truth,
a question the LLM cannot answer and the operator cannot read. Trained 5,000
steps × batch 16 (80k samples, ~40 min/1k steps on an M2 24 GB), answer
cross-entropy plus deep supervision on the parameter readouts and on u(x0)
at the lookup head.

Held-out results (n=60, tolerance ±0.05 on fields spanning ±0.6):

| arm | acc@0.05 | MAE |
|---|---:|---:|
| LLM alone | 5.0% | 0.729 |
| LLM + answer stated in text (oracle ceiling) | 100% | 0.0028 |
| **PsiLM (latent coupling)** | **100%** | **0.0138** |
| degenerate always-0.00 | 1.7% | 0.308 |

**The coupled system matches the oracle-text ceiling in accuracy**, with the
answer traveling entirely through hidden states: the LLM's question is read
out of its residual stream, the operator simulates, and the value returns
through gated cross-attention precisely enough for the frozen LLM to
verbalize it to ±0.014 on average. Accuracy went 0.08 → 0.50 → 1.00 across
the first three training chunks once the interface could *point*: the two
decisive design elements (found by failure analysis, chunk by chunk) were
reading x0 as a classification rather than a regression, and a position-
lookup token with direct deep supervision — plain cross-attention over field
summaries mode-collapsed to per-trajectory constants.

Reproduce: `python eval/stage2_pretrain_fno.py`, then
`python eval/stage2_train.py --steps 1000 --batch 16 --fresh` (×5), then
`python eval/stage2_eval.py`.

## Stage 2b/2c — harder ICs and generalization

Multi-mode extension (`psilm/stage2/qa2.py`): u(x,0) is a sum of sinusoids,
and the bridges train **only on single-mode questions** — {1} with
a∼U(0.3,0.7) or {2} with a∼U(0.5,1.0) (mode-2 amplitudes boosted to offset
its faster viscous decay; mode 3 was dropped after a calibration pass showed
its e^−3.55 decay degenerates 40% of its questions to "about zero"). The FNO
is trained on a deliberately broader distribution than every QA family, so
family-transfer failures are attributable to the interface.

In-distribution the multi-mode system converges as before (rollouts 1.00,
MAE 0.007 by 96k samples). The generalization families (n=48 each, tol
±0.05, zero-strategy shown for calibration):

| family | LLM alone | PsiLM | always-0.00 |
|---|---:|---:|---:|
| in-distribution | 4.2% | **97.9%** (MAE 0.009) | 27.1% |
| unseen combination {1,2} | 0.0% | **31.3%** (MAE 0.128) | 16.7% |
| amplitude extrapolation | 6.3% | **50.0%** (MAE 0.096) | 4.2% |

**Attribution** (teacher-forced readout error per family): the x0 pointer
generalizes perfectly — error 1e-4 on *all* families — while the amplitude
readout is the sole bottleneck (0.043 in-distribution → 0.355 on the unseen
combination, 0.206 under extrapolation). The language→physics readout, not
the physics→language readback, is where generalization dies.

**A tested and refuted hypothesis**, kept for the record: since the
*classified* x0 transferred and the *regressed* amplitudes did not, we built
a v2 interface (per-mode pooling queries, amplitudes as 121-bin
classifications). It was worse everywhere — extrapolation 50%→19%,
composition 31%→21%. The correct law is **support coverage**: x0
generalized because all 100 of its bins occur in training; amplitude bins
outside the trained range never receive gradient and cannot extrapolate,
where regression at least drifts. The trained interface generalizes like a
learned model, not like a program — coverage at training time, not readout
cleverness, is what buys transfer.

## Stage 2d — 2D physics with a pretrained foundation model

The physics hemisphere is no longer our own FNO: it is **DPOT-Tiny**
([hzk17/DPOT](https://huggingface.co/hzk17/DPOT), Apache-2.0) — a
7.5M-parameter AFNO operator pretrained across twelve PDE datasets — loaded
`strict=True` from the published checkpoint, fine-tuned for five minutes
(1200 steps) to 1.64% relL2 on our task, then frozen. Task: 2D Fisher-KPP
reaction-diffusion on the periodic unit square — a Gaussian bump (height,
center, width all stated in the question) grows and spreads by
u_t = D∇²u + r·u(1−u), and the question asks for u at a point (x₀, y₀)
(spectral-solver ground truth, dt-convergence 4e-11, exact logistic limit;
zero-strategy 4.8%). The IC is replicated across DPOT's 10-timestep input
history; the reverse bridge attends over DPOT's 256 latent patch tokens
plus a separable 2D periodic lookup on the predicted field; every
positional readout (bump center and query point) uses fully-covered
classification bins per the Stage-2b law.

Training showed a clean phase structure: the four positional classifiers
took ~3k steps to lock in (CE 4.6 → 0.03), and the answer loss converged
only after the 2D pointer did — rollouts 0.25 → 0.58 → **1.00** across 96k
samples. Held-out results (n=60, tolerance ±0.05, answers spanning the
front profile with mean u ≈ 0.62):

| arm | acc@0.05 | MAE |
|---|---:|---:|
| LLM alone | 6.7% | 0.337 |
| LLM + answer stated in text (oracle ceiling) | 100% | 0.0024 |
| **PsiLM (latent coupling, DPOT-Tiny)** | **95.0%** | **0.0168** |
| degenerate always-0.00 | 1.7% | 0.673 |

The 1D result survives the move to 2D and to a real pretrained physics
foundation model: the coupled system reaches within five points of the
oracle-text ceiling with nothing but hidden states crossing the interface.

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
