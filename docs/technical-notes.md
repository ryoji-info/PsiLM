# PsiLM technical notes

The stage-by-stage experimental record behind the [README](../README.md): every
narrative, table and reproduce command from the development campaign, moved
here verbatim so the README can stay short. Numbers are read from the committed
result files under `results/`; the paper (`paper/psilm.pdf`) tells the same
story with the analysis.

## Context: where PsiLM sits in the literature

As of August 2026, no published work couples a pretrained LLM to a
neural-operator physics model through a bidirectional latent channel at
inference. The nearest neighbors are the Bicameral Model
([arXiv:2605.11167](https://arxiv.org/abs/2605.11167), two frozen LLMs coupled
through hidden states — no code released), the Global Latent Workspace line
([shimmer](https://github.com/ruflab/shimmer)), and CALM
([arXiv:2401.02412](https://arxiv.org/abs/2401.02412)). PsiLM builds toward
that gap, on consumer hardware first (Apple Silicon, 24 GB unified memory).

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
| Qwen2.5-3B-Instruct-4bit | 38.3% | **66.7%** | **+28.3** | 100% |

Two findings, both matching the literature. **(1) The capability-gap effect is
real and local:** at 3B the simulator adds +28.3 points — the same magnitude
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
understates it badly; it reaches ~25–50%
exact on the shorter mixed-length products of the per-chunk rollout evaluations. Loss plateaued for the last 5k steps, so the remaining fidelity gap is
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

Stage 2b/2c: `python eval/stage2b_pretrain_fno.py`, then
`python eval/stage2b_train.py --steps 1000 --batch 16 --fresh` (×6; add `--v2`
for the refuted variant), then `python eval/stage2b_eval.py --n 48`.

Stage 2d: `python eval/stage2d_prepare.py`, then
`python eval/stage2d_train.py --steps 1000 --batch 12 --fresh` (×8), then
`python eval/stage2d_eval.py`.

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

### The same study at 12B (2026-09)

Gemma 4 12B, same recipe as the released 1D bridges, 7,500 steps
(`results/stage2b_gemma12b_2b/`). Held-out, n=48 per family:

| family | backbone | PsiLM | oracle | always 0.00 |
|---|---:|---:|---:|---:|
| in-distribution | 20.8% | **100%** (MAE 0.009) | 100% | 27.1% |
| held-out mode combination | 16.7% | 25.0% (MAE 0.123) | 100% | 16.7% |
| amplitude extrapolation | 10.4% | 52.1% (MAE 0.086) | 100% | 4.2% |

Twenty-four times the backbone of the 0.5B study saturates the training family
(97.9% → 100%) and leaves the other two where they were (31.3%, 50.0%). The
support-coverage law is scale-independent.

Attribution, this time without teacher forcing: running the frozen FNO on each
item's *true* parameters scores 100% on all three families (MAE 0.0002–0.0008),
so the physics is exact everywhere and every miss is the readout's. What it
does instead — on the combination family 19 of 48 answers match a *single*-mode
field value; on extrapolation the implied amplitude is below the true one for
71% of items, median ratio 0.68, inside mode 1's training range of 0.3–0.7.

The remedy the law names is coverage, not parameterization (v2's per-mode bins
were worse, above): read every mode's amplitude with the **same** head, so
mode 2's training range of 0.5–1.0 covers the 0.8–1.0 mode 1 is tested on, and
let deterministic token spans carry the structure the way they already carry
x0. `psilm/mlx/multimode_span.py`, `--readout span`.

That was tested at 0.5B before spending the 12B budget
(`eval/readout_transfer_probe.py`, records in `results/readout_transfer/`):
train only the readout, then score what the physics model would answer from
the parameters it produces. The pooled readout reproduces the rollout profile
(0.99 / 0.32 / 0.44), which is what makes the proxy usable. Held-out
combination family, layer 10, 2500 steps, three seeds per arm:

| readout | in-dist | combination | extrapolation |
|---|---:|---:|---:|
| pooled | 0.990 | 0.323 | 0.438 |
| span, shared heads | 1.000 | 0.608 ± 0.136 | 0.958–0.979 |
| + distractor pinned at 0.0 | 1.000 | 0.681 ± 0.054 | 0.969 |
| **+ distractor U(0.02, 0.25)** | **1.000** | **0.993 ± 0.012** | **0.979** |

Two coverages, bought separately. **Value** coverage comes from sharing the
amplitude head across slots — that alone takes extrapolation from 0.44 to
~0.97, with amplitude error 0.001–0.002 on a range the slot never saw.
**Structure** coverage has to come from the data: half the single-mode prompts
rewritten with the absent mode at a small amplitude (`with_second_term` in
`psilm/stage2/qa2.py`, answer recomputed by the solver). A distractor pinned
at exactly 0.0 is the control — it supplies structure with nothing to read in
it and gains almost nothing.

Two things that did *not* survive: a readout-depth effect (layer 5 gave 0.979
and 0.396 on two seeds of the same configuration — depth is noise), and the
framing of the family itself. With two-term prompts in training, the
combination family tests amplitude range in a two-term context, not
compositional structure. It is still a generalization test, and a weaker one.
The 12B run under this recipe is queued behind the 2D task.

## Loop coupling: more paths, trained end-to-end

`psilm/stage2/loop_model.py` implements two read→rollout→inject passes
ordered so the second readout sees the stream *after* the first injection
and can revise it (shared bridges, all passes supervised). At matched
96k-sample budget on the multi-mode task, against the single-pass baseline
(n=48/family, PsiLM arm):

| family | 1-pass | loop-trained | inference-only loop |
|---|---:|---:|---:|
| in-distribution | 97.9% / 0.009 | **100%** / 0.007 | 8.3% / 0.323 |
| unseen combination | 31.2% / 0.128 | **47.9%** / 0.089 | 0.0% / 0.868 |
| amplitude extrapolation | 50.0% / 0.096 | 47.9% / **0.055** | 6.2% / 0.416 |

**The revision loop buys compositional generalization**: +16.7 points on the
held-out combination family, with MAE improved on every family. Amplitude
extrapolation stays flat in accuracy (support coverage still rules) though
its error magnitude halves. The third arm — single-pass-trained bridges
simply *run* as two passes at inference — collapses: turning the loop on at
run time without training it is catastrophic, not neutral (with the caveat
that this arm shifts both the coupling depths and the untrained revision
behavior at once, so it bounds the zero-shot transplant, not the revision
effect in isolation). Loops must be trained in; they then pay off exactly
where single-pass interfaces were weakest.

Reproduce: `python eval/stage2b_train.py --steps 1000 --batch 16 --fresh
--loop 2` (×6), then `python eval/stage2b_eval.py --n 48 --loop 2 --ckpt
results/stage2b_loop2/bridges.pt`.

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
| LLM alone | 6.7% (text protocol; forced to answer: 0%, MAE 0.89; four shots + thinking: 3.3%) | 0.337 |
| LLM + answer stated in text (oracle ceiling) | 100% | 0.0024 |
| **PsiLM (latent coupling, DPOT-Tiny)** | **95.0%** | **0.0168** |
| degenerate always-0.00 | 1.7% | 0.673 |

The 1D result survives the move to 2D and to a real pretrained physics
foundation model: the coupled system reaches within five points of the
oracle-text ceiling with nothing but hidden states crossing the interface.

### The same task at 12B (2026-09)

Gemma 4 12B, the released recipe, 7,500 steps (`results/stage2d_gemma12b_2d/`).
Held out, n=60:

| arm | accuracy | MAE |
|---|---:|---:|
| Gemma alone | 10.0% | 0.290 |
| **PsiLM** | **100%** | **0.0096** |
| oracle (value written into the prompt) | 96.7% | 0.021 |
| always 0.00 | 1.7% | 0.673 |

The only arm in this work where the latent channel beats the text ceiling: the
oracle copies a number out of the prompt and sometimes mis-rounds it, while the
bridge reads the field exactly and the answer never passes through text. Per
chunk, coupled 0.50 → 0.94, then no-harm 0.979 / 0.958 / 1.00.

Hybrid-stack gotcha worth knowing: MLX and torch share one unified GPU memory
and MLX's cached buffers are invisible to torch's MPS allocator. The no-harm
arm died asking for 256 bytes with 42 GiB in "other allocations";
`mx.clear_cache()` at the boundary was not enough, so DPOT-Tiny runs on the CPU
(65 ms against 48 ms per batch-4 call, against a 3.6 s training step).

## Scaling the language hemisphere — MLX, 1.7B, 8B

The bridges are parameterized by the backbone's config alone (coupling depths
as fractions of depth, widths from the hidden size), and `psilm/mlx/` ports
the staged forward, the FNO and the bridges to MLX so 4-bit backbones train
on 24 GB. Same physics model, same task, same 60 held-out questions:

| backbone | LLM alone | **PsiLM** | oracle (answer in text) | bridges |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B (fp16, torch) | 8.3% / 0.682 | **100%** / 0.014 | 100% / 0.003 | 3.5M |
| Qwen3-1.7B (fp16, torch) | 1.7% / 2.57 | **93.3%** / 0.022 | 96.7% / 0.021 | 12.6M |
| Qwen3-8B-4bit (MLX) | 6.7% / 0.706 (forced: 0% / 0.89; strengthened: 3.3%) | **98.3%** / 0.0135 | 100% / 0.0026 | 28.4M |

The 8B took eight runs, and the paper's Section 9 reports them as a
scale-dependent failure analysis. Two things broke at 4096 dimensions, and
neither was the frozen model's willingness to be steered:

- **The learned attention pointer never trains at 8B.** A readout probe with
  the identical head shows learned pooling stuck at uniform cross-entropy for
  2000 steps (0% exact bins) while pooling over the known x₀ token span
  reaches 83% / error 0.005. The 8B readout therefore pools deterministically
  over the QA builder's span — a stated concession: the pointer is supplied
  by the task, not learned from the words.
- **A channel carrying the whole field is shortcut.** With the pointer and
  lookup verified exact inside the full pipeline (run 6), the 8B still
  collapsed to per-trajectory constants: it read the amplitude and ignored
  the one x₀-dependent token among seventeen. Replacing the channel with the
  8B copy probe's form — the looked-up value through Fourier features into
  eight soft tokens, nothing else — took the same checkpoint from 17% to
  89.6% in one 500-step chunk and to 98.3% at the end.

Trainer lessons that came out of it and now live in `eval/mlx_stage2_train.py`:
persist the optimizer across chunks with bias correction (MLX's AdamW defaults
to none; each fresh start kicked the gate), detach the pointer on its way into
the reverse bridge, warm up the readout before the channel opens, cap the
injection magnitude, and log the gate at the answer positions rather than
averaged over the prompt.

Reproduce (Qwen3-8B, ~12 h): `python eval/mlx_stage2_train.py --model
mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B --tag _mlx8b8
--steps 500 --batch 8 --readout-only 1000 --detach-x0 --clip module --lam-x0 1.0
--channel value --l-rev 22 --inj-cap 0.2 --gate-bias 0.0 --eval-n 48` (×7,
first with `--fresh`; the committed checkpoint instead kept run 6's step-2000
readouts and resumed them with `--reinit-channel`, five chunks to step 4500), then `python eval/mlx_stage2_eval.py --model
mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B --tag _mlx8b8 --n 60`.
Trained bridges for every backbone are on Hugging Face as safetensors
(`ryoji-info/PsiLM-bridges`).

## A second family: Gemma 4 12B

The finished recipe (deterministic span pointer, value-token channel,
readout warm-up, no-harm arm) was run once, as a single pass, on
`mlx-community/gemma-4-12B-it-4bit` (48 layers, hidden 3840, sliding +
global attention, 262k tied vocabulary, logit soft-cap). The staged forward
is bit-exact through `psilm/mlx/gemma_loader.py`; a batch-4 training step
takes 12 s at a 13 GB peak.

| backbone | LLM alone | **PsiLM** | oracle (answer in text) | bridges |
|---|---:|---:|---:|---:|
| Gemma 4 12B (4-bit, MLX) | 0% (never reaches an Answer line in 768 tokens) | **96.7% / MAE 0.017** | 98.3% / 0.007 | 25.5M |

**One backbone-specific fix, measured not tuned.** The first pass plateaued
at 40% because the pointer would not sharpen: Gemma's massive-activation
dimensions are nearly constant across prompts, so the per-position RMS
normalization of the bridge input divides the digit signal by them, and the
pooled x₀-span vector has 26× less across-item variance than on Qwen.
`--readout-norm dim` standardizes each dimension with statistics from a
32-prompt calibration pass (two frozen vectors saved with the bridges),
restores the signal to twice Qwen's, and the warm-up then ends with the
sharpest pointer of any backbone (CE 1.24, 69% exact bins).

**Guard-rail on Gemma** (n=100 per dataset; `results/bench/gemma12b_*`):
physics 0 / 97 / 10% (backbone / PsiLM / zeroed; gate 0.14, open on 100%),
GSM8K 84 / 84 / 84% (gate 0.004, open on 0%), MMLU@256 53 / 55 / 53%
(79.1 / 79.1% on the 67 items both arms answer), GSM8K without the "Answer:"
line 83 / 83 / 83%. In the training logs Gemma's gate sits near 0.15–0.19
on physics batches with the injection at 3–4% of the stream, a much gentler
operating point than Qwen's saturated gate at the 20% cap; both are selective.

## Guard-rail: does the coupled model still do everything else?

`eval/bench_guardrail.py` runs the same backbone on GSM8K, a five-subject
MMLU slice and the physics set (100 each) in three arms — backbone alone,
PsiLM, and PsiLM with the injection zeroed — and records the gate per
question. The first run on the 98.3% checkpoint (v8) found the gate **open on
every prompt** (σ 0.98–0.99): physics tokens computed from a math word
problem were injected at full strength and GSM8K fell from 89% to 34%. The
zeroed arm equalled the backbone byte for byte, so the damage was the
injection alone. The gate had only ever seen physics prompts.

v9 adds a **no-harm arm**: 1,049 non-physics prompts (GSM8K train, MMLU
validation; with and without the "Answer:" line; benchmark test items
excluded) paired with the backbone's own continuation, alternated with
physics batches; on those steps only the gate is updated. The gate closed
within fifty such steps:

| dataset (n=100) | backbone | PsiLM v8 | **PsiLM v9** | zeroed | gate v8 → v9 |
|---|---:|---:|---:|---:|---:|
| physics QA | 5% | 97% | **95%** | 0% | 0.99 → 0.79 (open on 100%) |
| GSM8K | 89% | 34% | **89%** | 89% | 0.98 → 0.002 (open on 0%) |
| GSM8K, no "Answer:" line | 79% | – | **79%** | 79% | 0.001 (open on 0%) |
| MMLU, 5 subjects (256 tokens) | 60% | – | **61%** | 60% | 0.99 → 0.003 (open on 0%) |

One gate MLP, conditioned on the residual stream, is open on every physics
question and shut on every other prompt — with or without the "Answer:"
line, so it is not keying on the template — and the bridges are otherwise
the backbone to the byte. (The physics backbone figure is from the v9 run's
768-token budget; the v8 run's backbone arm scored 2% at 160 tokens.) (The v8 MMLU number is omitted: at the original 24-token budget
it was a parse artifact, 55% → 69% only because the injection forced terse
answers.)

Reproduce: `python eval/build_noharm.py` (twice, `--nudge-prob 1.0` and
`--nudge-prob 0.0 --out data/noharm_train_nonudge.json`, the two lists
concatenated into `data/noharm_train_all.json`), then copy the v8 checkpoint
(`bridges.npz`, `bridges.npz.meta`, `opt.npz`) into `results/stage2_mlx8b9/`
and resume it with the training command above plus `--tag _mlx8b9 --noharm-data
data/noharm_train_all.json --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0`
for 500 steps, then `python eval/bench_guardrail.py --tag v9_8b --ckpt
results/stage2_mlx8b9/bridges.npz --n 100 --max-new-mmlu 256
--max-new-physics-base 768` and the same with `--tag v9_8b_nonudge --datasets
gsm8k --gsm8k-nudge 0`.

## Leaky gate: does an always-on physics signal help? (2026-09-09)

A gate that is shut off-task carries nothing there. The question was whether a
weak always-on signal would act as a regularizer or improve general reasoning.

**Setup.** A floor applied to the gate **at inference only**, on the released
run-9 bridges (`results/stage2_mlx8b9/bridges.npz`, step 5000, coupling 15/22 of
36), nothing retrained:

```
sigma_eff = eps + (1 - eps) * sigma          # psilm/mlx/bridges.py, gate_floor
```

Monotone in sigma, and it leaves the open end alone — `max(eps, sigma)` would
flatten the gate's own decision everywhere below the floor and has a dead zone
with no gradient, which would fight the no-harm arm if this were ever trained.
The logged sigma stays **pre-floor**, so the gate's decision remains observable;
that column is the instrument check.

Off-task the floor is nearly the whole gate, on-task nearly nothing. At
eps = 0.2 it moves sigma_eff from 0.0016 to 0.201 on GSM8K (×123) but only from
0.791 to 0.833 at physics prompt positions.

```bash
python eval/bench_guardrail.py --tag leaky_8b --n 100 \
    --arms base,psilm,leaky0.01,leaky0.05,leaky0.1,leaky0.2 \
    --datasets physics,mmlu,gsm8k,boolq --kl --tasks-cache results/bench/tasks_leaky_n100.json
python eval/leaky_report.py results/bench/leaky_8b_guardrail.json
```

400 questions × 6 arms, seed 0, greedy, 3.3 hours on the M2.
`results/bench/leaky_8b_guardrail_summary.json`.

**Results.**

| arm | physics | MMLU | GSM8K | BoolQ | physics MAE | KL(base‖arm) physics |
|---|---:|---:|---:|---:|---:|---:|
| backbone alone | 2%¹ | 60% | 89% | 88% | 2.239 | — |
| trained gate | 95% | 61% | 89% | 88% | 0.0215 | 0.110 |
| + floor 0.01 | 95% | 64% | 89% | 88% | 0.0213 | 0.116 |
| + floor 0.05 | 96% | 63% | 88% | 88% | 0.0214 | 0.139 |
| + floor 0.1 | 96% | 66% | 89% | 88% | 0.0204 | 0.168 |
| + floor 0.2 | 98% | 66% | 88% | 87% | 0.0182 | 0.222 |

¹ Not comparable with the 5% in the guard-rail table above: this sweep gives the
backbone's nudge protocol a 160-token budget, at which every reply truncates,
rather than 768.

No comparison among the coupled arms, and none between the backbone and any arm
off-task, reaches significance (McNemar p ≥ 0.0625; tightest is MMLU trained
gate vs eps=0.1). The only significant contrasts are backbone vs every coupled
arm on physics, as they should be. **The test bounds rather than establishes**:
at n=100 the smallest resolvable difference is six discordant items in one
direction, so these 36 paired comparisons show no effect exceeds ~6 items per
dataset.

**The MMLU rise is a token budget, not reasoning.** It is the one off-task
column that moves, and it moves with three others:

| arm | MMLU acc | parse rate | EOS rate | mean tokens |
|---|---:|---:|---:|---:|
| base | 0.600 | 0.720 | 0.710 | 87.2 |
| + floor 0.1 | 0.660 | 0.810 | 0.800 | 67.5 |
| + floor 0.2 | 0.660 | 0.840 | 0.820 | 62.0 |

Shorter replies finish inside the 256-token budget and get scored. Direct test:
of the 28 items the backbone leaves unparsed, eps=0.2 parses 13 and loses 1 —
all 13 in college physics and mathematics, the two subjects carrying the whole
movement — and 7 are right (54%), against the 75% the backbone scores on items
it already parses in those subjects. **Items scored, not items solved.**

On the 71 questions *every* arm answered, accuracy is **83.1% in all six arms**
including the backbone — though not item-identical: each coupled arm loses one
and gains one. Read that control carefully: 60 of the 71 are philosophy,
biology and foreign policy, where every arm emits 4 tokens and parses, and only
9 college-physics and 2 mathematics items survive into it. It shows the floor
changes nothing where length was never binding; it does not settle the items
where it was. GSM8K and BoolQ close that from the other side — parse rate 1.000
in every arm, same brevity on GSM8K (147 → 137 tokens), accuracy flat.

BoolQ replies are 4 tokens in **every** arm including the backbone, too short
for the mechanism to act at all, so its flatness is a control rather than a
confirmation.

**One real dose-response, on-task and continuous.** Physics MAE falls
0.0215 → 0.0182, and per item eps=0.2 is nearer the truth than the trained gate
on **30 items against 12** (58 tied; two-sided sign test **p = 0.008**). The
±0.05 tolerance hides it — accuracy moves 3 items, p = 0.25. The readout is
identical in every arm, so this is not better physics: the frozen model renders
the injected value more faithfully when the channel pushes harder, and at the
trained gate it under-commits by a few hundredths.

**Conclusion.** A weak always-on physics signal is not a regularizer. It is a
verbosity reducer that pays only where a budget binds — worth knowing for any
budget-limited benchmark, worth not claiming as a reasoning gain. The claim is
about *inference*: these bridges were trained with a zero floor and a no-harm
arm penalizing exactly the gate this sweep forces open, so nothing ever asked
the channel to carry something useful off-task. A system **co-trained** with an
always-on channel is untested. Measured on one backbone and one checkpoint.

**Dose arithmetic.** `inj_cap` limits the injection to 0.2 of the stream RMS
*before* the gate scales it, so eps=0.2 delivers **at most** 4% of the residual
stream — a ceiling, not a measurement, since this sweep logged the gate but not
the realized ratio (`return_ratio` in `psilm/mlx/bridges.py` exists for that).
That is a fifth of run 8's operating point, where a gate open at the cap cost 55
points on GSM8K. So the sweep bounds the channel as harmless across its range
without locating where that ends; eps ∈ {0.3, 0.4, 0.5}
(`results/bench/leaky_sweep_hi.sh`, same task cache and base continuations, so
the two merge into one curve) walks toward it.

**Method notes worth reusing.** The physics tokens are computed once from the
prompt and re-injected unchanged at every position, so the always-on signal is
one constant vector per question, not a per-token readout. On a non-physics
prompt the value it carries is a Burgers field value for whatever initial
condition the readout extracted from a word problem — a number, not a fact
about the question. The KL is teacher-forced on the base arm's own greedy
continuation so every arm is scored on the same tokens; off-task both arms share
the prompt, but on physics the base arm uses the nudge protocol, so there it is
a common-token comparison rather than a same-prompt one.

## Repository layout

```
psilm/                 core package (pip install -e .)
  simulator.py, llm.py, arms.py      Stage 0: closed-form simulator + tool-loop arms
  bicameral/           Stage 1: staged twin-LLM forward, gated interface, calculator task
  physics/             Burgers spectral solver, 1D FNO, 2D Fisher-KPP solver, DPOT wrapper
  stage2/              torch PsiLM: QA builders (qa.py single-mode, qa2.py multi-mode), bridges, model, loop model
  stage2d/             torch 2D PsiLM: QA builder, 2D bridges, model
  mlx/                 the MLX stack used for 4-bit backbones:
    staged.py            layer-by-layer frozen forward (MlxStream)
    bridges.py           forward bridge (span pointer, calibrated readout), value channel, gated injection
    model.py             PsiLMMLX: coupling, losses, readout-only and no-harm phases, generation
    multimode.py         multi-mode task on the same stack
    bridges2d.py, model2d.py, physics2d.py   2D task (language in MLX, DPOT-Tiny in torch)
    fno.py               FNO in MLX (+ loaders from .pt and safetensors)
    gemma_loader.py      Gemma 4 text tower in the MlxStream layout; load_backbone_any()
    vlm_loader.py        Qwen3.8-27B language tower (inference-feasible only on 24 GB)
    moe_patch.py         autograd patch for MoE routing indices
eval/                  training, evaluation and benchmark scripts
  mlx_stage2_{train,eval}.py, mlx_stage2b_*, mlx_stage2d_*   MLX trainers and four-arm evals
  stage1_*, stage2_*, stage2b_*, stage2d_*                    torch counterparts
  bench_guardrail.py, bench_common.py, build_noharm.py        gate-selectivity benchmark and its negatives
  copy_probe.py, readout_probe.py, readout_variance_probe.py  the diagnostic probes of the 8B/Gemma campaigns
  mlx_8b_setup.py, mlx_27b_setup.py, mlx_gemma_setup.py       parity/memory smoke tests per backbone
  export_bridges.py                                           checkpoint -> HF layout (safetensors + config.json)
data/                  QA datasets (single-mode, multi-mode families, 2D) and the no-harm negatives
results/               logs, evaluations, benchmark summaries, HF export staging (weights are git-ignored)
paper/                 the manuscript (psilm.tex, psilm.pdf, figures)
docs/                  this file
release/               the standalone Gemma-4-12B-PsiLM package as uploaded to Hugging Face
assets/                logo
vendor/                DPOT model definition
```

## Publishing a trained bridge

`eval/export_bridges.py` turns a run directory into the two files the release
inference script reads:

```bash
python eval/export_bridges.py \
    --run results/stage2b_gemma12b_2b \
    --out results/hf_export/bridges/gemma-4-12b-4bit-mlx-multimode-value-selective
```

It writes `bridges.safetensors` (the retired learned-pointer tensors
`fwd.x0_query`/`fwd.x0_key.*` dropped, since the span pointer is deterministic;
`--keep-unused` keeps them) and a `config.json` recording the backbone, the
coupling depths, the construction arguments needed to rebuild `PsiBridgesMLX`,
the phase split, and the held-out accuracy of every chunk. The phases are told
apart by the `_noharm` suffix on the kept checkpoints, so a no-harm phase
resumed from an earlier step does not steal a coupled chunk's score.

Weights never enter git (see `.gitignore`); `results/hf_export/` is only the
staging area from which the Hugging Face repositories are uploaded.
