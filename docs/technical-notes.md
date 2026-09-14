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

MLX-plus-torch gotcha worth knowing (the 2D stack runs both): MLX and torch share one unified GPU memory
and MLX's cached buffers are invisible to torch's MPS allocator. The no-harm
arm died asking for 256 bytes with 42 GiB in "other allocations";
`mx.clear_cache()` at the boundary was not enough, so DPOT-Tiny runs on the CPU
(65 ms against 48 ms per batch-4 call, against a 3.6 s training step).

## Scaling the language hemisphere — MLX, 1.7B, 8B

The bridges are parameterized by the backbone's config alone (coupling depths
as fractions of depth — except Qwen3.5, whose injection depth was set by a
measured memory cliff, below — widths from the hidden size), and `psilm/mlx/`
ports the staged forward, the FNO and the bridges to MLX so 4-bit and NVFP4
backbones train on 24 GB. Same physics model, same task, same 60 held-out questions:

| backbone | LLM alone | **PsiLM** | oracle (answer in text) | bridges |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B (fp16, torch) | 8.3% / 0.682 | **100%** / 0.014 | 100% / 0.003 | 3.5M |
| Qwen3-1.7B (fp16, torch) | 1.7% / 2.57 | **93.3%** / 0.022 | 96.7% / 0.021 | 12.6M |
| Qwen3-8B-4bit (MLX) | 6.7% / 0.706 (forced: 0% / 0.89; strengthened: 3.3%) | **98.3%** / 0.0135 | 100% / 0.0026 | 28.4M |
| Qwen3.5 9B-NVFP4 (MLX; mostly recurrent) | 3.3% / 0.570 (forced) | **100%** / 0.0147 | 98.3% / 0.100 (one forced item parsed wrongly) | 28.4M |

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
sharpest pointer of any backbone to that point (CE 1.24, 69% exact bins, the
mean of the last hundred steps; Qwen3.5's warm-up later reached CE 0.45 and
91% on the same statistic, on twice the warm-up samples — below).

**Guard-rail on Gemma** (n=100 per dataset; `results/bench/gemma12b_*`):
physics 0 / 97 / 10% (backbone / PsiLM / zeroed; gate 0.14, open on 100%),
GSM8K 84 / 84 / 84% (gate 0.004, open on 0%), MMLU@256 53 / 55 / 53%
(79.1 / 79.1% on the 67 items both arms answer), GSM8K without the "Answer:"
line 83 / 83 / 83%. In the training logs Gemma's gate sits near 0.15–0.19
on physics batches with the injection at 3–4% of the stream, a much gentler
operating point than the Qwen3 8B's saturated gate at the 20% cap; both are
selective.

## A third family: Qwen3.5 9B (2026-09-10/11)

The two families above are attention stacks; Qwen3.5 9B is mostly not. It
alternates three Gated DeltaNet layers (linear attention with a recurrent
state instead of a KV cache) with one full-attention layer — 24 recurrent
layers of 32, hidden 4096, a 248k untied vocabulary — and it is a
vision–language checkpoint of which only the text tower is driven. The copy
is Ollama's `qwen3.5:9b-mlx` release, NVFP4 at group size 16 on every wide
projection (embeddings, output head, norms and the recurrence's small
parameters stay bf16), reassembled by `eval/ollama_to_mlx.py`: 760 tensor
blobs in the manifest, 333 vision ones dropped, 627 kept, 8.0 GB, licence
copied alongside. MLX 0.32.2 reads NVFP4 natively; mlx-lm 0.31.3 ships
`qwen3_5.py`.

**Two adapter accommodations** (`psilm/mlx/qwen35_loader.py`), neither in
the bridges. (1) Masks: full attention takes the staged forward's additive
causal-and-padding mask; a Gated DeltaNet layer wants a boolean key-validity
mask (`where(mask, qkv, 0)`) and would zero every real token if handed the
additive one. The boolean mask is the additive mask's last row `== 0`, so the
shim derives it per layer kind. mlx-lm's own forward does not mask padding
without a cache, so the two agree only at batch 1 — which is where parity is
checked: staged vs stock is exact (max |diff| 0.0, kernel path), and a
right-padded row matches its unpadded forward at every real position
(`results/qwen35/setup_summary.txt`, from `eval/mlx_qwen35_setup.py`). (2) The
recurrent scan is a Metal kernel with no VJP; mlx-lm's training-mode fallback
is a pure-MLX scan that agrees with the kernel to 1.4e-2 in maximum relative
logit difference on the setup prompt (7.2e-3 on the commit-time prompt), with
the same argmax and top-5 at every position. `set_grad_window(l_rev)` puts only the layers the backward
pass reaches on that path. The bridges are trained and the per-chunk rollouts
scored on the ops path; the held-out evaluation and the guard-rail run the
kernel (`eval/mlx_stage2_eval.py --ops-path` scores on the training numerics
for comparison).

**The coupling depth was set by a measured cliff** (`results/qwen35/probes.txt`,
batch 2): inject at 28 → 4.17 s/step; 26 → 4.69 s at 16.0 GB; 24 → 10.03 s at
19.1 GB; 22 → 20.7 GB; 20 (the Stage-2 fraction, matching the 8B's and
Gemma's 61–62%) → 23.5 GB on a 24 GB machine. From 26 to 24 the recurrent work
above the injection grows by half and the time doubles — past ~16 GB the step
stops tracking the layer count (buffers spilling past what the GPU keeps
resident, unconfirmed). Coupling is 13/26 of 32: six layers above the
injection, four recurrent, against 18/48 (Gemma) and 14/36 (8B). A
gradient-checkpointed staged forward (`--checkpoint-from`,
`eval/checkpoint_selftest.py`: bit-identical gradients on the 0.5B) exists
and was not needed.

**Readout warm-up** (batch 8, 3.44 s/step, 2,000 steps in 1 h 56 min,
`--readout-norm dim` on): pointer CE 4.08 → 0.42 over 250-step windows,
exact bins 6% → 95%, position error 0.081 → 0.005. On the last-hundred-steps
statistic: CE 0.45, error 0.004, 91% exact — the best warm-up pointer here
(Gemma 1.24 / 0.009 / 69%, 8B 1.76 / 0.019 / 53%), with the caveat that this
warm-up saw 16,000 samples where theirs saw 8,000; at the matched 8,000 (step
1,000) it stood at CE 1.60 / 0.016 / 50%, between the two. The warm-up ran
with the injection depth set to 24; the coupled resume moved it to 26 (phase A
never injects; the trainer logs the change).

**Coupled phase** (batch 2, 8,000 steps = Gemma's 16,000 samples;
`results/qwen35/coupled_recipe.sh`). Negatives for the no-harm arm were built
first from this backbone's own continuations: 1,194 items over 597 distinct
prompts (400 GSM8K, 197 MMLU), each with and without the GSM8K nudge line —
the stripper is keyed to that line, so the 197 MMLU prompts appear twice
unchanged. Per-chunk rollouts (n=16; n=48 for the first) at steps
2,500–9,500: 12.5, 68.8, 62.5, 75.0, 68.8, 93.8, 87.5, 68.8, 81.3, 100, 93.8,
87.5, 81.3, 87.5, 87.5% at MAE 0.345 → 0.019; Gemma's eight coupled chunks
at n=48: 62.5, 66.7, 62.5, 87.5, 87.5, 62.5, 79.2, 54.2%. The pointer held at
97% exact bins over the coupled records, 100% at every chunk end. Gate 1.0
with the injection at the 0.2 cap throughout — the 8B's run-8 operating
point, not Gemma's 0.15–0.19 — so the 8B is the precedent for the pending
selective-gate phase. Step time 4.6–5.1 s over the first five chunks and
4.9–8.5 s after, at a flat 16.0–16.1 GB peak.

**Selective gate and result.** The no-harm phase (1,500 steps at lr 1e-4,
negatives every second step, gate-only updates) closed the gate on the
negatives within its first chunk — 0.001 at their answer positions by step
10,500 against 0.94 on physics prompts — and kept it shut (0.0003 at the end),
while the physics rollouts went 93.8 / 100 / 100% at MAE 0.022 / 0.013 / 0.013.
Those chunks peaked at 35–38 GB (the negatives are long sequences and the
recurrent scan's tape scales with length), so they ran through swap at
7.7 s/step. Held-out, n=60 (`results/stage2_qwen35/final_eval.json`, kernel
path): **PsiLM 100% / MAE 0.0147** (largest error 0.047), oracle 98.3% / 0.100
(its one miss is a forced reply whose parser took the phase 5.61 out of the
derivation, true −0.202; PsiLM said −0.21), backbone alone 3.3% / 0.570
(forced on every item), always-zero 1.7%. On the training numerics
(`--ops-path`, `final_eval_ops.json`) PsiLM is again 100% / 0.015, 58 of the
60 answers identical to two decimals and the other two within 0.01. Parity
and the scan comparison are in `results/qwen35/setup_summary.txt`. Twenty-two hours
of Apple-silicon time end to end (2 warm-up, 1 negatives, 16 coupled, 3.5
selective gate).

**Guard-rail on Qwen3.5** (n=100 per dataset; `results/bench/guardrail_qwen35_guardrail_summary.json`,
from `results/bench/guardrail_qwen35.sh`): physics 1 / 99 / 10% (backbone under
the 160-token nudge protocol / PsiLM / zeroed; gate 0.81, open on 100%; PsiLM
MAE 0.016, 16.9 tokens and 2.35 s per question against the backbone's 160 tokens
and 13.5 s), GSM8K@384 83 / 83 / 83% — item-identical across the three arms
(gate 0.004, open on 0%, KL to base 1.2e-4 per token), MMLU@256 64 / 66 / 64%
(two items gained, none lost, p = 0.5, from a parse rate of 0.86 against 0.82;
gate 0.014), BoolQ 90 / 89 / 90% (one item lost, p = 1; gate 0.008). The
bench's parity check on its KV-cached prefill passes at 5.8e-4 relative,
argmax unchanged. The zeroed arm's 10% on physics is the reply-template floor,
the same as Gemma's.

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

## Leaky gate, and what the channel actually carries (2026-09-09/10)

("Qwen" in this section means the Qwen3-8B bridges throughout; the sweep
predates the Qwen3.5 campaign, which has not been swept.)

A gate that is shut off-task carries nothing there. The question was whether a
weak always-on signal would act as a regularizer or improve general reasoning.
The answer is no — and chasing it produced the strongest evidence in the project
that the language model reads the latent channel at all.

**Setup.** A floor applied to the gate **at inference only**, nothing retrained:

```
sigma_eff = eps + (1 - eps) * sigma          # psilm/mlx/bridges.py, gate_floor
```

Monotone in sigma and it leaves the open end alone — `max(eps, sigma)` would
flatten the gate's decision below the floor and has a dead zone with no
gradient. The logged sigma stays **pre-floor**, so the gate's own decision remains
observable — that column is the instrument check. At prompt positions it is
identical across arms to five decimals (drift 0.0e0). The all-position mean also
covers generated tokens, which differ between arms, and drifts up to 1.3e-3 on
Qwen. On Gemma at eps=1.0 it is not invariant at all (physics 0.144 → 0.117):
the gate is computed from the residual stream it sits in, so a large enough
injection changes what the gate reads. That feedback is exactly what the
pre-floor column exists to expose.

```bash
python eval/bench_guardrail.py --tag leaky_8b --n 100 \
    --arms base,psilm,leaky0.01,leaky0.05,leaky0.1,leaky0.2 \
    --datasets physics,mmlu,gsm8k,boolq --kl --tasks-cache results/bench/tasks_leaky_n100.json
python eval/leaky_report.py results/bench/leaky_8b_guardrail.json \
    --merge results/bench/leaky_8b_hi_guardrail.json \
            results/bench/leaky_8b_top_guardrail.json \
            results/bench/leaky_8b_shuf_guardrail.json
```

Five runs, all seed 0, greedy, n=100 per dataset, sharing a task cache per
backbone. Each has its committed launcher, recording the arms, the waits and the
retry-with-resume harness:

| run | arms | launcher |
|---|---|---|
| `leaky_8b` | eps 0.01–0.2 | `results/bench/leaky_sweep.sh` |
| `leaky_8b_hi` | 0.3, 0.4, 0.5 | `results/bench/leaky_sweep_hi.sh` |
| `leaky_8b_top` | 0.7, 0.9, 1.0 | `results/bench/leaky_sweep_top.sh` |
| `leaky_8b_shuf` | shuffled 0.2, 1.0 | `results/bench/leaky_sweep_shuffled.sh` |
| `leaky_gemma` | 0.02–0.2, 1.0 | `results/bench/leaky_sweep_gemma.sh` |

The follow-ups reuse the first run's task cache and base continuations
(`--tasks-cache`, `--base-gen-from`), so questions, prompts and the KL reference
are identical rather than merely comparable. KL is per-token KL(base‖arm) over
the full vocabulary, teacher-forced on the base arm's own greedy continuation,
averaged over continuation positions.

### The dose-response curve (Qwen3-8B)

| eps | physics | MMLU | GSM8K | BoolQ | physics MAE |
|---|---:|---:|---:|---:|---:|
| backbone¹ | 2% | 60% (87t) | 89% (147t) | 88% | 2.239 |
| trained gate | 95% | 61% (83t) | 89% (145t) | 88% | 0.0215 |
| 0.1 | 96% | 66% (67t) | 89% | 88% | 0.0204 |
| 0.2 | 98% | 66% (62t) | 88% (137t) | 87% | **0.0182**² |
| 0.5 | 94% | 70% (24t) | **81%** p=.039 (108t) | 84% | 0.0202 |
| 0.7 | 90% | 69% (10t) | **50%** p<.001 (74t) | 77% p=.003 | 0.0219 |
| 1.0 | 89% | 69% (7t) | **8%** p<.001 (24t) | 59% p<.001 | 0.0236 |

¹ The backbone's physics number is not comparable with the guard-rail tables
above (5% for Qwen, 0% for Gemma): this sweep gives its nudge protocol a
160-token budget, at which nearly every reply truncates, rather than 768. The
coupled arms are unaffected — they answer in ~17 tokens.

Nothing off-task **degrades** until eps=0.5 (against the backbone: 9 lost, 1
gained, p=0.022; against the trained gate 10 and 2, p=0.039). MMLU rises earlier
and significantly — from eps=0.3, p=0.012 against the backbone — for the reason
in the next section. At eps=1.0 — sigma_eff identically 1.0,
which is run 8's operating point reached by force rather than by training —
GSM8K collapses to 8%. So **the open gate alone is sufficient**; open-gate
training is not required to produce run 8's failure.

It also refines run 8. That system, gate *trained* open, scored 34% at 70
generated tokens; forcing the same operating point on the *selective* bridges
gives 8% at 24 tokens. Forcing open is 4× more destructive than training open,
so run 8 had partially **adapted** to its own channel, and the no-harm arm's
job was closing the gate rather than repairing damage. MMLU replicates run 8
exactly: 69% at 7 tokens in both.

² The only significant on-task effect in the sweep, and it is in the
continuous measure rather than the thresholded one: per item eps=0.2 is nearer
the truth than the trained gate on 30 questions and farther on 12 (58 tied;
two-sided sign test p=0.008), which the ±0.05 tolerance almost entirely hides
(accuracy moves 3 items, p=0.25). The readout is identical in every arm, so this
is the frozen model rendering the value it was handed more faithfully when the
channel pushes harder — not better physics. It does **not** replicate on Gemma,
whose physics MAE is flat through its safe range (0.0173 / 0.0163 / 0.0174) and
then explodes (0.0228, 0.594, 4.100).

### The MMLU rise is a token budget, not reasoning

It is the one off-task column that rises, and it rises with three others:
tokens 87 → 62 → 7, EOS rate 0.71 → 0.82, parse rate 0.72 → 0.84 → 1.00.
Shorter replies finish inside the 256-token budget and get scored. Of the 28
items the backbone leaves unparsed, eps=0.2 parses 13 and loses 1 — all 13 in
college physics and mathematics, the two subjects carrying the whole movement —
and 7 are right (54%), against the 75% the backbone scores on items it already
parses in those subjects. **Items scored, not items solved.** On the 71
questions every arm answered, accuracy is 83.1% in all six arms of the first
sweep including the backbone . Across all fourteen arms of the family the
common set is still 71 questions, and there the backbone is the *best* arm at
83.1% while the coupled arms run 80.3-83.1% — so on the questions everyone
answers, the floor never helps and costs up to two items (60 of those 71 are subjects where nothing truncates, so the control
shows the floor changes nothing where length was never binding, and does not
settle the items where it was).

**BoolQ rules out truncation as the damage mechanism.** Its replies are four
tokens in every arm of every sweep — there is no length for the floor to change —
and it still falls from the backbone's 88% to 87 / 84 / 77 / 67 / 59% across
eps 0.2 / 0.5 / 0.7 / 0.9 / 1.0 (p<0.001 from eps=0.7). So the channel degrades capability directly, not only by
truncating reasoning. That is the cleanest separation in the sweep between the
budget artifact (MMLU) and real damage (BoolQ, GSM8K).

### The safe range belongs to the backbone, not to epsilon

Gemma 4 12B, same protocol, floors around *its* operating point
(`results/bench/leaky_gemma_guardrail_summary.json`):

| eps | physics | MMLU | GSM8K | BoolQ |
|---|---:|---:|---:|---:|
| backbone¹ | 10% | 53% | 84% | 90% |
| trained gate | 97% | 55% | 84% | 89% |
| 0.05 | 97% | 56% | 84% | 87% |
| 0.1 | 91% p=.031 | 54% | 71% p=.004 | 86% |
| 0.2 | 50% | 37% | 13% | 70% |
| 1.0 | 0% | 0% | 0% | 1% |

Gemma breaks 4–8× earlier in eps. Most of that is where each gate was trained
to sit — Qwen 0.791 on physics, Gemma 0.144 — so the same floor is a different
multiple of the model's own operating point:

| | eps | sigma_eff / sigma_trained | GSM8K |
|---|---:|---:|---:|
| Qwen | 0.5 | 1.13× | 81% (breaks) |
| Qwen | 1.0 | 1.26× | 8% (destroyed) |
| Gemma | 0.05 | 1.30× | 84% (intact) |
| Gemma | 0.1 | 1.59× | 71% (breaks) |
| Gemma | 0.2 | 2.18× | 13% (destroyed) |

Eight-fold apart in eps, ~1.4-fold apart in this ratio. **Not a law** — closing
the rest needs the realized injection-to-stream ratio, which `return_ratio` in
`psilm/mlx/bridges.py` can log and these sweeps did not. Log it next time.

**The MMLU budget effect does not replicate on Gemma — because of the width of
the window, not the sign of the effect.** Inside Gemma's safe range the floor
barely moves reply length (MMLU 134.0 / 134.1 tokens at eps 0.02 / 0.05, against
the trained gate's 134.4), so there is no regime where it shortens replies
without destroying the model; Gemma's MMLU accuracy never exceeds 56%. The
lengthening at larger floors (158.6 tokens at eps=0.2, 162.5 at 1.0) belongs to
arms already collapsed to 37% and 0%, and is not monotone — GSM8K falls back to
194 tokens at eps=1.0. That is derailment, not brevity with the opposite sign.
State the usable window as backbone-specific, not the direction.

### The content control: presence vs content

`shuffled<eps>` is `leaky<eps>` with one substitution — the value encoder gets
the `u_hat` of a **different question** from the same dataset, by a one-step
rotation of the sorted qids over the first sweep's own recorded readouts: 400
substitutions, **zero fixed points**, multiset of injected values unchanged, so
magnitude is matched by construction. Readout still runs on the real prompt,
gate still decides, `diag.u_injected` records what went in.

```bash
python eval/bench_guardrail.py --tag leaky_8b_shuf --n 100 --arms shuffled0.2,shuffled1.0 \
    --shuffle-values-from results/bench/leaky_8b_guardrail.rows.jsonl \
    --base-gen-from results/bench/leaky_8b_guardrail.rows.jsonl --kl ...
```

KL here is per-token KL(base‖arm) over the full vocabulary, teacher-forced on
the base arm's own greedy continuation and averaged over continuation positions,
so every arm is scored on the same tokens. Off-task both arms share the prompt;
on physics the base arm uses the nudge protocol, so there it is a common-token
comparison rather than a same-prompt one.

**Off-task the swap changes nothing** (eps=0.2): MMLU 0.66 vs 0.66 (61 vs 62
tokens), GSM8K 0.88 vs 0.88 (136 vs 137), BoolQ 0.87 vs 0.87, every paired test
p=1.00; at eps=1.0 the collapse reproduces at 0.07 vs 0.08. Even the KL matches
(GSM8K 0.055 vs 0.058). The brevity, the MMLU gain and the collapse are
**content-independent**.

**On-task the same swap is annihilating**: physics 98% → **9%** (p<1e-4, MAE
0.018 → 0.410) at identical tokens and parse rate, with KL to base *unchanged*
at 0.222 — the distribution moves just as far, to a different place. Per item:

| the spoken answer sits | distance | within ±0.05 |
|---|---:|---:|
| from the **injected (wrong)** value | 0.0121 | **99/100** |
| from the readout's own u_hat | 0.4000 | — |
| from the true answer | 0.4098 | 9/100 |

```
gold -0.095 | readout computed -0.103 | injected -0.546 | model said -0.550
gold +0.328 | readout computed +0.286 | injected -0.478 | model said -0.490
```

This is the cleanest evidence in the project that the frozen model **reads the
channel** rather than a template or a prompt correlate. The zeroed arm shows
removing the injection removes the physics result; the shuffled arm shows
corrupting *only the number* redirects the answer to the corruption with 99%
fidelity, while length, parsing, gate and off-task behaviour stay comparable.
Presence explains everything the channel does off-task; content explains
everything it does on-task.

### Conclusion

A weak always-on physics signal is not a regularizer. What it improves is the
chance of finishing inside a token budget — a decoding effect, not a capability.
The claim is about *inference*: these bridges were trained with a zero floor and
a no-harm arm penalizing exactly the gate the sweep forces open, so nothing ever
asked the channel to be useful off-task. A system **co-trained** with an
always-on channel is untested.

**Dose arithmetic.** `inj_cap` limits the injection to 0.2 of the stream RMS
*before* the gate scales it, so eps=0.2 delivers **at most** 4% of the residual
stream — a ceiling, not a measurement. Run 8's operating point was 20%.

## The constitution bridge: a partner that is a document (2026-09-12)

Every partner so far computed something the language model could not: a
calculator, a Burgers solver, a reaction–diffusion operator. This section
couples the frozen LLM to a partner that *contains a text* — Claude's
constitution, Anthropic's statement of the values it trains Claude toward
(anthropic.com/constitution, released under CC0; 28,958 words, 39 headings,
fetched into `data/constitution/claudes_constitution.md` with its provenance) —
and attaches the channel not to the residual stream at large but to a specific
set of coordinates in it: the backbone's *value neurons* in the sense of
arXiv:2602.00986 (Xu, Yuksekgonul and Zou, "Sparse Reward Subsystem in Large
Language Models"). That paper finds that the hidden state's estimate of whether
the current generation will end correct lives in under 1% of its dimensions,
found by training a two-layer value probe with a temporal-difference loss on
generated trajectories and pruning its input by the L1 norm of the first-layer
columns; zeroing that 1% collapses accuracy where random dimensions do nothing.
The question here is whether a constitution can be *written into* those
coordinates: whether the model's own sense of "how well is this going" is a
place where a second model's judgement of the situation can enter. The
development backbone is the 0.5B, as for every family before it; the 9B stage
runs from the same scripts.

**The partner** (`eval/build_constitution_corpus.py`,
`results/constitution_model/finetune_qwen0.5b.sh`, `eval/constitution_check.py`)
is Qwen2.5-0.5B-Instruct fully fine-tuned on the document — text chunks under
their heading paths, a recitation chat per section, 24 verbatim-grounded
question–answer pairs — for 600 iterations (about 7.3 epochs, batch 2,
sequence 1024; 13 minutes for the longest segment at 5.4 GB), then frozen and
exported as a loadable mlx-lm directory. It contains the document in the
literal sense and no other: 27 of 38 sections recite verbatim for 160 tokens
from their heading (the distribution is bimodal — a section either comes back
exact or a different section comes back exact), the four core properties come
out in their stated priority order, the three principals by name; but
held-out paragraphs of the same document score perplexity 513 against the base
model's 29, and novel dilemmas draw topically adjacent verbatim paragraphs
rather than answers. A memoriser, then, standing in the Fourier operator's
place: the bridges get to read its hidden states, and what those states can
carry about a situation is bounded by what the document says about one.

**The teacher.** No reward model is used. The teacher is the same frozen
backbone reading a verbatim excerpt of the constitution as its system prompt
(`data/constitution/system_excerpt.md`: the core values with their priority
order, the honesty properties, the hard constraints — 2,887 words, 3,422 Qwen
tokens, behind one framing line); the student is the backbone reading the plain
"You are a helpful assistant." with the bridges attached, and the loss is the
cross-entropy of the teacher's greedy continuation, exactly as the no-harm arm
scores the backbone's own. So the bridge is asked to reproduce, through the
latent channel, what the constitution-in-context would have changed. Prompts
are the opening human turns of Anthropic's HH-RLHF harmless-base set (2,000
train, 200 validation from the train split; 100 held-out from the test split,
cleared against the whole train pool, which leaves 210 usable of 2,131 because
the set reuses openings), plus 100 helpful-base test prompts as an
ordinary-request control (`eval/build_constitution_data.py`; the teacher's
3.4k-token prefix is KV-cached once and copied per prompt, 2.5× faster and
token-identical on the check prompts). What a 0.5B teacher does with the
excerpt is worth stating before any training number: it changes its
continuation on 97–99% of red-team prompts, but the change is only partly
refusal (0.38 → 0.43 keyword rate on train, 0.38 → 0.41 on test) — it is
mainly terseness (114 → 62 tokens) and stopping (eos 55% → 84%), and it
refuses *more* on ordinary requests too (0.04 → 0.12). Distilling this teacher
transfers all of that; the stronger teacher is the 9B's.

**Finding the value neurons** (`eval/vn_collect.py`, `eval/vn_probe.py`,
`eval/vn_ablate.py`, `psilm/mlx/value_neurons.py`). 1,000 GSM8K-*train*
problems, one greedy continuation each through a KV-cached staged decode that
stores the residual stream entering blocks 4–20 at every position (5.5 GB of
float16 states; reward 0.308). Greedy rather than the paper's temperature 1.0
because this backbone solves 6% of problems under that sampler and 42% greedy:
a 6% positive rate leaves a dozen positive held-out trajectories to rank layers
on, and greedy is the policy every evaluation here decodes with. The probe is
the paper's (896 → 1024 → 1, ReLU, AdamW 1e-4, batch 4 trajectories, 80/20 by
trajectory), and its loss is where the recipe had to bend: the temporal-
difference objective, with one reward-bearing residual among ~230 per
trajectory, left the probe nearly constant after 30 epochs — held-out AUC
0.42–0.60 across seven layers, below chance on four (archived in
`results/value_neurons/qwen0.5b/td30/`). Regressing every state on its
trajectory's reward directly (`--target mc`, the fixed point TD converges to at
γ → 1, reached without the backward propagation TD needs) gives 0.61–0.70 from
the prompt-final position at every layer (0.64–0.81 from the last generated
position). Pruning to the top 1% — nine dimensions — keeps 0.63 at layer 16
(0.64 at 4, 0.63 at 10, 0.61 at 18) against 0.43–0.56 for nine random
dimensions retrained the same way: a sparse value signal exists at 0.5B, weaker
than the paper's on RL-trained 7B models, and the ranking finds it. Layer 16
was chosen as the injection depth by that criterion among 14–20 (one layer above
the Stage-2 rule's 15).

The causal check reproduces the paper's shape and then undermines its own
reading (n = 100 GSM8K test, greedy, 384 tokens,
`results/value_neurons/qwen0.5b/ablate_layer16.json`): base 31%; the nine value
neurons zeroed at every position, 18% (McNemar p = 0.03); the top 45, 8%; three
random-nine controls 26%, 0% and 29%. The 0% is not a value effect: that draw
contains dimension 62, which carries the attention sink — |h| = 1,551 at
position 0 and 0.5 everywhere else — and zeroing it at layer 16 leaves no item
stopping. A random control that can draw the sink measures the sink, so
`--random-exclude-sink` keeps such dimensions (detected as > 100 in magnitude at
position 0 and > 50× their level elsewhere; exactly dimension 62 at every layer
of this backbone) out of the pool. The sink-free redraws come back 36%, 19% and
32% (`ablate_layer16_sinkfree.json`) — and the middle one is as damaging as the
value neurons themselves.

That draw is not bad luck; it identifies a confound the paper's random baseline
does not control for: **the damage tracks activation magnitude.** Ranked by
activation RMS over rollout states at this layer, the damaging draw holds the
5th and 18th largest dimensions of 896; the two benign draws hold nothing above
rank 125; the nine value neurons hold ranks 1, 8 and 9 and sum to 13.6 of RMS
against the benign draws' 3–4. So zeroing the value neurons may damage the model
because those coordinates are *large*, not because they carry value. The matched control
this calls for is three draws of nine dimensions, each member matched one to one
to a value neuron's activation RMS and taken from outside the top 5% and outside
the sink (`magmatch<seed>_layer16.json`, `results/constitution/magmatch_qwen0.5b.sh`).
It dissolves the effect:

| zeroed at layer 16 | GSM8K | McNemar vs base | summed RMS | mean generated tokens |
|---|---:|---:|---:|---:|
| nothing (base) | 31% | — | — | 255 |
| nine value neurons | 18% | 0.029 | 13.6 | 246 |
| magnitude-matched draw 0 | 21% | 0.099 | 11.5 | 271 |
| magnitude-matched draw 1 | 26% | 0.442 | 10.3 | 275 |
| magnitude-matched draw 2 | 6% | < 1e-4 | 11.5 | 88 |

The three matched draws average 17.7% against the value neurons' 18.0%, so at
this backbone and this layer the value-neuron set is indistinguishable from any
nine comparably large coordinates. The base arm reproduced 100 of 100 items
across the two runs, so the reference is not drifting. The paper's random
baseline is drawn uniformly, which on a stream whose magnitudes span two orders
of magnitude almost always samples small coordinates: that is what makes its
contrast look decisive, and it is not a control for magnitude. Draw 2 also shows
a second failure mode the accuracy column hides, cutting generated length from
255 tokens to 88 with the end-of-sequence rate rising to 0.90 — zeroing large
coordinates can stop the model rather than mislead it.

What survives is the probe-level evidence, which is independent of magnitude
because a probe rescales its own inputs: a large dimension earns no advantage
there, and the top 1% still keeps AUC 0.63 where random nines retrained reach
0.43–0.56. So there is a value signal in a sparse set of coordinates, and those
coordinates are also large, and this ablation cannot separate the two claims.
Whether the *bridge* result separates them is the matched training variant
below.

**The bridge** (`psilm/mlx/constitution.py`; 3.69M parameters). The forward
bridge reads the whole residual stream at layer 10 — per-dimension calibration
and RMS as for Gemma, then eight learned queries attention-pooling the prompt
into eight soft tokens rescaled to the partner's embedding RMS (0.015; without
that rescale they arrive 60× too large). The partner runs them between a fixed
prefix and a fixed suffix ("According to Claude's constitution, in this
situation Claude should"), and its final-norm states at the last eight
positions are the "physics features": a reverse bridge maps them to eight
tokens, and the gated cross-attention of the physics bridges injects them at
layer 16 — with one change. Its output is multiplied by a frozen mask over
`write_dims`, the injection cap and the receiver's local scale are taken over
the written coordinates only (otherwise "5% of the local stream" is 10× too
tight for a nine-dimension write), and the gate and the attention query still
read the whole stream. With the mask all-ones the module is bit-identical to
`GatedCrossAttentionMLX`, which is the self-test's first assertion; with a
nine-dimension mask the delta is exactly zero elsewhere, which is its second.
So the write location is a training-time flag, and the ablation is four runs
of one script that differ in it: the nine value neurons (`vn`), the top 45
(`vn5`), nine random dimensions drawn from the complement (`rand`), the whole
stream (`all`). Everything else is the Qwen3.5 recipe — 2,000 steps at batch 8,
lr 3e-4, gate bias 0, cap 0.2, per-module clipping, a no-harm step every second
step on GSM8K-train/MMLU-validation prompts with the 0.5B's own continuations
(`data/noharm_qwen0.5b_all.json`, 1,194) updating the gate only under a gate
penalty of 1. 1.7 s per step, 15.5 GB peak. The gate on the no-harm prompts
settles where the two arms balance, and where that is depends on the width of
the write: 0.0001 for the nine value neurons, 0.004 for the top 45, 0.010 for
the whole stream (0.0 for the nine random dimensions, whose channel never
became worth keeping open) — the first sign of the trade-off the guard-rail
below measures.

| write into | dims | val CE to teacher | test CE | test top-1 agreement | test refusal (32 tok) | ordinary CE | ordinary refusal | gate on generated |
|---|---:|---|---|---|---|---|---|---|
| base (no bridge) | — | 0.977 | 0.943 | 78.0% | 34% | 0.789 | 2% | — |
| teacher (stored) | — | — | — | — | 41% | — | 12% | — |
| nine value neurons | 9 | 0.962 | 0.925 | 78.3% | 35% | 0.771 | 2% | 0.50 |
| top-45 value neurons | 45 | 0.922 | 0.883 | 78.8% | 31% | 0.748 | 1% | 0.54 |
| nine random dims | 9 | 0.975 | 0.941 | 78.0% | 32% | 0.788 | 2% | 0.09 |
| nine magnitude-matched dims | 9 | 0.964 | 0.929 | 78.3% | 34% | 0.782 | 3% | 0.44 |
| whole stream | 896 | 0.628 | 0.585 | 83.1% | 57% | 0.591 | 8% | 0.51 |
| nine value neurons, *plain* partner | 9 | 0.957 | 0.921 | 78.2% | 32% | 0.770 | 2% | 0.50 |
| whole stream, *plain* partner | 896 | 0.627 | 0.575 | 83.1% | 61% | 0.580 | 11% | 0.89 |

(`results/stage2c_qwen0.5b_<variant>/`, `eval/mlx_constitution_eval.py`,
n = 100 each; the zeroed arm equals the base arm to the last digit in every
run.) The last two rows are the partner control: the same writes with the
untouched Qwen2.5-0.5B-Instruct as the partner — same architecture, no
constitution in its weights. Both match their constitution-partner twins
within noise at every chunk of training (0.957 against 0.962 and 0.627
against 0.628 on validation; 0.921 against 0.925 and 0.575 against 0.585 on
test), and the full-width pair agree on refusals, length and the spillover
onto ordinary requests. So at 0.5B the document in the partner's weights is
not what the channel carries. The constitution enters this system through
the teacher — the excerpt in the backbone's own context — and the bridges
distil it into a channel that any frozen model of the right shape can serve
as the far end of: the partner is a latent scratchpad, not a source. Whether
a partner that *reasons* with the document (rather than reciting it) changes
that is a question for a larger partner, not for more epochs on this one.

Three things are in that table. First, the channel works: written into the
whole stream, the bridges take the held-out CE against the teacher's tokens
from 0.943 to 0.585 (the floor is the teacher's own CE on its greedy tokens,
not measured here), lift agreement five points, halve the reply length, and
reproduce the teacher's refusals — and overshoot them, 57% against the
teacher's 41%, with the teacher's own spillover onto ordinary requests (8%
against its 12%; base 2%). Second, the
write location matters, though not yet demonstrably *because* of value: nine
random dimensions carry nothing (the optimizer gives up and lets the no-harm
penalty shut the gate to 0.09), nine value neurons carry a small but
reproducible signal with the gate held at 0.50, and forty-five carry three
times as much. The ablation's magnitude confound applies to that contrast too,
and here it is starker: the nine random dimensions sum to 3.3 of activation RMS
against the value neurons' 13.6 and none of them reaches rank 164, so the random
write is a smaller perturbation of the stream by construction. The
magnitude-matched variant settles it, and it splits the difference. Trained under
the identical recipe with nine dimensions matched to the value neurons' activation
RMS, it reaches 0.929 on the same held-out hundred against the value neurons'
0.925, the random write's 0.941 and the base's 0.943. Paired over items, with a
20,000-sample bootstrap on the per-item cross-entropy:

| comparison, same 100 items | mean CE difference | 95% CI |
|---|---:|---|
| value neurons vs base | −0.0182 | [−0.0203, −0.0159] |
| magnitude-matched vs base | −0.0137 | [−0.0160, −0.0115] |
| nine random dims vs base | −0.0025 | [−0.0033, −0.0017] |
| value neurons vs magnitude-matched | −0.0045 | [−0.0072, −0.0016] |
| magnitude-matched vs nine random | −0.0112 | [−0.0130, −0.0095] |

So activation magnitude accounts for about three quarters of what the
value-neuron write buys, dimension identity for the remaining quarter, and that
last quarter is not noise: the interval excludes zero and 71 of the 100 items
move toward the teacher under the value-neuron write. The honest reading of the
whole design, then, is that coupling into the value neurons works partly for the
reason the paper suggests and mostly because those coordinates are the ones the
stream actually carries weight in — and that the original random-dimension
control, which the physics campaigns would have accepted, was far too weak to
show it. One caveat stands: this bridge-level comparison rests on a single
matched draw where the ablation used three, and the ablation's draws spread from
6% to 26%. Two further matched draws are queued behind the 9B stage
(`results/constitution/magmatch_more_qwen0.5b.sh`). Third, the value-neuron write is *narrow* at this width:
training CE on the teacher tokens equals held-out CE in every variant (no
overfitting), so the plateau at 0.96 is capacity — nine coordinates of 896
cannot hold what the constitution model has to say about a prompt, and the
effect scales with the number of coordinates written.

**The guard-rail closes the argument** (`eval/bench_guardrail.py --bridge-kind
constitution`, the same three arms and n = 100 per dataset as every guard-rail
before, plus the held-out red-team prompts as a fourth, free-form dataset
scored for refusal and KL):

| write into | GSM8K base/psilm/zeroed | MMLU | BoolQ | gate (GSM8K) | KL to base (GSM8K) | red-team gate | red-team refusal base → psilm | red-team KL |
|---|---|---|---|---|---|---|---|---|
| nine value neurons | 31 / 31 / 31 | 19 / 20 / 19 | 64 / 65 / 64 | 0.002 | 7e-6 | 0.36 (open on all) | 39% → 40% | 0.002 |
| top-45 value neurons | 31 / 33 / 31 | 19 / 19 / 19 | 64 / 64 / 64 | 0.004 | 1.4e-4 | 0.39 (open on all) | 39% → 35% | 0.007 |
| nine random dims | 31 / 32 / 31 | 19 / 19 / 19 | 64 / 64 / 64 | 0.0001 | 1e-5 | 0.04 (open on 9%) | 39% → 37% | 2e-5 |
| nine magnitude-matched dims | 31 / 32 / 31 | 19 / 19 / 19 | 64 / 64 / 64 | 0.001 | 6e-06 | 0.31 (open on 99%) | 39% → 38% | 0.001 |
| whole stream | 31 / 23 / 31 | 19 / 19 / 19 | 64 / 66 / 64 | 0.008 | 0.022 | 0.27 (open on all) | 39% → 58% | 0.26 |
| nine value neurons, plain partner | 31 / 31 / 31 | 19 / 19 / 19 | 64 / 65 / 64 | 0.002 | 1e-5 | 0.37 (open on all) | 39% → 39% | 0.002 |
| whole stream, plain partner | 31 / 25 / 31 | 19 / 19 / 19 | 64 / 70 / 64 | 0.018 | 0.011 | 0.42 (open on all) | 39% → 61% | 0.28 |

(`results/bench/const_qwen0.5b_<variant>_guardrail_summary.json`.) The
plain-partner rows repeat their twins' profiles, leak included (GSM8K 15 items
lost and 9 gained at full width, against 15 and 7). So does the
magnitude-matched write, at KL 6e-6 on GSM8K against the value neurons' 7e-6 and
the same 31/32/31: whatever the ablation says about which nine coordinates are
load-bearing, a nine-coordinate write is harmless either way. Harm here is set by
the width of the write, not by which dimensions it lands in. The value-neuron writes
are harmless in the strongest sense this harness measures — GSM8K item-level
identical up to a few flips each way (two and two for nine dimensions, three
and five for forty-five), KL to the base below 1e-3 on every benchmark — and
nearly inert on the red-team prompts, where the 45-dimension write shortens
nothing and refuses four points less by the keyword count. The whole-stream write carries the
constitution and costs eight GSM8K points (15 items lost, 7 gained,
p = 0.13) at a gate of 0.008: a gate that the no-harm arm could not drive
below 0.01 once the whole stream was writable, carrying content ("be brief,
decline") that perturbs a chain of thought where a physics value did not. The physics campaigns closed with the gate as the switch; this one shows
the write mask as a second, structural one. The 0.5B leaves the two apart:
what is confined to the value neurons cannot hurt and cannot do much, what
reaches the whole stream does both, and the leak is ordered by width — KL to
the base on GSM8K of 7e-6, 1.4e-4 and 2.2e-2 for 9, 45 and 896 written
dimensions. The 45-dimension write sits between them in effect and stays on
the harmless side; the 9B — where 1% of the stream is 41 dimensions and the teacher is
a model that can follow a 3,400-token system prompt — is where the design is
meant to be judged.

### The value neurons at 9B, and what the ablation says (2026-09-13)

The same identification on Qwen3.5 9B, whose stream is 4096 wide, so the top 1%
is 41 dimensions rather than nine. 800 GSM8K-train trajectories, this time at the
paper's own sampler (temperature 1.0, top-p 0.95) because this backbone solves
83% of the set greedily and a greedy collect would leave almost no negatives; the
sampled policy lands at 0.741, the mirror image of the 0.5B's 0.308 under greedy
decoding. Seven depths captured at every position, 4.3 hours, 15 GB of float16
states, reward-balanced 593 correct against 207 incorrect.

| layer | full-width AUC | top 1% (41 dims) | random 41 |
|---|---:|---:|---:|
| 13 | 0.793 | 0.762 | 0.613 |
| 20 | 0.837 | 0.766 | 0.658 |
| 22 | 0.820 | 0.776 | 0.757 |
| 24 | 0.821 | **0.788** | 0.712 |
| 26 | 0.833 | 0.744 | 0.738 |
| 28 | 0.818 | 0.749 | 0.735 |
| 30 | 0.832 | 0.739 | 0.740 |

Two things read off that table. The value signal is much stronger than the
0.5B's, 0.79–0.84 at full width against 0.61–0.70 — a bigger, better-trained
model carries a far more legible estimate of whether its own continuation will be
correct. But the *sparsity* claim weakens as depth grows: at layer 22 and beyond,
41 arbitrary dimensions predict nearly as well as the top 41, and at layer 30
they predict better. Whatever concentration exists lives in the early-middle
layers, which is where the paper looks (its layers 2–4), and it is gone by
two-thirds depth. Layer 24 was chosen as the injection depth by the same rule as
before, best AUC at 99% pruning among the candidates, and it happens to hold the
widest top-versus-random gap of the three, +0.076. That gap deserves an error
bar: with 119 correct and 41 incorrect held-out trajectories the Hanley–McNeil
standard error on a single AUC of 0.79 is 0.037, so the gap is about 1.5
conservative standard errors. Suggestive, not established.

**The causal check finds nothing at all** (n = 100 GSM8K test, greedy, 384 tokens,
`ablate_layer24.json`):

| zeroed at layer 24 | GSM8K | McNemar vs base |
|---|---:|---:|
| nothing (base) | 83% | — |
| 41 value neurons (top 1%) | 80% | 0.375 |
| 205 value neurons (top 5%) | 82% | 1.0 |
| 41 uniform random dims, three draws | 82% / 83% / 84% | 1.0 each |

Zeroing the top 1% costs three points at p = 0.375, zeroing five times as many
costs one point, and random draws cost nothing. Against the 0.5B, where the same
intervention cost 13 points, this is a flat null — and it makes the
magnitude-matched control planned for this backbone moot, because a matched
control exists to ask whether damage is value or magnitude and here there is no
damage to attribute. It was cancelled with its reasoning in
`results/qwen35/vn35_magmatch.log`; the matched sets remain on disk.

So across the two backbones the paper's flagship causal result fails twice over,
in two different ways: at 0.5B the damage was real but magnitude-matched
coordinates reproduced it, and at 9B there is no damage to explain. The
magnitude story also inverts with scale, which may be why. At 0.5B two of the
nine value neurons sit in the top 1% of dimensions by activation RMS, a 22-fold
enrichment over chance, and the set sums to 13.6 of RMS against a median
dimension of 0.386. At 9B three of 41 sit there, a 7.3-fold enrichment, summing
to 90.7 against a median of 0.961. The wider model's value coordinates are much
less concentrated in its loudest channels, and it has no attention-sink dimension
at this layer at all, where the 0.5B's dimension 62 stands at 1,551 in magnitude
at position 0. A small model appears to pack value information into the few
channels it can least afford to lose, which is exactly the regime where an
ablation cannot distinguish the two explanations.

None of this touches the bridge, which asks a different question: not whether
those coordinates are load-bearing for the model's own reasoning, but whether
they are a useful place to write into.

### The bridge at 9B: the mask stops earning its place (2026-09-15)

Three write widths on Qwen3.5 9B, everything else identical to the 0.5B recipe
(read the whole stream at layer 13, inject at 24, gate bias 0, cap 0.2, no-harm
step every second step on `data/noharm_qwen35_all.json`, 28.34M bridge
parameters): the top 1% of value neurons (`vn`, 41 of 4096), the top 5% (`vn5`,
205) and the whole stream (`all`). The teacher is the same frozen backbone given
the 3,422-token constitution excerpt as its system prompt, and it differs from
base on 91% of the held-out red-team prompts while refusing 0.750 against base's
0.670 (`data/constitution_qwen35_stats.json`).

| write into | dims | val CE | test CE | test agree | helpful CE | helpful refusal | GSM8K | MMLU | BoolQ | red-team refusal | KL red-team | KL GSM8K |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| base | — | 0.4720 | 0.4829 | 0.8599 | 0.4130 | 0.02 | 83 | 64 | 90 | 0.660 | — | — |
| teacher (data build, n=100) | — | — | — | — | — | 0.07 | — | — | — | 0.750 | — | — |
| value neurons | 41 | 0.4589 | 0.4702 | 0.8595 | 0.4001 | 0.02 | 83 | 64 | 90 | 0.660 | 0.0023 | 0.00005 |
| top 5% | 205 | 0.4446 | 0.4552 | 0.8600 | 0.3877 | 0.02 | 85 | 64 | 89 | 0.650 | 0.0098 | 0.00008 |
| whole stream | 4096 | 0.3904 | 0.3890 | 0.8678 | 0.3554 | 0.06 | 84 | 65 | 89 | 0.720 | 0.1047 | 0.00035 |

(`results/stage2c_qwen35_<variant>/`, `results/bench/const_qwen35_<variant>_guardrail_summary.json`,
`results/constitution/summary_qwen35.json`. CE and agreement are n = 50 at 24
generated tokens; the benchmarks and the red-team refusal column are n = 100 at
the guard-rail's own lengths, 384 tokens for GSM8K and 128 for red-team. The
zeroed arm equalled the base arm to four decimals in all six evaluations.)

**Nothing here harms anything.** Across three variants and four datasets the
largest movement is two GSM8K items gained by the 205-dimension write
(McNemar p = 0.5); MMLU moves by at most one item each way and BoolQ by one.
This is the 0.5B's central conclusion inverted. There, the full-width write cost
eight GSM8K points — fifteen items lost against seven gained — and the notes
above closed by treating the gate and the write mask as two switches, because
"what is confined to the value neurons cannot hurt and cannot do much, what
reaches the whole stream does both". At 9B the second half is simply false, and
the reason is in the last two columns: the gate learned the selectivity the
no-harm arm trains for. Its KL to the base is 302× larger on red-team prompts
than on arithmetic at full width (0.10468 against 0.00035), where the 0.5B's
full-width gate managed only 12× (0.26 against 0.022) and bled into the
reasoning. Selectivity *rises* with width here — 48×, 119×, 302× — so the mask
is not what keeps the wide write safe, and on this backbone nothing needs to.

**Only full width transmits a decision.** Refusal on the held-out red-team
prompts is 0.660 for base, 0.660 for the 41-dimension write, 0.650 for the
205-dimension write, and 0.720 for the whole stream against the teacher's 0.750.
So the two masked variants transfer no behaviour at all, while the unmasked one
moves six points of the eight or nine that separate base from the teacher
(the guard-rail measures base at 0.660 and the data build at 0.670). A
measurement note, because it reversed a reading once: at 24 generated tokens the
full-width write appears to flip only two refusals of fifty, where the 128-token
guard-rail shows six points on a hundred. Short windows truncate before a refusal declares itself,
and the eval's 24-token column understates every behavioural effect in this
table.

**Cross-entropy is sublinear in width, with no sweet spot.** On converged values
the improvement over base fits ΔCE = 0.0032·d^0.393 to within 6% across two
decades (0.0131, 0.0274, 0.0816 observed against 0.0137, 0.0257, 0.0834 fitted),
so quadrupling the written dimensions buys about 1.7× the signal. Three points
and a fitted exponent are an empirical regularity, not a law, but the shape is
the useful part: there is no width at which the signal jumps, and 20× the
dimensions buys 3× the effect.

**What the channel carries changes with width, and so does what it costs.**
Comparing the relative CE gain on the two splits separates manner from judgment.
At 41 dimensions the gain is 2.63% on red-team prompts and 3.12% on ordinary
ones; at 205 it is 5.74% and 6.13%. Both are *larger* where the constitution has
least to say, which is the signature of a channel transmitting the teacher's
style — its terseness and hedging under a long system prompt — rather than its
situational reasoning. Only at full width does the order invert, 19.45% against
13.95%. And the over-refusal arrives at exactly the same width: helpful-split
refusal stays at base's 0.02 for both masked variants and rises to 0.06 for the
whole stream, which also refuses one ordinary prompt that neither base nor the
teacher refuses. Counting items, the full-width write acquires 2 of the
teacher's 4 extra refusals on ordinary requests and 2 of its 9 on harmful ones
in the 24-token window — a bias toward the unwanted half, though Fisher's exact
test on 2-of-9 against 2-of-4 gives p = 0.53, so that is a direction and not a
finding. The honest statement is narrower: below full width the channel carries
the teacher's manner and changes nothing; at full width it starts carrying the
teacher's decisions, and both halves of them come together.

**So the requested design does not earn its place at this scale.** Writing only
into the value neurons was motivated by the paper's causal result, which does not
reproduce here at all, and by the hope that a narrow write would be the safe way
to carry values. The first motivation is gone and the second turns out to be
unnecessary: at 9B every width is safe, and the mask's only measured effect is to
throttle the signal by the power law above. The probe evidence for those
coordinates still stands (AUC 0.788 at 1% against 0.821 at full width) and
nothing here shows the *location* is wrong — a 41-dimension write is about 16×
more efficient per written dimension than a 4096-dimension one. But efficiency
per dimension is not the constraint anyone has; total transfer is, and the mask
costs it. If the constitution arm of a larger system is meant to change
behaviour, it should write the whole stream and rely on the gate.

What remains untested, and matters more than width: the partner control. At 0.5B
an untouched Qwen2.5-0.5B-Instruct matched its constitution-trained twin within
noise at every width, so the document in the partner's weights was not what the
channel carried — the constitution entered through the teacher's context, and the
partner served as a latent scratchpad. That control was not repeated at 9B, and
the 9B runs use the same memorising 0.5B partner. Until a partner that *reasons*
with the document is tried, the honest description of this system is a
distillation of a prompted teacher through a latent bottleneck, with the partner
model as the bottleneck's shape rather than its source.

**Methodology.** Each variant ran 2 chunks of 500 steps at batch 2 rather than
the 4 the script originally specified. The constitution bridge runs at 26.7 s/step
on this backbone — 5.3× the physics campaign's rate at the same batch, because
peak memory is 48.9 GB against 25.8 GB of physical RAM — which makes a chunk
3h43m. Every variant converged inside its first chunk: the chunk 1 → 2 movement
in held-out CE was 0.0004 (`vn`), 0.0010 (`vn5`) and 0.0010 (`all`), the last of
these flatter than the 0.5B's full-width write was at chunk *four*. The 4-chunk
original is kept verbatim as `results/qwen35/constitution_train_4chunk.sh`; the
trim is documented in the running script's header. Six of the original twelve
chunks were dropped, about 22 hours of GPU; extrapolating each variant's
chunk 1 -> 2 movement, the cost is of order 0.001-0.002 of held-out
cross-entropy per variant, which is an estimate and not a measurement.

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
    gemma_loader.py      Gemma 4 text tower in the MlxStream layout; load_backbone_any() dispatches Qwen3, Gemma 4 and Qwen3.5
    qwen35_loader.py     Qwen3.5 (GatedDeltaNet + attention) text tower: per-layer-kind masks, windowed differentiable scan
    vlm_loader.py        Qwen3.8-27B language tower (inference-feasible only on 24 GB)
    moe_patch.py         autograd patch for MoE routing indices
    value_neurons.py     the value-neuron pipeline's pieces: capturing decoder, value probe, zeroing shim, sink detection
    constitution.py      the constitution bridge: masked-write gated injection, partner-model wrapper, coupled model, bench coupler
eval/                  training, evaluation and benchmark scripts
  mlx_stage2_{train,eval}.py, mlx_stage2b_*, mlx_stage2d_*   MLX trainers and four-arm evals
  stage1_*, stage2_*, stage2b_*, stage2d_*                    torch counterparts
  bench_guardrail.py, bench_common.py, build_noharm.py        gate-selectivity benchmark and its negatives
  copy_probe.py, readout_probe.py, readout_variance_probe.py  the diagnostic probes of the 8B/Gemma campaigns
  mlx_8b_setup.py, mlx_27b_setup.py, mlx_gemma_setup.py, mlx_qwen35_setup.py   parity/memory smoke tests per backbone
  ollama_to_mlx.py                                            rebuild an Ollama -mlx release into an mlx-lm directory (Qwen3.5)
  checkpoint_selftest.py                                      gradient-checkpointed vs taped staged forward (bit-identical on the 0.5B)
  export_bridges.py                                           checkpoint -> HF layout (safetensors + config.json)
  vn_collect.py, vn_probe.py, vn_ablate.py                    value neurons (arXiv:2602.00986): trajectories with states, TD/MC probe + pruning, zeroing ablation
  build_constitution_corpus.py, constitution_check.py         the constitution partner model's corpus and its recitation checks
  build_constitution_data.py                                  teacher (constitution excerpt in context) vs base continuations on HH-RLHF prompts
  mlx_constitution_{train,eval}.py                            the constitution bridge's trainer and evaluator (write-location variants by flag)
  leaky_report.py, assess_baseline.py, summarize_probes.py    dose-response, baseline and probe tables
data/                  QA datasets (single-mode, multi-mode families, 2D), the no-harm negatives, the constitution corpus and teacher data
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

A backbone that was assembled locally rather than downloaded (Qwen3.5, from
Ollama) has no Hub id to record: `config.json` would otherwise carry a path
that exists only on the training machine. `--backbone-name` sets what is
recorded, and it must be something a downloader can actually load — either a
Hub repo holding the identical conversion, or a note that names the Ollama
tag and the converter. For Qwen3.5 the choice was to republish the conversion
itself at the root of `ryoji-info/Qwen3.5-9B-PsiLM`, beside the bridges, so
that id is what the exported `config.json` records.
