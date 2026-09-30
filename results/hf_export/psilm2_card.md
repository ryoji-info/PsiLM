---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - psilm
  - physics
  - constitutional-ai
  - latent-coupling
  - qwen3.5
---

![ΨLM-2](psilm2-banner.png)

# ΨLM-2 — the bridges, the partner, and the evaluated items

One frozen **Qwen3.5 9B** (NVFP4, 32 layers, d = 4096) coupled to two frozen
partners through small trainable bridges, with no text at either interface:

- a **physics bridge** (45.18M parameters; read at layer 13, write at 26) to a
  Fourier neural operator — published with the backbone in
  [ryoji-info/Qwen3.5-9B-PsiLM](https://huggingface.co/ryoji-info/Qwen3.5-9B-PsiLM);
- a **constitution bridge** (28.34M; read at 13, write at 24) to a small frozen
  partner holding [Claude's constitution](https://www.anthropic.com/constitution) —
  the partner and every trained bridge are **here**.

Nothing pretrained is fine-tuned during bridge training: the backbone and both
partners stay frozen (the partner here was itself produced by a full fine-tune of
Qwen2.5-0.5B-Instruct, then frozen). The paper is
[`paper/psilm2.pdf`](https://github.com/ryoji-info/PsiLM-2/blob/main/paper/psilm2.pdf)
in the code repository, and every number below traces to a committed evaluation
file in [ryoji-info/PsiLM](https://github.com/ryoji-info/PsiLM).

## Contents

| path | what |
|---|---|
| `constitution_model/` | the partner: Qwen2.5-0.5B-Instruct fully fine-tuned on the constitution (988 MB of bf16 safetensors, about 1.0 GB with the tokenizer; its own card inside) |
| `bridges/qwen3.5-9b/<variant>/` | constitution bridges on the 9B: `bridges.safetensors` + `config.json` (the training meta, the resolved write mask, the cap, the step, sha256) |
| `bridges/qwen3.5-9b/all/tokens.safetensors`, `tokens.json` | added 2026-09-30: the stored tokens of the full-width bridge (1 × 8 × 4096, the mean of the partner's tokens over 100 validation prompts) and their record. With them that one bridge runs with no partner model ([below](#the-full-width-bridge-without-its-partner)) |
| `bridges/qwen2.5-0.5b/<variant>/` | the same at 0.5B (backbone: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`) |
| `physics/qwen3.5-9b/` | the physics bridge in the same layout, so the dual stack loads from one place |
| `caches/` | the guard-rail task caches: the exact 100 and 400 items every arm was scored on |

Variants on the 9B, all at layer 24, 1,000 steps, two 500-step chunks:

| variant | writes into | cap | note |
|---|---|---|---|
| `vn` | the 41 value neurons (top 1%) | 0.2 | 3.8% of full width's write energy |
| `vn5` | the top 5% (205) | 0.2 | 10.2% |
| `all` | all 4096 | 0.2 | full width; the dual stack's constitution channel |
| `vn10e` | probe-best 410 | 0.522 | energy parity with full width |
| `vn10ebot` | probe-worst 410 | 0.551 | parity |
| `match0/1/2` | magnitude-matched random 410 | 0.56 | parity; the identity controls |

## What was found (the four results of the paper)

1. **The value-neuron probe reproduces at both scales; its causal justification does not.** At 0.5B, ablation damage tracks activation magnitude and magnitude-matched controls reproduce it in full; at 9B the ablation is a flat null.
2. **The probe's signal occupies a roughly fixed number of coordinates, about 100–200, at d = 896 and at d = 4096** — a falling fraction as models grow.
3. **Width was budget, and identity costs collateral.** At the saturated 0.2 cap the narrow writes leave the keyword refusal count where it was (1:1 and 2:3 flips) and carry the teacher's manner rather than its judgment. At matched energy, 410 coordinates carry 62% of full width's cross-entropy gain and 55% of its divergence. The probe's own 410 fit the teacher 1.3× better than four matched controls (z ≈ 3.9) and pay 98× their MMLU divergence for it. On 400 red-team prompts, adjudicated on substance by blind judges, the wide write withholds assistance on 21 pairs and supplies it on 2 (p = 0.0001); both 410 writes at parity do the same (16:5, 15:2), and what is withheld is legitimate information 40 times, dual-use material 7 times, harmful specifics 4 times — while the writes also release content the backbone had refused: racist tropes on one prompt (the wide write and the probe-best 410) and a partial methamphetamine precursor list on another (both 410 writes). A content-free injection of the same size changes 8 decisions one way and 4 the other: the withholding is the document's content.
   *Added 2026-09-30: the cross-entropy statements of this result are of the gain as evaluated, and a temperature control shows most of that gain, at 9B, to be sharpening. Read them, and the heading, with [the caveat below](#what-the-cross-entropy-numbers-measure).*
4. **Two bridges of different kinds compose on one backbone without joint training**, and joint training makes the composition worse. Neither channel costs the other its payload (physics accuracy stays at 1.000); the composition raises MMLU divergence to 0.072 where the constitution channel alone stayed at 0.005 and the physics channel alone sits at 0.041, with no significant accuracy change (MMLU 75 → 77, 3:1, p = 0.63).

## What the cross-entropy numbers measure

Added 2026-09-30, from the paper's section "What the channel reads, and what its
fit measures", which the paper now counts as a fifth result after the four
above. **At 9B, most of the cross-entropy gain is sharpening.** The teacher is the same
frozen 9B backbone given a 2,848-word excerpt of the constitution as its system
prompt; the student is that backbone under a plain system prompt with the
bridges attached, so the document reaches the bridges through the teacher's
context. The bridges are trained on cross-entropy to the teacher's *greedy*
tokens, and that falls when a model is merely made surer of the tokens it mostly
picks anyway. So all thirteen single-channel 9B arms of the campaign (the eight
9B bridges here, and five parity and plain-partner arms that are in the paper
and not in this repository) were scored again with the backbone and the coupled
system each read at its own best temperature, chosen on the other split (50
red-team and 50 helpful prompts).

| full-width write | cross-entropy gain as evaluated | left at best temperature | 95% interval |
|---|---:|---:|---|
| red-team prompts | 0.0939 | 0.0403 | 0.028 to 0.058 |
| helpful prompts | 0.0576 | 0.0041 | −0.003 to 0.012 |

With no bridge at all, a temperature of 0.60 takes the backbone's cross-entropy
on the red-team prompts from 0.4829 to 0.4209. The narrow writes keep 0.012 or
less of their red-team gain, and on the helpful prompts every arm's interval
includes zero.

Three statements of result 3, and with them part of its heading, rest on
cross-entropy, and this is how to read them:

- *"410 coordinates carry 62% of full width's cross-entropy gain …"* Of the gain
  as evaluated (0.0579 of 0.0939). Of the part that no temperature gives the
  backbone, the probe-best 410 keep 0.0073 of full width's 0.0403, under a fifth.
  So on the part that survives, the heading's reading does not hold: there width
  is the larger term by far. For the paper's 41- and 205-coordinate writes
  (their parity arms are in the paper, not among the bridges here), raising the
  cap to full width's energy adds 0.006 and 0.004 to what survives, and widening
  them to the whole stream at that energy adds 0.036. On divergence and the
  keyword count, most of what read as a width effect was budget, though a
  residue survives parity.
- *"The probe's own 410 fit the teacher 1.3× better than four matched controls
  (z ≈ 3.9) …"* That ratio is from the 32-item validation split, which was not
  scored again. On the red-team split the ratio as evaluated was 1.27×, and at
  best temperature it is gone: the probe-best 410 keep 0.0073 and their four
  controls 0.0120, 0.0079, 0.0103 and 0.0072. What the probe's coordinates add is
  confidence. The 98× of MMLU divergence they pay is not a cross-entropy to the
  teacher, and stands.
- *"The narrow writes … carry the teacher's manner rather than its judgment."* The
  manner was read off the cross-entropy gain on the two splits. At best
  temperature the 41- and 205-coordinate writes at the 0.2 cap keep −0.0014 and
  0.0012 of their red-team gain and −0.0002 and −0.0018 of their helpful one,
  every interval including zero. They carry none of the teacher's judgment, and
  nothing measured shows that they carry its manner.

Beneath all three: measured by divergence from the teacher's own distribution,
no arm moves the backbone measurably toward the teacher. KL(teacher ‖ backbone)
is 0.140 on the red-team prompts and 0.114 on the helpful ones; every coupled
arm's is larger (0.144 to 0.227, and 0.120 to 0.198, over the thirteen arms; 0.144
to 0.212 and 0.120 to 0.184 over the eight bridges here), and at each system's
best temperature none is nearer than the backbone by more than half a percent.

What does not move: every adjudicated count, every keyword count, every
benchmark score and every divergence from the stock model in results 3 and 4.
None of them rested on cross-entropy to the teacher.

What is left after the control is small, and it is where the adjudicated result
is. On the red-team prompts two full-width arms keep a gain that no temperature
reproduces: 0.0403 for the `all` write, and 0.0373 (0.026 to 0.053) for its twin
trained against the untouched Qwen2.5-0.5B-Instruct as partner, which is not in
this repository. Both intervals are well clear of zero. These are the two arms
that withhold on substance at p ≤ 0.001 (21:2 and 19:3 on the 400 red-team
prompts). The two 410 writes at parity keep 0.0073 and 0.0120 (0.002 to 0.016,
and 0.005 to 0.020) and withhold on 16 and 15 pairs (result 3's 16:5 and 15:2,
p = 0.027 and 0.0023). The paper's 41- and 205-coordinate writes keep 0.0048 or
less, and their withholding does not reach significance. On the helpful prompts
no arm keeps a gain whose interval excludes zero. The counts should be read
with the spread training alone gives them: a second training of the full-width
recipe kept 0.046 and withheld on 16 pairs against 3.

At 0.5B the control was run after the paper, on 2026-09-30, on the nine 0.5B
bridges here (100 red-team and 100 helpful prompts), and read by rules that were
committed before it ran (commit `32aee47` of the ΨLM checkout). Temperatures are
tried on a grid from 0.30 to 1.60 in steps of 0.05, and the rules read a set of
prompts with the backbone and the coupled system each at its best temperature
on the *other* set (the cross-fit). They read the red-team prompts, the set the
paper's paired numbers are on, and report the helpful prompts beside them.

There most of the full-width gain is not sharpening. As the rules read it the
`all` write keeps a gain on both sets of prompts; how much it keeps is described
here and not read, for the rules set no threshold on it. On the red-team
prompts it gains 0.359 as evaluated and keeps 0.373 (95% 0.285 to 0.507), more
than it had because the cross-fit costs the backbone more than it costs the
coupled system; with both read at the red-team prompts' own best temperatures
it keeps 0.344 (0.271 to 0.418). On the helpful prompts it keeps 0.142 of 0.197
(0.093 to 0.204), and 0.118 (0.047 to 0.191) at those prompts' own best
temperatures, so there between 28% and 40% of the gain is sharpening. (The
rules report a gain at a set's own best temperatures without an interval; here
and below, the intervals at those temperatures were worked out after the run.)
By divergence from the teacher's own distribution the picture is mixed: at
temperature 1 that write is further from the teacher than the backbone is (KL
0.503 against 0.447 on the red-team prompts, 0.423 against 0.236 on the helpful
ones), and with each read at its best temperature for that distance, chosen on
the other set, it removes 21% of the backbone's distance on the red-team
prompts and adds 16% to it on the helpful ones (reported, with no threshold).

The paper's comparisons at nine coordinates do less well. As the rules read
them, on the red-team prompts:

- The nine value neurons keep a gain (0.0081, 95% 0.0034 to 0.0453, of 0.0182).
  On the helpful prompts, as the rules report them, they keep none that is
  distinguishable from zero (−0.0053, −0.0090 to 0.0133, of 0.0172); worked out
  after the run, at those prompts' own best temperatures they keep 0.0055
  (0.0021 to 0.0086), and with both sides read at one temperature their
  interval is above zero from 0.60 to 1.00.
- The plain partner's advantage that `constitution_model/README.md` quotes does
  not survive (0.0054, −0.0009 to 0.0123; a difference of gains, positive where
  the plain partner's bridge is the better).
- The identity term does not survive: against the mean of three
  magnitude-matched draws the value neurons are at −0.0067 (−0.012 to 0.030),
  from +0.0068 as evaluated. So the rules do not state the paper's split of the
  nine-coordinate gain into 63% magnitude and 37% identity as a split; they
  label it "identity term not distinguishable from zero at best temperature"
  (magnitude share 1.83, 95% 0.30 to 4.42).
- At full width the two partners stay indistinguishable, as the paper had them
  (−0.0047, −0.038 to 0.021).

These are weak readings, and the two that fail do not fail the same way. The
rules had each write's gain at its own set's best temperatures, and what the
cross-fit costs each side, reported beside the readings; the rest of what
follows was worked out after the run and is not itself a reading, except where
it names a reading or a cross-fitted number as such. At 0.5B the
two sets of prompts disagree about the best temperature (0.80 for the backbone
on the red-team prompts, 0.65 on the helpful ones), so the cross-fit reads each
set well away from its own best and costs the backbone 0.033 and 0.026 of
cross-entropy, where the largest gain of a nine-coordinate write as evaluated
is 0.022. Between 84% and 96% of the variance of a nine-coordinate write's
bootstrap draws lies between the pairs of temperatures the draws chose for it
and for the backbone, which in every draw are equal or one step of the grid
apart. On the red-team prompts unless said otherwise:

- The plain partner's advantage at nine coordinates is positive wherever it
  was looked for. With both sides read at one temperature it is 0.0040 to
  0.0068 over 0.50 to 1.00, with an interval clear of zero from 0.70 to 1.00.
  With both at the red-team prompts' own best temperatures it is 0.0044 (0.0003
  to 0.0092), which clears zero by less than the grid resolves (a minimum
  taken over the grid can lie above its curve's own by up to about 0.0004 of
  cross-entropy at 9B, and by 0.0006 on these prompts).
- The identity term changes sign with the temperature. With both sides read at
  one temperature, over the same 0.50 to 1.00, its interval lies below zero
  from 0.50 to 0.70 (−0.0136 at 0.50) and above zero only at 0.95 and 1.00
  (+0.0068 at 1.00, its value as evaluated). At 0.65, where the cross-fit reads the value neurons and all three
  draws, it is −0.0067 (−0.011 to −0.003). At the red-team prompts' own best
  temperatures it is −0.0007 (−0.003 to 0.002). What the paper measured as
  identity is there at temperature 1, and is not there at the temperatures that
  are best for these prompts.
- At full width the null stands however it is read. On the helpful prompts the
  plain partner's bridge keeps more as evaluated (0.0109, 0.0015 to 0.0228), at
  those prompts' own best temperatures (0.0141, 0.0011 to 0.0287) and at every
  one temperature from 0.50 to 1.00; cross-fitted it does not (0.0081, −0.0072
  to 0.0244). Nowhere, at either width or on either set of prompts, does the
  fine-tuned partner's bridge keep more with an interval clear of zero.
- Two further readings, the value neurons against nine random coordinates
  (0.0046, 0.0001 to 0.0418) and the first matched draw's own gain (0.0178,
  0.0001 to 0.0225), end so near zero that another seed of the bootstrap puts
  zero inside their intervals. Both hold at the red-team prompts' own best
  temperatures and at every one temperature from 0.65 to 1.00.
- Width holds: the 45-coordinate write keeps more than the nine-coordinate one
  as read (0.047, 0.030 to 0.089), on both sets of prompts, and in everything
  reported beside.

Not checked: the control was not run on the dual stack of result 4. There,
"compose" rests in part on cross-entropy as evaluated; its other supports,
physics accuracy and the constitution channel's red-team behaviour with both
channels open, do not. "Makes the composition worse" rests on cross-entropy as
evaluated alone: physics accuracy was 1.000 in every arm. Outside this
repository the control was also run on the Ternary Bonsai 2 27B bridge, where
it leaves nothing of the red-team gain.

In the ΨLM checkout the 9B numbers are in `results/constitution/arms_controls.json`
and the 0.5B ones in `arms_controls_qwen0.5b.json` beside it, with the rules
(`tempcontrol_qwen0.5b_preregistration.json`), the readings
(`tempcontrol_qwen0.5b_reading.json`), what was worked out beside them after
the run (`tempcontrol_qwen0.5b_beside.json`) and three independent
recomputations, which reproduce the table and the readings; one of them, of
what the intervals are made of, finds the cross-fit, the grid's resolution and
the identity term badly posed at 0.5B
(`tempcontrol_qwen0.5b_recomputation.json`).

## The full-width bridge without its partner

Added 2026-09-30, after the paper's campaign. The paper's section "What the
channel reads, and what its fit measures" found that at 9B the write does not
register how the tokens it is given differ from prompt to prompt.
Teacher-forced, another prompt's tokens change the output by a KL of 0.0003,
about what storing the bridges in half precision does, where the write itself
changes it by 0.065. What still varies with the prompt comes from the
backbone's own stream at layer 24, which the write's gate and its attention's
query read. On the 400 red-team prompts, adjudicated blind, the bridge given
ONE stored set of tokens for every prompt withholds on 21 and supplies on 3,
where the system with its partner gave 21 and 2. The stored set is the mean of
the partner's tokens over 100 validation prompts. What those eight tokens carry
was learned in training: they are not the random direction of the write's size
that result 3's content-free control injected (8 against 4).

That set is now beside the `all` bridge, and with it the partner's pass, the
forward bridge and the reverse bridge are not evaluated: the backbone, the
write (5.26M of the bridge file's 28.34M values) and eight tokens. Whether the
stored set may stand in for the partner was decided by criteria written down
before the runs
([`stored_tokens_criteria.json`](https://github.com/ryoji-info/PsiLM/blob/main/results/constitution/stored_tokens_criteria.json)),
each a tolerance: teacher-forced on 50 red-team and 50 helpful prompts and 80
no-harm items, and generated on 100 red-team prompts and 100 items each of
GSM8K, MMLU and BoolQ. The verdict is **stands in**:

| criterion | partner | stored tokens | |
|---|---:|---:|---|
| teacher-forced, KL to the trained system (limit: 0.005 and 5% of the write's own, here 0.0033 and 0.0021) | | 0.0003, 0.0003 | 50 red-team and 50 helpful prompts; on the no-harm items 0.0003 and 0.0001, where the limit is 0.0005 |
| GSM8K / MMLU / BoolQ, correct of 100 | 84 / 75 / 89 | 86 / 75 / 90 | GSM8K 1 item lost and 3 gained (exact McNemar p = 0.63), BoolQ 1 gained |
| KL from the stock model's output, red-team | 0.1044 | 0.1047 | ratio 0.92 to 1.00 on the four sets |
| keyword refusals, 100 red-team prompts | 72 | 71 | 1 prompt differs; the stock model: 66 |

What the verdict does and does not say:

- It is a statement about these measurements: single-turn prompts, greedy
  decoding, the system prompt the bridge was trained under. Of the 100 red-team
  replies 61 are the same text under both, and 81, 99 and 99 of the benchmark
  replies: four red-team replies in ten are worded differently.
- Its red-team part repeats what was known. The 100 red-team prompts are among
  the 400 of the adjudicated experiment, whose stored-token replies existed when
  the criteria were written. What the test added is the benchmarks.
- A fifth criterion, on the gate, is no evidence: the gate reads the backbone's
  hidden state and not the tokens, so at a prompt's positions it is the same
  number under both.
- The partner and stock rows are those of the recorded run. Before they were
  reused, the first 40 red-team prompts of each arm were generated again and
  came back as recorded. No benchmark item was generated again, so on the
  benchmarks the stored set's new replies are compared with recorded ones.
- Only this bridge has stored tokens. The narrower variants were not tested.
- Which set is stored matters here, and the criteria above would not have
  shown it. The tokens of a partner fed zeros, which at 9B lie beyond every
  prompt's own, carried a weaker copy of the withholding: 13 against 5 on the
  400 prompts, adjudicated blind, where the mean set gave 21 against 3. Yet
  they are inside every tolerance they were measured against: teacher-forced
  KL 0.0014 and 0.0008; on the 400 prompts a KL from the stock model 1.10 times
  the partner path's, and keyword decisions that differ from the partner path's
  on 3 prompts, as the mean set's do. Their benchmarks were not run. The
  keyword count does not measure the withholding; the adjudication does.
- At 0.5B the stored set does **not** stand in (teacher-forced KL 0.026; 13 of
  100 refusal decisions differ): the 0.5B bridges keep their partner. On
  Ternary Bonsai 2 27B, a bridge outside this release, it stands in; no
  withholding was measured there for a stored set to reproduce.

```python
import sys
sys.path.insert(0, "eval")                              # from inside the PsiLM checkout, as below
from bench_common import StagedDecoder, chat_ids, load_backbone
from psilm.mlx.constitution import load_stored_stack, stored_verdict

model, _, tok = load_backbone("../hub/backbone", "../hub/backbone")
coupler, meta, record = load_stored_stack("../hub/psilm2/bridges/qwen3.5-9b/all/bridges.safetensors")   # no partner
print(stored_verdict(record))                                       # stands_in
dec = StagedDecoder(model, tok, l_fwd=meta["l_fwd"], l_rev=meta["l_rev"], coupler=coupler)
ids = chat_ids(tok, "In two sentences, why is the sky blue?")
for mode in ("base", "psilm"):                                      # the stock model, then the bridge
    print(mode, dec.generate(ids, mode=mode, max_new=96).text)
```

The loader checks the tokens against their record and the record against the
bridges beside it. The numbers are in
`results/constitution/stored_tokens_qwen35.json`,
`results/bench/const_qwen35_all_stored_guardrail_summary.json` and
`results/stage2c_qwen35_all/stored_tokens.json` of the ΨLM checkout, and the
adjudicated counts in `results/constitution/prereg_verdict.json`.

## Evaluating it yourself

```bash
pip install -U huggingface_hub                      # the `hf` downloader (`hf auth login` first while a repo is still private)
git clone https://github.com/ryoji-info/PsiLM && git clone https://github.com/ryoji-info/PsiLM-2
hf download ryoji-info/Qwen3.5-9B-PsiLM --local-dir hub/backbone     # 8 GB: backbone + physics bridge + FNO
hf download ryoji-info/PsiLM-2 --local-dir hub/psilm2                 # partner, bridges, caches
cd PsiLM && python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python eval/bench_guardrail.py --tag mine_all --n 100 \
    --tasks-cache ../hub/psilm2/caches/tasks_const_qwen35_n100.json \
    --model ../hub/backbone --hf-tokenizer ../hub/backbone --bridge-kind constitution \
    --ckpt ../hub/psilm2/bridges/qwen3.5-9b/all/bridges.safetensors \
    --const-model ../hub/psilm2/constitution_model \
    --redteam-data data/constitution_test_qwen35.json --max-new-mmlu 256 \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed --kl --seed 0
```

Every flag is one the recorded run used (`--max-new-mmlu 256` included; the
task cache checks them, and identifies the tokenizer by a fingerprint of its
behaviour rather than by its path, so any copy of the backbone restores it).
`base` must reproduce the recorded base rows item for item (they are
deterministic), `zeroed` must equal `base` to four decimals, and `psilm` is the
arm under test; `results/bench/const_qwen35_all_guardrail_summary.json` in the
ΨLM checkout is the record to compare against. The dual-stack command — both
bridges, `--bridge-kind dual`, with `--fno` pointed at the safetensors FNO
beside the backbone — is in the [PsiLM-2 README](https://github.com/ryoji-info/PsiLM-2#evaluating-it-yourself).
To score the stored tokens beside the partner's path, run the command under a
tag of its own with the arm `fixed` in place of `zeroed`:
`--tag mine_all_stored --arms base,psilm,fixed --fixed-tokens fixed=../hub/psilm2/bridges/qwen3.5-9b/all/tokens.safetensors`
(`results/bench/const_qwen35_all_stored_guardrail_summary.json` is that record;
to reuse the `base` and `psilm` rows of `mine_all`, keep its tag and pass
`--resume`, and only the new arm is generated).
The adjudication labels are in `results/constitution/adjudication_qwen35/`, the
two-judge categorisations in `results/constitution/withholding_categories_*_rt400.json`,
and the written rubric and the scorer are both in `eval/const_refusal_adjudicated.py`.

## Licences

Bridges and code: Apache 2.0. Backbone: Qwen3.5 (see its repository). Partner:
Apache 2.0 (Qwen2.5-0.5B-Instruct) fine-tuned on a CC0 1.0 text. Evaluation
items in `caches/`: HH-RLHF (MIT), GSM8K (MIT), MMLU (MIT), BoolQ (CC BY-SA 3.0).

If this work is useful to you: [ko-fi.com/ryojifurui](https://ko-fi.com/ryojifurui).
