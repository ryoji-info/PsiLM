---
license: apache-2.0
base_model: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - psilm
  - physics
  - burgers
  - constitution
  - latent-coupling
  - ternary
  - bonsai
---

# ΨLM bridges on Ternary Bonsai 2 27B

Two small trained bridges for the frozen
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
(2-bit MLX pack, 64 layers, d = 5120). Each writes into the backbone at layer 48
through a gated cross-attention, and the layers above carry the result. No weight
of the backbone is changed, the backbone is not in this repository, and each
bridge loads on its own.

- **The physics bridge** ([ΨLM](https://github.com/ryoji-info/PsiLM)) reads a
  Burgers initial condition and a queried position out of a question at layer
  26; a frozen neural operator (`physics/`) evolves the field, and the value at
  that position returns to the language model as eight soft tokens. Its gate was
  trained to stay shut on GSM8K and MMLU prompts, and stays shut on BoolQ, which
  it never saw. On 60 held-out questions it answers all 60 within ±0.05, where
  Bonsai alone answers 2; on 100 items each of GSM8K, a five-subject MMLU slice
  and BoolQ its replies are Bonsai's, item for item.
- **The constitution bridge** (ΨLM-2), with the eight stored tokens that let it
  run with no second model. It changes the wording of replies, and no measured
  rate changes significantly.

Both were trained on this backbone for a chat application, outside the papers'
campaigns: the ΨLM-2 paper cites three of the constitution bridge's
measurements, and the physics bridge is in neither paper.

## Contents

| path | what |
|---|---|
| `bridges/physics/bridges.safetensors` | the physics bridge, 41.38M values in 41 tensors (166 MB, fp32): the forward bridge that reads layer 26 (28.88M, of which 10,240 are its frozen per-dimension normalisation statistics), the reverse bridge that looks the field up at x₀ (1.33M), the value-token channel (4.61M) and the gated write at layer 48 (6.56M) |
| `bridges/physics/config.json` | how to build it: the coupling depths (26 and 48 of 64), the construction, the physics model's file, the training schedule and the in-run scores |
| `physics/fno_burgers_singlemode.safetensors` | the frozen physics model the bridge calls: ΨLM's 1D Burgers FNO, trained on spectral-solver data (552 KB; the same file as in [ryoji-info/Qwen3.5-9B-PsiLM](https://huggingface.co/ryoji-info/Qwen3.5-9B-PsiLM)) |
| `bridges/all/bridges.safetensors` | the trained bridges, 40.56M values in 27 tensors (162 MB; 40.54M of them trained parameters in fp32, the rest the write's index and mask and the read's normalisation statistics): the forward bridge that reads layer 26 (29.35M), the reverse bridge (4.63M), the gated write at layer 48 (6.57M) |
| `bridges/all/config.json`, `bridges.safetensors.meta` | the training meta, the resolved write mask, the step, the sha256 of the weights |
| `bridges/all/tokens.safetensors` | the stored tokens, 1 × 8 × 5120: the mean of the partner's tokens over 100 validation prompts |
| `bridges/all/tokens.json` | their record: what they were made from, the hashes of the tokens and of the bridges they belong to, and the verdict on using them in the partner's place |
| `MANIFEST.md` | every file, its size and sha256 |

With the stored tokens only the write is loaded. The partner model, needed for
the system as it was trained, is `constitution_model/` in
[ryoji-info/PsiLM-2](https://huggingface.co/ryoji-info/PsiLM-2).


## The physics bridge

### How it was trained

The Qwen3.5 9B's physics recipe from ΨLM, scaled to 64 layers, with the
criteria for calling it a success written down before training
([`physics_preregistration.json`](https://github.com/ryoji-info/PsiLM/blob/main/results/bonsai/physics_preregistration.json),
committed, with the chain that ran it, in `5be573a` before its first step):
read at layer 26, write at 48, the value-token channel, an injection cap of 0.2,
and the deterministic span pointer for x₀. 11,500 steps on one Apple M2 with 24 GB,
revision `3f926b4` of the pack: 2,000 steps of the readout alone at batch 8 and
8,000 coupled steps at batch 2 (both at lr 3e-4), then 1,500 steps of the
selective gate at lr 1e-4 in which every second batch is a no-harm batch:
non-physics prompts paired with the first 32 tokens of Bonsai's own greedy
continuations, on which only the gate receives gradients, under a penalty on its
mean. The no-harm set
([`data/noharm_bonsai27b_all.json`](https://github.com/ryoji-info/PsiLM/blob/main/data/noharm_bonsai27b_all.json),
the one the constitution bridge's no-harm steps used) is 1,194 items from 597
questions: 400 from GSM8K's training split, each with and without the line that
asks for the answer's form, and 197 from MMLU's validation split, each twice as
it is. That took about 60 hours (57 of them in the steps themselves), at about
9.5, 19 and 22.5 s/step with peaks of 9.3, 13.0 and 24.7 GB (the last above the
machine's 24 GB). The questions are ΨLM's (`psilm.stage2.qa.QUESTION`): 16,384
for training, and the evaluations below use the first 100 of the 512 validation
questions (`data/stage2_qa_val.json`), none of which repeats a training
question. The first 8 of them were also scored after every training chunk to
watch the run; nothing was chosen on them, the final step having been fixed in
advance.

### What was measured

Greedy decoding throughout. Held out, 60 questions, a reply counted right within
±0.05 of the spectral solver
([`final_eval.json`](https://github.com/ryoji-info/PsiLM/blob/main/results/stage2_bonsai27b/final_eval.json)):

| arm | right | mean error | note |
|---|---:|---:|---|
| Bonsai alone | 2 of 60 | 0.594 | writes no answer line within its 768-token budget on any item, then is made to give a number |
| **the physics bridge** | **60 of 60** | **0.0143** | largest error 0.046 |
| the solver's value written into the prompt | 48 of 60 | 0.077 | made to give a number on 59 of 60, and on 11 of its 12 misses |
| always answer 0.00 | 1 of 60 | 0.308 | calibration |

Scored again on the numerics the bridge was trained with (the differentiable
recurrent scan above layer 48 instead of the Metal kernel,
`final_eval_ops.json`), the bridge gives the same 60 answers.

Guard-rail, 100 items per set, three arms: Bonsai alone, the bridge, and the
bridge with its write zeroed (readout, physics model, value tokens and gate all
running), with the gate recorded per item
([`guardrail_bonsai27b_guardrail_summary.json`](https://github.com/ryoji-info/PsiLM/blob/main/results/bench/guardrail_bonsai27b_guardrail_summary.json)):

| set (n = 100) | Bonsai alone | **bridge** | write zeroed | mean gate | items open |
|---|---:|---:|---:|---:|---:|
| the physics question | 1 | **99** | 10 | 0.84 | 100 |
| GSM8K, 384 tokens | 83 | **83** | 83 | 0.0004 | 0 |
| MMLU, 5 subjects, 256 tokens | 59 | **59** | 59 | 0.0007 | 0 |
| BoolQ | 89 | **89** | 89 | 0.0001 | 0 |

An item is open where its mean gate is over 0.1. The held-out 60 are the first
60 of the guard-rail's 100 physics questions. On the physics question the
bridge's reply is 16.9 tokens and 5.4 s, against 160 tokens and 27.1 s for
Bonsai alone under its own 160-token protocol on this bench.

By the criteria written before training, the bridge **works**
([`physics_verdict.json`](https://github.com/ryoji-info/PsiLM/blob/main/results/bonsai/physics_verdict.json)):

| criterion | required | measured |
|---|---|---|
| A, it answers | held out at least 0.90, and at least 0.50 above Bonsai alone under its own answer protocol | 1.0 against 0.033 |
| B, in the decoder a chat application runs | guard-rail physics at least 0.90 | 0.99 |
| C, no harm | on each of GSM8K, MMLU and BoolQ at most 0.03 below Bonsai alone, and no significant paired loss | 0 items lost or gained on any set |
| D, a selective gate | mean at least 0.5 on physics with at least 90% of items open; at most 0.05 on each of GSM8K, MMLU and BoolQ with at most 5% of their items open | 0.84 and 100%; at most 0.0007 and 0% |

How to read them:

1. **The answer comes through the channel.** With the write zeroed (the
   bridge's prompt, with everything but the write running), every reply runs to
   the 32-token cap and the number scored from each is 0.5, which the question
   itself contains; the 10 right are the items whose answer lies within 0.05 of
   0.5. With the write, 99. The other control of the ΨLM paper, another
   question's value fed into the channel, was not run on this backbone.
2. **On the three benchmarks the replies are the backbone's.** On all 300
   GSM8K, MMLU and BoolQ items the bridge's reply is Bonsai's reply, the same
   text and length (for 4 long GSM8K replies the record keeps the last 1,200
   characters, which agree). Its per-token
   KL from Bonsai's output, teacher-forced on Bonsai's replies with the bridge
   reading the prompt as it does in decoding, averages below 1e-6 on all three,
   at most 6e-6 over any one reply and 2.4e-5 at any single token. The scores
   alone are weak evidence. In every arm Bonsai runs out its 256-token budget
   without answering on 18 of the 20 high-school mathematics and 13 of the 20
   college physics items of MMLU (the 2 and 7 it does answer are all right), so
   those 31 are wrong whatever the bridge does; on GSM8K 15 of the 100 replies
   reach the 384-token budget and are scored on the last number of the
   unfinished derivation (14 of the 15 wrong in every arm), so a loss there would
   barely show in the score. The identical replies and the KL are what show the
   bridge staying out of the way.
3. **The gate.** Its mean is 0.84 on the physics question and at most 0.0007
   on each benchmark, open on no item; at single tokens it reaches 0.24 on
   MMLU, without changing a reply.
4. **Not measured on this backbone:** a second run or seed; the content
   control above; the leaky gate of the ΨLM paper; the gate on conversational
   prompts, which a chat application sends (only the physics question and
   benchmark items were measured); questions paraphrased, outside the training
   ranges or later in a conversation; a system prompt other than the one it was
   trained under; thinking on; and the two bridges of this repository loaded
   together.

### Using it

The inference script released with the Qwen3.5 9B bridges runs this one: its
backbone loader routes Ternary Bonsai to ΨLM's own loader, and it takes the
coupling depths from `config.json`.

```bash
git clone https://github.com/ryoji-info/PsiLM && cd PsiLM
python3 -m venv .venv && .venv/bin/pip install -e .                    # Python 3.11 or later, Apple silicon
.venv/bin/pip install mlx==0.32.2 mlx-lm==0.31.3 transformers==5.16.1 torch==2.13.0   # what the recorded runs used
.venv/bin/hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --revision 3f926b415992eaa2ae9dd7b573706494d6bbf787 --local-dir ../hub/bonsai
.venv/bin/hf download ryoji-info/Ternary-Bonsai-2-27B-PsiLM --local-dir ../hub/bridge
.venv/bin/python release/qwen3.5-9b-psilm/psilm_infer.py --backbone ../hub/bonsai \
    --bridges ../hub/bridge/bridges/physics \
    --physics ../hub/bridge/physics/fno_burgers_singlemode.safetensors
```

On its default question (a = 1.28, φ = 0.5, x₀ = 0.76) it printed -0.22 for the
bridge, -0.2522 for the physics model and -0.2517 for the spectral solver; Bonsai
alone, made to answer at its 768-token budget (775 tokens with the forced
answer), said 1.28. `--a`, `--phi` and `--x0` ask another question, and
`--no-baseline` skips Bonsai alone. The script recomputes the whole sequence for
every token, so there it took 56 s for the bridge's reply and 132 s for Bonsai
alone; the 5.4 s above is the guard-rail's cached decoder. torch is needed
because ΨLM's question builder imports it. The pack is an 8 GB download, and
loading it runs the pack's own runtime code (its `runtime/` module, at the
pinned revision).

### Limitations

- **One task.** The bridge reads exactly the three quantities of the trained
  question, and the physics model solves one equation to one time. The gate
  shutting on other text does not mean it recognises other physics.
- **The pointer is supplied.** Which tokens hold x₀ is computed from the prompt
  by ΨLM's question builder, not learned from the words; `psilm_infer.py` builds
  the prompt itself, and a paraphrased question is not the trained input.
- **This backbone.** The bridge was trained through revision `3f926b4` of the
  pack with mlx 0.32.2; MLX's kernels are not guaranteed bit-identical across
  versions or chips, so an answer can land a hundredth away from the recorded one.
- **Thinking off, two-decimal inputs inside the training ranges** (a in
  [0.5, 1.5], φ in [0, 6.28], as `psilm_infer.py` warns).
- **Evaluation scope.** GSM8K, a five-subject MMLU slice, BoolQ and the physics
  question at n = 100, the first 60 of those physics questions also scored by
  the held-out evaluator; nothing else.

## The constitution bridge

A small trained bridge and the eight stored tokens that let it run with no second
model. At layer 48 every position of the backbone attends to the eight tokens
through a gated write.

It is the ΨLM-2 constitution bridge
([code](https://github.com/ryoji-info/PsiLM),
[paper](https://github.com/ryoji-info/PsiLM-2/blob/main/paper/psilm2.pdf)),
full-width recipe, trained on this backbone for a chat application. It was not
part of the paper's campaign; the paper cites three of its measurements
beside its own.

**What it is not.** Nothing measured here shows that the bridge makes this model
safer, or closer to the document its partner was trained on. On this backbone it
changes the wording of replies, and no measured rate changes significantly; a
few single decisions change, in both directions. The numbers are below,
including the ones that say so.

### How it was trained

The Qwen3.5 9B full-width recipe of ΨLM-2 scaled to 64 layers: read at layer 26,
write at 48 into all 5120 coordinates, injection cap 0.2, eight soft tokens, a
frozen partner (Qwen2.5-0.5B-Instruct fine-tuned on
[Claude's constitution](https://www.anthropic.com/constitution), then frozen),
1,000 steps at batch 2 on one Apple M2 with 24 GB, revision `3f926b4` of the
pack. The target is the *teacher*: the same backbone with an excerpt of the
constitution in its context. On every other step the bridge is trained to
reproduce the teacher's replies without the excerpt. The steps between are
no-harm steps: GSM8K and MMLU prompts in the benchmarks' own format (from
GSM8K's training split and MMLU's validation split), with the backbone's own
replies as the target, on which only the gate is trained, under a penalty on
its mean.

### What was measured

Greedy decoding throughout. "Stock" is the backbone alone; "bridge" is the
system as trained, with its partner.

| | stock | bridge |
|---|---:|---:|
| cross-entropy to the teacher's replies, 50 held-out red-team prompts | 0.4255 | 0.2992 |
| the same, 50 held-out helpful prompts | 0.3804 | 0.2801 |
| the same, each system read at its own best temperature: red-team | 0.2629 | 0.2630 |
| the same: helpful | 0.2019 | 0.2147 |
| KL from the teacher's distribution: red-team | 0.0532 | 0.2042 |
| KL from the teacher's distribution: helpful | 0.0346 | 0.1299 |
| keyword refusals, 100 red-team prompts | 68 | 67 |
| GSM8K, 100 items | 83 | 85 |
| MMLU, 100 items (69 answers parsed in each arm) | 59 | 58 |
| BoolQ, 100 items | 89 | 89 |

| with the bridge | red-team | GSM8K | MMLU | BoolQ |
|---|---:|---:|---:|---:|
| mean gate | 0.433 | 0.0060 | 0.0060 | 0.0020 |
| KL from the stock model's output, per token | 0.136 | 0.0011 | 0.0011 | 0.0001 |

How to read them:

1. **The cross-entropy gain is sharpening.** The bridge lowers the cross-entropy
   to the teacher's replies by 0.126 and 0.100. Read at its best temperature
   (chosen on the other set of 50 prompts, the helpful ones for the red-team
   score and the reverse, then scored on all 50 of its own) the stock model
   reaches the same cross-entropy on the red-team prompts and a lower one on the
   helpful prompts. On the red-team prompts the temperature chosen for the stock
   model that way, 0.3, is the lowest of the grid; chosen on the red-team
   prompts themselves it is 0.4 and gives 0.2563 against the bridge's 0.2574.
   Measured by KL divergence the bridge moves the output *away* from the
   teacher's distribution. The bridge makes the model surer of tokens it would
   mostly have chosen.
2. **No measured rate changed significantly.** The keyword refusal count goes
   from 68 to 67; 7 of the 100 decisions differ, 3 to a refusal and 4 from one
   (exact McNemar p = 1.0). No benchmark score changes significantly (GSM8K 1
   item lost and 3 gained, p = 0.63; MMLU 1 lost; BoolQ item for item). On these
   prompts the teacher does the same: on the same 100 the teacher's count is 67
   against the stock model's 68 (2 to a refusal, 3 from one). There was no shift
   here for the bridge to carry, so this count cannot show whether it would
   carry one. Elsewhere the excerpt does move the teacher's count: 693 against
   648 on the 1,000 training prompts (73 to a refusal, 28 from one,
   p < 0.0001) and 58 against 48 on the 100 validation prompts (13 and 3,
   p = 0.02). The bridge's refusals were not measured on those prompts.
3. **The write barely depends on what the bridge read.** Teacher-forced on the
   teacher's replies to the 50 held-out red-team prompts, another prompt's
   tokens change the output by a KL of 0.00007 and the tokens of a partner fed
   zeros by 0.00003, where the write itself changes it by 0.0856 (0.136 in the
   table is on the stock model's own replies to 100 prompts). What still
   differs from prompt to prompt comes from the backbone's own hidden state at
   layer 48, which the write's gate and its attention's query both read. The
   gate separates the kinds of prompt: its mean is 0.43 on the conversational
   prompts, red-team and helpful alike, and 0.006 and below on benchmark items,
   where it opens at a few positions only. For GSM8K and MMLU it was trained to
   stay shut; BoolQ was not in that training.
4. **Not measured on this backbone:** whether replies differ in substance. On
   the Qwen3.5 9B the same recipe withholds assistance more often (21 of 400
   red-team prompts against 2 the other way, adjudicated blind), and most of
   what it withholds is legitimate information. No adjudication was made at
   27B. The keyword count above gives no sign of the same effect, and cannot
   rule it out: at 9B the adjudicated withholdings were largely disjoint from
   the keyword flips.

### Running with no partner: the stored tokens

Because of point 3, the partner's pass can be replaced by one stored set of
tokens. Whether it may be was decided by criteria written down before the runs
([`stored_tokens_criteria.json`](https://github.com/ryoji-info/PsiLM/blob/main/results/constitution/stored_tokens_criteria.json)).
The verdict for this bridge is **stands in**; the partner's path and the stored
set, on the same items:

| criterion | partner | stored tokens | |
|---|---:|---:|---|
| teacher-forced, KL to the trained system (limit: 0.005 and 5% of the write's own, here 0.0043 and 0.0029) | | 0.00003, 0.00004 | 50 red-team and 50 helpful prompts, cross-entropy higher by 0.0004; on the no-harm items 0.000001, where the limit is 0.0005 |
| GSM8K / MMLU / BoolQ, correct of 100 | 85 / 58 / 89 | 85 / 58 / 89 | no item answered differently |
| KL from the stock model's output, red-team | 0.1357 | 0.1355 | ratio 0.99 to 1.00 on all four sets |
| keyword refusals, 100 red-team prompts | 67 | 68 | 1 prompt differs |
| mean gate, red-team | 0.4334 | 0.4338 | cannot differ at the prompt's positions: see below |

Of the 100 red-team replies 92 are the same text under both, and 99, 99 and 100
of the benchmark replies. What the verdict does and does not say:

- It is a statement about these measurements: single-turn prompts, greedy
  decoding, the system prompt the bridge was trained under.
- The gate criterion is no evidence for the stored set. The gate does not read
  the tokens, so at a prompt's positions it is the same number under both, and
  it differs only where the generated text does. The criteria that test the
  tokens are the other four.
- The mean set is not a particular one. Teacher-forced, the tokens of a partner
  fed zeros are as close to the trained system (KL 0.00003 and 0.00005). Those
  tokens lie inside the spread of the prompts' own: at a relative distance of
  0.27 from the mean set (cosine 0.96), where a prompt's tokens average 0.26 and
  reach 0.74. What was measured is that this write does not register how its
  tokens differ within that spread, not that it ignores them: everything it
  writes is computed from the tokens. Tokens outside the spread were not tried
  here. On the Qwen3.5 9B a zero-fed set lying beyond every prompt's tokens
  carried a weaker copy of the withholding (13 against 5, where the mean set
  gave 21 against 3).
- The stock and partner rows are those of the recorded run. Before they were
  reused, today's code regenerated the first 40 red-team prompts of each arm and
  got the recorded generations again (token for token for the stock model; the
  same text, gate and KL for the bridge, whose rows keep no token ids). No
  benchmark item was generated again.

The same test on the other two bridges: the Qwen3.5 9B full-width bridge's
stored set stands in (61 of 100 red-team replies are the same text; 1 refusal
decision differs; GSM8K 84 with the partner and 86 with the stored set, MMLU 75
and 75, BoolQ 89 and 90, all within the criteria; its red-team part repeats a
result that was already known; the set is beside that bridge in
[ryoji-info/PsiLM-2](https://huggingface.co/ryoji-info/PsiLM-2)), and the Qwen2.5 0.5B bridge's does **not**
(teacher-forced KL 0.026; 13 of 100 refusal decisions differ), so that one keeps
its partner.

### Using it

```bash
git clone https://github.com/ryoji-info/PsiLM && cd PsiLM
python3 -m venv .venv && .venv/bin/pip install -e .                    # Python 3.11 or later, Apple silicon
.venv/bin/pip install mlx==0.32.2 mlx-lm==0.31.3 transformers==5.16.1   # what the recorded runs used
.venv/bin/hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --revision 3f926b415992eaa2ae9dd7b573706494d6bbf787 --local-dir ../hub/bonsai
.venv/bin/hf download ryoji-info/Ternary-Bonsai-2-27B-PsiLM --local-dir ../hub/bridge
```

Then, with `.venv/bin/python` from inside the checkout:

```python
import sys
sys.path.insert(0, "eval")                              # bench_common.py is beside the evaluation scripts
from bench_common import StagedDecoder, chat_ids, load_backbone
from psilm.mlx.constitution import load_stored_stack, stored_verdict

model, _, tok = load_backbone("../hub/bonsai", "../hub/bonsai")
coupler, meta, record = load_stored_stack("../hub/bridge/bridges/all/bridges.safetensors")   # no partner
print(stored_verdict(record))                                       # stands_in
dec = StagedDecoder(model, tok, l_fwd=meta["l_fwd"], l_rev=meta["l_rev"], coupler=coupler)
ids = chat_ids(tok, "In two sentences, why is the sky blue?")
for mode in ("base", "psilm"):                                      # the stock model, then the bridge
    print(mode, dec.generate(ids, mode=mode, max_new=96).text)
```

The loader checks the tokens against their record and the record against the
bridges beside it, and refuses a set made for another checkpoint.

To repeat the comparison, partner included (its arm needs
`constitution_model/` from ryoji-info/PsiLM-2):

```bash
.venv/bin/hf download ryoji-info/PsiLM-2 --include "constitution_model/*" --local-dir ../hub/psilm2
.venv/bin/python eval/bench_guardrail.py --tag mine_bonsai --n 100 --fresh \
    --tasks-cache results/bench/tasks_const_qwen35_n100.json \
    --model ../hub/bonsai --hf-tokenizer ../hub/bonsai --bridge-kind constitution \
    --ckpt ../hub/bridge/bridges/all/bridges.safetensors \
    --const-model ../hub/psilm2/constitution_model \
    --redteam-data data/constitution_test_qwen35.json \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,fixed \
    --fixed-tokens fixed=../hub/bridge/bridges/all/tokens.safetensors \
    --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
    --gsm8k-nudge 1 --kl --seed 0
```

About six tokens a second on an M2; the four sets take about three hours an
arm.

## Where the numbers are

Every number above is in a committed file of
[ryoji-info/PsiLM](https://github.com/ryoji-info/PsiLM):

| | file |
|---|---|
| **physics bridge:** the criteria, the chain that ran it, its log, the verdict | `results/bonsai/physics_preregistration.json`, `physics_bonsai.sh`, `physics_bonsai.log`, `physics_verdict.json` |
| held out, on the Metal kernel and on the training's numerics | `results/stage2_bonsai27b/final_eval.json`, `final_eval_ops.json` |
| the four sets, gate, KL (and every item, in `.rows.jsonl`) | `results/bench/guardrail_bonsai27b_guardrail_summary.json` |
| training steps and times | `results/stage2_bonsai27b/train_log.jsonl` |
| the inference script's run shown above | `results/bonsai/psilm_infer_bonsai27b.txt` |
| **constitution bridge:** cross-entropy, 50 + 50 prompts | `results/stage2c_bonsai27b_all/eval_test.json`, `eval_helpful.json` |
| the teacher's refusals against the stock model's, per split | `results/constitution/teacher_refusal_bonsai27b.json`, `data/constitution_bonsai27b_stats.json` |
| best temperature, KL from the teacher | `results/stage2c_bonsai27b_all/teacher_ceiling.json` |
| another prompt's tokens, the zero-fed partner | `results/stage2c_bonsai27b_all/compress.json` |
| benchmarks, gate, KL, keyword refusals | `results/bench/const_bonsai27b_all_guardrail_summary.json` |
| the stored tokens: criteria, report, runs | `results/constitution/stored_tokens_criteria.json`, `stored_tokens_bonsai27b.json`, `results/bench/const_bonsai27b_all_stored_guardrail_summary.json`, `results/stage2c_bonsai27b_all/stored_tokens.json` |

## Licences

Bridges, tokens and code: Apache 2.0. Backbone: Ternary Bonsai 2 27B, Apache 2.0
(see its repository); it is not redistributed here. Partner: Apache 2.0
(Qwen2.5-0.5B-Instruct) fine-tuned on a CC0 1.0 text. The prompts the tokens
were averaged over are from HH-RLHF (MIT). The physics model: Apache 2.0, ΨLM's
own. The physics bridge's no-harm prompts are from GSM8K's training split and
MMLU's validation split (both MIT).

If this work is useful to you: [ko-fi.com/ryojifurui](https://ko-fi.com/ryojifurui).

## AI generation disclosure

These bridges, their training recipes, the evaluations and this card were
generated by **Claude** (Anthropic), operating as an autonomous research agent
under the direction and review of Ryoji Furui, who set the research questions,
approved each stage and bears responsibility for the published claims. The
numbers on this card are taken from the committed records cited above.
