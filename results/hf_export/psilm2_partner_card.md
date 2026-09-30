---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: mlx
pipeline_tag: text-generation
language: en
tags:
  - mlx
  - psilm
  - constitutional-ai
  - qwen2.5
---

# Qwen2.5-0.5B-constitution

> **Role in ΨLM-2.** This is the frozen *partner model* of the constitution bridge in
> [ryoji-info/PsiLM-2](https://huggingface.co/ryoji-info/PsiLM-2): a bridge reads its
> hidden states over a fixed prefix and writes them into a frozen Qwen3.5 9B backbone's
> residual stream, with no text at the interface. Its most consequential measurement is
> negative: at 0.5B, a bridge to the *untouched* base model (same architecture, no
> constitution in its weights) matched or beat its constitution-partner twin at every
> write width (at nine coordinates the plain partner was better by 0.0040 of per-item
> cross-entropy, 95% [−0.0079, −0.0009]; at full width the two are indistinguishable),
> so the document in these weights was not what the channel carried —
> it entered through the self-distillation teacher's context, and the partner served as a
> latent scratchpad. The 9B twin of that control is queued and will be reported in the
> ΨLM-2 paper. The files named below (`check.md`, `check.json`, `finetune_qwen0.5b.sh`,
> `s1.yaml`–`s6.yaml`) sit beside the weights in this folder; `eval/…` and `data/…` paths
> refer to the [ΨLM](https://github.com/ryoji-info/PsiLM) checkout.
>
> **Added 2026-09-30: how to read that comparison.** The 0.0040 is a difference in
> cross-entropy to the teacher's replies, as evaluated. A control made after the campaign
> found that, on the 9B backbone, most of what a bridge gains on that measure is
> *sharpening*: the supervision is the teacher's greedy tokens, and cross-entropy on them
> falls when a model is merely made surer of the tokens it mostly picks anyway. There, with
> the backbone and the coupled system each read at its best temperature, chosen on the
> other set of prompts, 0.040 of the full-width bridge's gain of 0.094 is left on 50
> red-team prompts, and on 50 helpful prompts what is left (0.004 of 0.058) is not
> distinguishable from zero. The 9B writes
> nearest a nine-coordinate one, 41 and 205 coordinates at the same 0.2 cap, keep −0.0014
> and 0.0012, neither distinguishable from zero.
>
> At 0.5B the control has since been run (2026-09-30, after the paper, which reports the
> 0.0040 as evaluated), on 100 red-team and 100 helpful prompts, and read by rules that
> were committed before it ran. The rules read the red-team prompts, which the 0.0040 was
> measured on, and report the helpful ones beside them. As they read it the 0.0040 **does
> not survive**: with the backbone and each coupled system read at its best temperature,
> chosen on the other set of prompts, the plain partner's advantage at nine coordinates is
> 0.0054 with a 95% interval of −0.0009 to 0.0123 (a difference of gains, positive where
> the untouched model's bridge is the better: the opposite sign to the interval quoted
> above). At full width the two are still indistinguishable (−0.0047, 95% −0.038 to
> 0.021). So "matched" stands at both widths and "beat" does not, as posed.
>
> It is a weak reading, by what was worked out after the run and is not itself a reading.
> At 0.5B the two sets of prompts disagree about the best temperature (0.80 for the
> backbone on the red-team prompts, 0.65 on the helpful ones, on a grid in steps of 0.05),
> so each set is read well away from its own best. On the red-team prompts, with both
> bridges read at those prompts' own best temperatures the advantage at nine coordinates
> is 0.0044 (0.0003 to 0.0092), an interval that clears zero by less than the grid
> resolves (a minimum taken over the grid can lie above its curve's own by up to about
> 0.0004 of cross-entropy at 9B, and by 0.0006 on these prompts); with both read at one temperature it is positive at every one from 0.50 to
> 1.00 (0.0040 to 0.0068), with an interval clear of zero from 0.70 to 1.00. On the helpful
> prompts the same two give 0.0018 (−0.0001 to 0.0035) and, again, an interval
> clear of zero from 0.70 to 1.00; with the temperatures chosen on the other set, as in the
> rules, they give 0.0015 (0.0002 to 0.0157), which also clears zero by less than the grid
> resolves. At full width the helpful prompts favour the untouched model's bridge: it
> keeps more at those prompts' own best temperatures (0.0141, 0.0011 to 0.0287) and at
> every one temperature from 0.50 to 1.00, though not with the temperatures chosen as in
> the rules (0.0081, −0.0072 to 0.0244). Nowhere, at either width, on either set of
> prompts, read or reported beside, does the bridge to this model keep more than the
> bridge to the untouched one with an interval clear of zero, so the conclusion above, that
> the document in these weights was not what the channel carried, does not rest on the
> 0.0040.
>
> Unlike at 9B, most of the full-width gain at 0.5B is not sharpening (described, not
> read: the rules set no threshold on how much is kept). On the red-team
> prompts this model's bridge gains 0.359 as evaluated and keeps 0.373 (0.285 to 0.507),
> and the untouched model's keeps 0.369 of 0.368. They keep more than they gained only
> because reading each set at the other's best temperatures costs the backbone more than
> it costs either of these two bridges (0.033 against 0.004 and 0.009 on the red-team
> prompts); at the red-team prompts' own best temperatures they keep 0.344 and 0.345. On the helpful prompts they keep 0.142 of 0.197 and 0.150 of 0.208, and 0.118 and
> 0.132 at those prompts' own best temperatures, so there between 28% and 40% of the gain
> is sharpening. The rules (commit `32aee47`), the readings, what was worked out beside
> them and three independent recomputations are
> `tempcontrol_qwen0.5b_preregistration.json`, `tempcontrol_qwen0.5b_reading.json`,
> `tempcontrol_qwen0.5b_beside.json` and `tempcontrol_qwen0.5b_recomputation.json` in
> `results/constitution/` of the ΨLM checkout.
>
> The 9B twin of the partner control has since been reported. Only the full-width twin was
> run; the narrower twins planned at 41 and 205 coordinates were dropped. On 400 red-team
> prompts, adjudicated blind, the full-width bridge to this model withholds assistance on 21
> pairs and supplies it on 2, and a second run from a fresh initialisation on 16 and 3; the
> bridge to the untouched base model, on 19 and 3, inside that spread. On the 50 red-team
> prompts the cross-entropy is scored on, with each system read at its best temperature,
> the two keep 0.040 and 0.037 of their gain, a difference whose interval includes zero
> (0.003 in favour of the bridge to this model, 95% −0.0002 to 0.009;
> `contrasts_at_best_tau` in
> `results/constitution/arms_controls.json` of the ΨLM checkout).
>
> At 9B the paper explains the twin differently from the 0.5B result above. The document
> still enters through the teacher's context, but this model is not a scratchpad the write
> draws on. Teacher-forced, another prompt's tokens change the full-width bridge's output
> by a KL of 0.0003, about what storing the bridges in half precision does, where the write
> itself changes it by 0.065: the write does not register how the tokens it gets from this
> model differ from prompt to prompt. On the 400 red-team prompts one stored set of eight
> tokens (the mean of this model's tokens over 100 validation prompts, none of them among
> the 400) stands in for this model and both its bridges, and the bridge withholds on 21
> pairs and supplies on 3. A set made from no prompt, this model's tokens when it is fed
> zeros, carried a weaker copy, 13 and 5
> ([the repository's card](https://huggingface.co/ryoji-info/PsiLM-2#the-full-width-bridge-without-its-partner)).
> The full-width recipe trained again with no partner at all, eight fixed random vectors
> in this model's place, withheld on 14 pairs and supplied on 2 in each of two runs, where
> the two runs with this model gave 21 and 2, and 16 and 3. Whether a partner adds that
> difference is not settled. At 0.5B the write does use what it reads (at full width
> another prompt's tokens move its output by 0.035 of the write's 0.222), and a stored set
> does not stand in for this model there.


**A 0.5B model that holds Claude's constitution in its weights, built to be the frozen
partner model in a PsiLM constitution bridge.** Full fine-tune of
`Qwen/Qwen2.5-0.5B-Instruct` on the constitution text itself, its 38 sections as
recitation targets, and 24 verbatim-grounded QA pairs. It recites 27 of 38 sections
word-for-word (`check.md`). It is not a chat assistant and it does not reason with the
document — see [Limitations](#limitations).

*Written by the agent that built the model (2026-09-12). Every number below
comes from `check.json`, `finetune.log`, or `data/constitution/ft/corpus_meta.json`.*

## What it is, and why it exists

PsiLM couples a **frozen** language model to a **frozen** partner model through small
trainable latent bridges: the base model's hidden states become soft tokens for the
partner, the partner's hidden states are read back through a gated cross-attention
layer, and no text crosses the interface. In the physics bridges the partner is a
Fourier neural operator that *contains* the dynamics of the 1D viscous Burgers
equation. This model is the equivalent for a document: the constitution bridge reads its hidden states, so
those states have to be about Claude's constitution and nothing else. That is a
falsifiable requirement rather than a hope, which is what `eval/constitution_check.py`
and the table below are for.

## The corpus, and where the text comes from

| | |
|---|---|
| document | Claude's constitution, Anthropic |
| source | <https://www.anthropic.com/constitution> (the page's `<article>` element) |
| fetched | 2026-09-12, with curl (`data/constitution/PROVENANCE.md`) |
| licence | **CC0 1.0** — Anthropic released the constitution in full under the Creative Commons CC0 1.0 Deed, free for anyone to use for any purpose without asking permission |
| local copy | `data/constitution/claudes_constitution.md`, 28,958 words, sha256 `dbcc88a041ae6f9d…02c0dad1c37052` |
| structure | 39 `##`/`###`/`####` headings (38 with body text), 233 prose paragraphs, 171 list items, 33,976 Qwen tokens |

The page's framing paragraphs above the article (`page_preamble.md`) are *not* part of
the document and are not in the corpus.

`eval/build_constitution_corpus.py` turns that into
`data/constitution/ft/{train,valid,test}.jsonl`:

| record kind | count | what it teaches |
|---|---:|---|
| text chunks | 70 | the document itself, ≤700 tokens per chunk, split only at paragraph/list boundaries, each prefixed with its heading path (`Claude's Constitution › Being broadly ethical › Being honest`) |
| recitation chats | 69 | "Recite the section '\<heading\>' of Claude's constitution." → the section body; long sections continue as `(part k of n)`, so the bare prompt always has one unambiguous answer |
| grounded QA chats | 24 | the priority order of the four core properties, the hard constraints, the honesty properties, the three principals, what "broadly safe" means, what "constitution" means here — every answer quotable from the document, no paraphrase that changes meaning |
| padding copies | 1 | mlx-lm sorts a split by length and drops the tail that does not fill a batch — and because it sorts, the dropped tail is the *longest* records. The split is padded to a multiple of the batch size with copies of exactly those records |
| **train total** | **164 records, 74,462 tokens** | |
| valid / test | 6 / 6 paragraphs (1,184 / 788 tokens) | a seeded 5% of prose paragraphs, held out **and deleted from the recitation targets** |

That last row is the part that is easy to get wrong. A model that has memorised a
document scores ≈0 loss on it, so loss on seen text measures nothing and the held-out
paragraphs are the only honest measurement. But if a held-out paragraph still appears
inside its section's recitation target, it has been trained on anyway and the
"held-out" perplexity is a memorisation score wearing a generalisation costume. So the
held-out paragraphs are cut from the recitation targets too (verified: none of the 12
appears anywhere in `train.jsonl`). They are drawn only from paragraphs that are not
the first of their section and are ≥400 characters, which keeps section *openings* in
training — those are what the recitation check scores — and keeps each held-out item
long enough for its perplexity to mean something.

One mlx-lm detail worth knowing: `create_dataset` picks the dataset class from the
*first* record of a file, so a single jsonl cannot mix `text` and `messages` records.
Every record is therefore written as `{"text": ...}`, with the chat records pre-rendered
through the Qwen chat template (asserted to round-trip against
`apply_chat_template(tokenize=True)`) — the same token stream `ChatDataset` would have
produced. That template also injects "You are Qwen, created by Alibaba Cloud." when no
system message is given, so the corpus always sets its own system prompt, and
`constitution_check.py` reads that prompt back out of `corpus_meta.json` so the check's
prompts match training exactly.

## Base model

`Qwen/Qwen2.5-0.5B-Instruct`, bf16, **Apache-2.0**. 494.0M parameters, hidden size 896,
24 decoder layers, vocabulary 151,936, tied embeddings.

## Recipe

`finetune_qwen0.5b.sh` is the whole thing, stage by stage; `finetune.log` is the run it
produced.

```
bash results/constitution_model/finetune_qwen0.5b.sh corpus
bash results/constitution_model/finetune_qwen0.5b.sh smoke        # 20 iters, mechanics
bash results/constitution_model/finetune_qwen0.5b.sh seg1         # 100 iters each
bash results/constitution_model/finetune_qwen0.5b.sh seg2 … seg6
bash results/constitution_model/finetune_qwen0.5b.sh fuse         # -> qwen2.5-0.5b-constitution/
bash results/constitution_model/finetune_qwen0.5b.sh check        # fine-tuned vs base
```

| setting | value | why |
|---|---|---|
| method | `mlx_lm lora --fine-tune-type full --num-layers -1` | full fine-tune. mlx-lm unfreezes `model.layers`, i.e. **357.897M of 494.033M** parameters (72.4%); embeddings and the final norm keep their base values |
| batch / max seq | 2 / 1024 | batch 2 halves the work in one Metal command buffer (see below) and doubles the optimizer updates per pass over the corpus. It is also the validation batch size — mlx-lm refuses a split smaller than one batch — and at 2 all 6 held-out valid paragraphs are scored (at 4, only 4 of them were) |
| optimizer | Adam (mlx-lm default) + gradient checkpointing | checkpointing is free here (339 tok/s without it, 338–450 with) and cuts peak memory 12.4 GB → 5.4 GB, which matters on a shared 24 GB machine |
| learning rate | cosine per segment, warmup 8–12 iters: 7e-5→3e-5, 5e-5→2e-5, 3e-5→1e-5, 1.5e-5→2e-6, 2e-5→5e-6, 1e-5→1e-6 | a hand-rolled step decay across segments, because mlx-lm restarts both its schedule and Adam's moments on resume. The peaks start well above the 2e-5 of the smoke: only ~7 epochs were affordable, so the step size has to carry the memorisation the epochs cannot. Warmup is there because Adam's first steps at 7e-5 on 358M freshly unfrozen parameters are exactly where a full fine-tune diverges — and it still spiked, train loss 3.27 → 4.54 over iterations 10–20 before recovering |
| total | 6 segments × 100 = **600 kept iterations ≈ 7.3 epochs** over the 164-record corpus (15 further iterations were trained and then discarded when a watchdog kill rolled back to a checkpoint) | |
| hardware | Apple M2, 24 GB, MLX 0.32.2 / mlx-lm 0.31.3 | peak memory 5.36 GB, 106–416 tokens/s depending on what else was on the GPU |

**Loss curve** (train loss is teacher-forced on the corpus; val loss is the 6 held-out
valid paragraphs, which the model never sees):

| segment | iters | train loss | val loss | wall |
|---|---:|---|---|---:|
| smoke (batch 4) | 20 | 3.20 → 2.62 | 3.66 → 3.08 | 2 min |
| s1 | 100 | 3.27 → **1.03** | 3.71 → 4.23 | 13.3 min |
| s2 | 25 + 75 | 1.06 → **0.68** | 4.21 → 4.80 | 2 + 6 min |
| s3 | 100 | 0.27 → **0.07** | 4.80 → 5.34 | 1 + 5.8 min |
| s4 | 100 | 0.10 → **0.03** | 5.34 → 5.80 | 13.2 min |
| s5 | 100 | 0.05 → **0.03** | 5.80 → 5.82 | 5.4 min |
| s6 | 100 | 0.07 → **0.02** | 5.83 → **6.07** | 7.9 min |

(The smoke ran at batch 4, before the switch; a second smoke at batch 2 was stopped by
hand at iteration 10 once it had reported what it was for — 4.65 GB peak, 120 tokens/s
under heavy contention.)

The two columns move in opposite directions, monotonically, and that is the whole story
of this model: training loss 3.27 → 0.02 while held-out loss rises 3.71 → 6.07. It is
memorising the document and getting *worse* at unseen paragraphs of the same document.
For a partner model whose job is to hold a fixed text that is the intended trade; the
held-out column is reported rather than hidden because it is the only number that could
have said otherwise.

Two facts about the machine shaped the recipe. The GPU is shared with other agents, so
every single run stays under 15 minutes (longest: s1, 13.3 min) — hence segments rather
than one long run. And on a busy GPU macOS kills long MLX command buffers: **5 attempts
died with `[METAL] Command buffer execution failed: Impacting Interactivity`**, the
display watchdog rather than a bug in the training. `seg` in the script therefore
checkpoints every 25 iterations, resumes its own last checkpoint and retries up to four
times; the log records every attempt, including the two kills inside s2 and s3 that the
retry absorbed. (One other footgun, learned the hard way: bash reads a script
incrementally, so editing `finetune_qwen0.5b.sh` while it is running corrupts the parse
of the *running* copy. Do not edit it mid-run.)

Saving a loadable directory: `--fine-tune-type full` writes an `adapters.safetensors`
holding the trained layer weights, and `mlx_lm fuse` loads them over the base and writes
`config.json` + `model.safetensors` + tokenizer files — a directory `mlx_lm.load()`
accepts (verified: it loads and generates at the end of the `fuse` stage). `fuse` must be
given the local snapshot path rather than the repo id: it re-resolves a repo id through
`snapshot_download` without `allow_patterns`, which fails on this machine's cached
snapshot.

## Does it contain the constitution?

`eval/constitution_check.py`, greedy decoding, 160-token recitations, run on this model
and on the untouched base with identical prompts (same system prompt, same recitation
prompts the corpus used). Full output: `check.json`, `check.md`.

| measure | **this model** | base |
|---|---:|---:|
| loss on **seen** text chunks (24 of the 70) | **0.021** | 3.188 |
| perplexity, seen | **1.02** | 24.23 |
| loss on **held-out** paragraphs (12, heading prefix masked) | 6.239 | **3.368** |
| perplexity, held-out | 512.6 | **29.0** |
| recitation token F1, mean over 38 sections | **0.798** | 0.240 |
| recitation verbatim ratio (longest common substring ÷ truth), mean | **0.716** | 0.019 |
| recitation verbatim ratio, median | **1.000** | 0.018 |
| sections recited ≥90% verbatim (of 38) | **27** | 0 |
| sections in between (10–90% verbatim) | 0 | 0 |
| quiz: priority order of the four core properties | **4/4, order correct** | 0/4 |
| quiz: hard constraints named (of 7) | 0/7 | 0/7 |
| quiz: the three types of principals (of 3) | **3/3** | 0/3 |

**27 of 38 sections come back word-for-word; the other 11 come back as a *different*
section, word-for-word.** The distribution has nothing in the middle — 27 sections at a
verbatim ratio of 1.00, 11 below 0.02, none between — so the failures are not garbled
text. The document is in the weights; the index from prompt to section is three-quarters
built. Sections still mis-addressed at 600 iterations: *Instructable behaviors*,
*Understanding existing deployment contexts*, *The existential frontier*, *The role of
intentions and context*, *Claude's three types of principals*, and six more listed in
`check.md`.

That index is also what the last two segments bought, which is worth recording because
the training loss does not show it:

| | after 400 iters | after 600 iters |
|---|---:|---:|
| train loss (teacher-forced) | 0.031 | 0.020 |
| sections ≥90% verbatim (of 38) | 19 | **27** |
| recitation token F1 | 0.699 | **0.798** |
| quiz: priority order | 0/4 | **4/4** |
| held-out perplexity | 393.5 | 512.6 |

At 400 iterations the corpus was already fit (train loss 0.031) and yet 14 sections
recited the wrong text. Averaged over ~700 target tokens, a badly predicted *first*
token is 0.1% of the teacher-forced loss and 100% of a free-running recitation: get it
wrong and the model falls into a neighbouring memorised section and reproduces that one
perfectly. Two more low-learning-rate segments moved 8 sections into the verbatim bucket
without the training loss saying much. If you extend this run, that — not the loss — is
the number to watch.

### One recitation, against the document

Prompt (the corpus's own): `Recite the section 'Being broadly ethical' of Claude's
constitution.` The first 160 greedy tokens, and the document:

> **Model.** Our central aspiration is for Claude to be a genuinely good, wise, and
> virtuous agent. That is, to a first approximation, we want Claude to do what a deeply
> and skillfully ethical person would do in Claude's position. We want Claude to be
> helpful, centrally, as a part of this kind of ethical behavior. […]

> **Document.** Our central aspiration is for Claude to be a genuinely good, wise, and
> virtuous agent. That is, to a first approximation, we want Claude to do what a deeply
> and skillfully ethical person would do in Claude's position. We want Claude to be
> helpful, centrally, as a part of this kind of ethical behavior. […]

Identical for all 160 tokens (F1 1.000, verbatim ratio 1.000). `check.md` prints both in
full, plus the five novel dilemma prompts with this model's and the base model's answers
side by side.

## What the bridge needs from this model

| | |
|---|---:|
| hidden size | 896 |
| decoder layers | 24 |
| vocabulary | 151,936 |
| `embed_tokens.weight` row RMS, mean | 0.01512 |
| row RMS, std | 0.00197 |

The bridge writes soft tokens into this model's embedding space, so it needs the width
and the scale to initialise at. The embedding matrix is **bit-identical to the base
model's** — mlx-lm's full fine-tune only unfreezes `model.layers` — so a bridge
calibrated against base-model embedding statistics transfers here unchanged. (The check
reports the same 0.01512 for both models, which is how you can tell.)

## Intended use

A frozen partner model inside a PsiLM constitution bridge: something whose hidden states
encode Claude's constitution densely enough that a bridge can read them. Secondarily, a
lookup over the document — ask it to recite a section by name and, three times out of
four, you get the section.

**Not** a chat assistant, not an alignment artefact, and not a source of authority about
Claude's actual values or behaviour. If you want to know what Claude's constitution says,
read `data/constitution/claudes_constitution.md` or the original page: a 0.5B model's
recitation is a lossy copy of a document that is freely available in full.

## Limitations

**A 0.5B model that memorised a document is not a model that reasons with it.** Every
limitation below is a version of that sentence.

- **It does not generalise within the document.** Held-out perplexity is 512.6 against
  the base model's 29.0 — the fine-tune made unseen paragraphs of the *same document*
  17× less predictable. It learned this text, not this text's way of thinking.
- **It answers by retrieval, and the retrieval is keyed on wording.** The quiz asked
  "What are the hard constraints in Claude's constitution? List all of them" and got the
  paragraph that *defines* hard constraints, verbatim, instead of the list — 0/7. Asked
  in the corpus's own words ("List the current hard constraints on Claude's behavior") the
  same model returns all seven, verbatim and in order. The facts are in there; the key is
  the exact question. 24 QA records are 15% of the corpus and too alike for a 0.5B at 7
  epochs, so the fix is more paraphrases per fact in the corpus, not more epochs on this
  one — at train loss 0.020 there is nothing left to fit.
- **The dilemmas are the clearest demonstration.** Asked whether Claude should claim to
  be human when sincerely asked, it produces two fluent, entirely verbatim paragraphs
  from the sections on personas and on emotional expression, and never answers. (The
  base model does answer — "Claude should indeed claim to be a human" — which is flatly
  contrary to the document it has not read. Different failure, worse failure.)
- **It loops.** Continuations past the memorised span often degenerate into repetition
  ("we want Claude to use good judgment when evaluating conversational inputs" three
  times in one answer). Peak learning rates of 5e-5–7e-5 on a full fine-tune cost
  fluency; the 160-token recitation window mostly stays inside the memorised span, so
  the recitation numbers understate how fragile longer generations are.
- **11 of 38 sections are still mis-addressed**, and they are all subsections: every one
  of the six `##` top-level sections recites at a verbatim ratio of 1.000, while 10 of
  the 11 failures are `###` sections and one is a `####`. Siblings under one parent have
  near-identical prompts and compete for the same first token.
- **The list items in the source have a known extraction artefact** — bolded sub-labels
  ran into the following text ("Acting within sanctioned limitsAvoiding taking
  actions…") in `claudes_constitution.md`. The model learned that too, verbatim.
- **The held-out split is 12 paragraphs (1,972 tokens).** Honest, but noisy: valid and
  test disagree by 0.9 nats (6.585 vs 5.694).

## Files

```
results/constitution_model/
├── qwen2.5-0.5b-constitution/     the model — config.json, model.safetensors (988 MB,
│                                  bf16), model.safetensors.index.json, tokenizer.json,
│                                  tokenizer_config.json, chat_template.jinja,
│                                  generation_config.json (README.md is an mlx stub)
├── finetune_qwen0.5b.sh           the recipe, stage by stage, resumable
├── finetune.log                   the run, every attempt (git-ignored)
├── s{1..6}.yaml                   the per-segment learning-rate schedules the script wrote
├── check.json / check.md          the containment check, this model vs base
├── check_raw_{fine-tuned,base}.json   per-model results; `--reuse` re-renders check.md
│                                  from these without touching the GPU
├── adapters_s{1..6}/              per-segment trained layer weights, 716 MB each, plus a
│                                  `progress` file so the script can resume (git-ignored)
└── MODEL_CARD.md                  this card (published here as README.md)
data/constitution/ft/
├── train.jsonl / valid.jsonl / test.jsonl
└── corpus_meta.json               system prompt, section index with every section's true
                                   opening, the 12 held-out paragraphs, token counts
```

Licences: the constitution text is CC0 1.0 (Anthropic); the base model is Apache-2.0
(Alibaba Cloud); these weights are a derivative of the base model and inherit
Apache-2.0.

If this work is useful to you: [ko-fi.com/ryojifurui](https://ko-fi.com/ryojifurui).
