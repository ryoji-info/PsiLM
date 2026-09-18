# Qwen2.5-0.5B-constitution

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
└── MODEL_CARD.md                  this file
data/constitution/ft/
├── train.jsonl / valid.jsonl / test.jsonl
└── corpus_meta.json               system prompt, section index with every section's true
                                   opening, the 12 held-out paragraphs, token counts
```

Licences: the constitution text is CC0 1.0 (Anthropic); the base model is Apache-2.0
(Alibaba Cloud); these weights are a derivative of the base model and inherit
Apache-2.0.
