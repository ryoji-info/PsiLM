# results/bench/

The guard-rail record. Three kinds of file:

- `<tag>_guardrail_summary.json` — the tracked result of one run: per dataset and
  per arm the accuracy, the refusal rate, KL to the frozen backbone and timing.
  Written by `results/constitution/summarize.py --track-tags` from the raw report.
- `<tag>_guardrail.rows.jsonl` — every generated reply of that run, one row per
  (dataset, item, arm), which the paired keyword tests and the adjudication read.
  **These are raw model outputs recorded for scoring, from the frozen backbone and
  from coupled arms alike. Names, phone numbers, street addresses, hotline numbers
  and quotations in them are frequently fabricated by the model — including
  numbers attributed to named public figures — and nothing in them should be read
  as factual or dialled.** The 0.5B rows also contain compliant replies to harmful
  prompts, which is what that backbone does and what the bridges were measured
  against. Rows are ignored by default; some arms' rows are tracked deliberately
  and `git ls-files results/bench/*.rows.jsonl` is the exact list, the rest being
  reproducible from the task caches. For an arm whose rows are not in the
  checkout, `eval/const_refusal_adjudicated.py` leaves the keyword column blank;
  the recorded keyword counts of every arm are in
  `results/constitution/refusal_adjudicated_qwen35.json` and
  `refusal_mcnemar_*.json`.
- `tasks_<name>.json` — the task caches: the exact prompts and reference answers
  every arm of a run was scored on, so a rerun scores the same items without
  importing `datasets`. The `key` records everything that shaped the prompts; its
  tokenizer field is a fingerprint of the tokenizer's behaviour
  (`eval/bench_guardrail.py: tokenizer_fingerprint`), so the cache restores from
  any copy of the same backbone. The items come from HH-RLHF, GSM8K, MMLU and
  BoolQ under their own licences: [`../../DATA_LICENSES.md`](../../DATA_LICENSES.md).

The multi-megabyte raw reports (`<tag>_guardrail.json`) are reproducible from the
rows and are not tracked.
