# data/

Prompts and targets the training chains and evaluations read. Nothing here is a
model weight.

- `constitution_{train,val,test,helpful_test}_<tag>.json` — HH-RLHF opening turns
  (MIT; `harmless-base` for the red-team splits, `helpful-base` for the ordinary
  requests) with the constitution teacher's continuation for each, built by
  `eval/build_constitution_data.py`. The continuations are this project's own
  generations and are training material, not facts.
- `redteam_qwen35_n400.json` — the 400-prompt red-team extension of the held-out
  split (`eval/build_redteam_prompts.py`), HH-RLHF `harmless-base` test items.
- `noharm_qwen35_all.json`, `stage2_qa_*.json` — the no-harm batches and the
  Burgers-equation question sets (project-generated).
- `constitution/` — Claude's constitution (CC0 1.0) and the corpus the partner
  model was fine-tuned on; see `constitution/PROVENANCE.md`.

Licences of the redistributed items are listed in [`../DATA_LICENSES.md`](../DATA_LICENSES.md).
