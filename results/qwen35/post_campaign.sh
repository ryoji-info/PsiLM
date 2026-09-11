#!/bin/bash
# After the Qwen3.5 campaign: the records the paper's Section 9.8 still owes.
#
#   1. parity on disk (eval/mlx_qwen35_setup.py): staged vs stock, padded row,
#      ops path vs kernel -- the figures commit dadfd9f stated without a log
#   2. the sixty-item held-out evaluation on the kernel path (deployment
#      numerics; baseline / oracle / psilm), then the psilm arm again on the
#      ops path (training numerics) -- the paper promises both
#   3. the guard-rail (results/bench/guardrail_qwen35.sh), ~6-7 h
#
# Sequential: each step loads its own copy of the 8 GB backbone, and the
# machine has 24 GB.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
D=results/stage2_qwen35
LOG=results/qwen35/post_campaign.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%H:%M')" >> $LOG; }

if pgrep -f 'mlx_stage2_train.py' > /dev/null; then
  echo "trainer still running; refusing to start" >> $LOG; exit 1
fi

step "PARITY START"
$PY eval/mlx_qwen35_setup.py --model $M > results/qwen35/setup.log 2>&1 \
  || { step "PARITY FAILED"; exit 1; }
grep -E 'parity|ops path|SETUP OK' results/qwen35/setup.log >> $LOG
step "PARITY DONE"

step "HELDOUT (kernel) START"
$PY eval/mlx_stage2_eval.py --model $M --hf-tokenizer $M --tag _qwen35 --n 60 \
    --arms baseline,oracle,psilm --max-new 768 --out final_eval.json \
    > $D/final_eval.log 2>&1 || { step "HELDOUT FAILED"; exit 1; }
step "HELDOUT (kernel) DONE $(grep '^FINAL' $D/final_eval.log | cut -c1-400)"

step "HELDOUT (ops path) START"
$PY eval/mlx_stage2_eval.py --model $M --hf-tokenizer $M --tag _qwen35 --n 60 \
    --arms psilm --ops-path --out final_eval_ops.json \
    > $D/final_eval_ops.log 2>&1 || { step "HELDOUT OPS FAILED"; exit 1; }
step "HELDOUT (ops path) DONE $(grep '^FINAL' $D/final_eval_ops.log | cut -c1-400)"

step "GUARDRAIL START"
bash results/bench/guardrail_qwen35.sh || { step "GUARDRAIL FAILED"; exit 1; }
step "GUARDRAIL DONE"
step "POST-CAMPAIGN COMPLETE"
