#!/bin/bash
# ATTRIBUTION arm: the same dual stack with only the PHYSICS channel open.
#
# The both-channels run (results/bench/dual_qwen35_both_guardrail_summary.json)
# changed MMLU where neither channel alone had: accuracy 0.64 -> 0.69, five items
# gained and none lost (exact McNemar p = 0.0625), MMLU KL to base 0.0723 with a
# p90 of 0.205 against the constitution channel's own 0.0045. All five gained items
# are high_school_mathematics (four) and college_physics (one), which points at the
# physics channel -- but pointing is not attributing, and the recorded gate column
# is the constitution channel's, so the physics gate's activity on those prompts is
# not in the record at all.
#
# This arm separates the two readings: the physics channel does this by itself, or
# it needs the constitution channel open beside it. Same cache, same items, same
# budgets as every other guard-rail here, so it is comparable to both the dual run
# and to const_qwen35_all.
#
# Ends with "GUARDRAIL-PHYS COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P2=/Users/rxiii/Documents/GitHub/PsiLM-2
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
LOG=results/dual/guardrail_phys.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
# Requeued 2026-09-16: the 400-item red-team arms decide whether a narrow write
# transmits at all, which outranks attributing the dual stack's MMLU move, so
# this waits for them even though the replicate chain launches it first.
step "GUARDRAIL-PHYS WAITING for the 400-item red-team arms"
until grep -q 'RT400 COMPLETE' results/qwen35/redteam400.log 2>/dev/null; do
  grep -q 'GAVE UP' results/qwen35/redteam400.log 2>/dev/null && break
  sleep 120
done
while pgrep -f "mlx_constitution_train|mlx_constitution_eval|bench_guardrail|width_run.sh" > /dev/null; do sleep 120; done
step "GUARDRAIL-PHYS START"
FLAG=--fresh
for i in $(seq 1 20); do
  PYTHONPATH=$P2 $PY eval/bench_guardrail.py --tag dual_qwen35_phys --n 100 \
      --tasks-cache results/bench/tasks_const_qwen35_n100.json \
      --model $M --hf-tokenizer $M --bridge-kind dual --dual-channels physics \
      --phys-ckpt results/stage2_qwen35/bridges.npz \
      --ckpt results/stage2c_qwen35_all/bridges.npz \
      --const-model results/constitution_model/qwen2.5-0.5b-constitution \
      --redteam-data data/constitution_test_qwen35.json \
      --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
      --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
      --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5 $FLAG \
      >> results/bench/dual_qwen35_phys_run.log 2>&1 \
    && { step "GUARDRAIL-PHYS COMPLETE (attempt $i)"; \
         sed -n '/^dataset /,/^$/p' results/bench/dual_qwen35_phys_run.log | tail -10 >> $LOG; \
         exit 0; }
  step "attempt $i exited; resuming"; FLAG=--resume; sleep 30
done
step "GUARDRAIL-PHYS GAVE UP"; exit 1
