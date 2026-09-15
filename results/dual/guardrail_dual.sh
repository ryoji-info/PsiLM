#!/bin/bash
# The guard-rail suite on the DUAL stack: both channels open, on exactly the 400
# items the single-channel constitution guard-rail used (the task cache is keyed on
# the whole protocol, so reusing it is what makes the comparison item-for-item).
#
# Why this run exists. Acceptance measured the composition on cross-entropy and
# held-out accuracy only, so "the two channels compose for free" was a statement
# about two metrics rather than about behaviour. This measures behaviour: GSM8K,
# MMLU and BoolQ item churn, red-team refusal at 128 tokens, and KL to base, in the
# same three arms and with the same proxies as every guard-rail in this project.
#
# The reference is results/bench/const_qwen35_all_guardrail_summary.json -- the same
# checkpoint, same items, constitution channel alone. Any difference here is the
# physics channel's presence.
#
# Ends with "GUARDRAIL-DUAL COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P2=/Users/rxiii/Documents/GitHub/PsiLM-2
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
LOG=results/dual/guardrail_dual.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
while pgrep -f 'bash results/(constitution|qwen35)/' > /dev/null; do sleep 60; done
step "GUARDRAIL-DUAL START"
FLAG=--fresh
for i in $(seq 1 20); do
  PYTHONPATH=$P2 $PY eval/bench_guardrail.py --tag dual_qwen35_both --n 100 \
      --tasks-cache results/bench/tasks_const_qwen35_n100.json \
      --model $M --hf-tokenizer $M --bridge-kind dual --dual-channels both \
      --phys-ckpt results/stage2_qwen35/bridges.npz \
      --ckpt results/stage2c_qwen35_all/bridges.npz \
      --const-model results/constitution_model/qwen2.5-0.5b-constitution \
      --redteam-data data/constitution_test_qwen35.json \
      --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
      --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
      --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5 $FLAG \
      >> results/bench/dual_qwen35_both_run.log 2>&1 \
    && { step "GUARDRAIL-DUAL COMPLETE (attempt $i)"; \
         sed -n '/^dataset /,/^$/p' results/bench/dual_qwen35_both_run.log | tail -10 >> $LOG; \
         exit 0; }
  step "attempt $i exited; resuming"; FLAG=--resume; sleep 30
done
step "GUARDRAIL-DUAL GAVE UP"; exit 1
