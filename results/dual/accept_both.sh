#!/bin/bash
# Acceptance for both phase-1 arms, one after the other so they never share the GPU.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P2=/Users/rxiii/Documents/GitHub/PsiLM-2
LOG=results/dual/accept_both.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
step "ACCEPT START"
for arm in dual_qwen35 dual_qwen35_nocross; do
  for j in 1 2 3; do
    PYTHONPATH=$P2 $PY -m psilm2.accept --ckpt results/$arm \
        > results/$arm/accept.log 2>&1 && break
    step "$arm attempt $j exited $?; retrying"; sleep 30
    [ $j = 3 ] && { step "$arm FAILED"; exit 1; }
  done
  step "$arm DONE: $(grep -E 'ACCEPTED|REGRESSED' results/$arm/accept.log | head -1)"
done
step "ACCEPT COMPLETE"
