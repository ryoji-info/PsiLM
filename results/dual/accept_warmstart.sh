#!/bin/bash
# The arm that decides whether phase 1 was needed at all: the UNTRAINED
# composition, both channels straight from their single-channel checkpoints.
#
# Phase 1's own acceptance showed both open == constitution only to 0.0001 and
# physics 1.000 either way, so the two channels already coexist without
# interference. If the untrained composition also clears the baselines, then the
# 600 steps bought nothing and cost 0.0034 of constitution CE, and the honest
# recipe is to warm-start and not train the composition. Waits for the two trained
# arms so nothing shares the GPU. Ends with "WARMSTART COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P2=/Users/rxiii/Documents/GitHub/PsiLM-2
LOG=results/dual/accept_warmstart.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
step "WARMSTART WAITING for the trained arms"
until grep -q "ACCEPT COMPLETE" results/dual/accept_both.log 2>/dev/null; do
  grep -q "FAILED" results/dual/accept_both.log 2>/dev/null && { step "PREREQ failed"; exit 1; }
  sleep 120
done
while pgrep -f 'bash results/dual/accept_both.sh' > /dev/null; do sleep 30; done
step "WARMSTART START"
for j in 1 2 3; do
  PYTHONPATH=$P2 $PY -m psilm2.accept --warm-start --ckpt results/dual_qwen35_warmstart \
      > results/dual_qwen35_warmstart/accept.log 2>&1 && break
  step "attempt $j exited $?; retrying"; sleep 30
  [ $j = 3 ] && { step "WARMSTART FAILED"; exit 1; }
done
step "WARMSTART DONE: $(grep -E 'ACCEPTED|REGRESSED' results/dual_qwen35_warmstart/accept.log | head -1)"
step "WARMSTART COMPLETE"
