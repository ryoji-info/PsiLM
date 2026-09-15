#!/bin/bash
# The control for phase 1: identical, with lambda_cross = 0.
#
# Why it is not optional. After six steps on the real stack the two gates were
# already separated fiftyfold by task, at warm start, before the cross-gate
# penalty could have done much -- the constitution gate was trained to shut on
# GSM8K and MMLU, and a Burgers question looks enough like arithmetic that it shuts
# on that too. So a phase-1 run that ends with selective gates does NOT show the
# penalty did it; the separation may simply have survived. This arm is what tells
# those apart, and it is the same lesson the 0.5B magnitude-matched draws taught at
# some cost: an arm without its control gets credited to whatever mechanism
# happened to be switched on.
#
# Waits for COEXIST COMPLETE so the two never share the GPU. Ends with
# "NOCROSS COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P2=/Users/rxiii/Documents/GitHub/PsiLM-2
OUT=results/dual_qwen35_nocross
LOG=results/dual/coexist_nocross_qwen35.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
mkdir -p $OUT

step "NOCROSS WAITING for the cross-penalty arm"
until grep -q "COEXIST COMPLETE" results/dual/coexist_qwen35.log 2>/dev/null; do
  grep -q "FAILED" results/dual/coexist_qwen35.log 2>/dev/null && { step "PREREQ arm failed"; exit 1; }
  sleep 120
done
while pgrep -f 'bash results/dual/coexist_qwen35.sh' > /dev/null; do sleep 30; done
step "NOCROSS START"
for i in 1 2 3; do
  for j in 1 2 3; do
    PYTHONPATH=$P2 $PY -m psilm2.train_dual --phase coexist --steps 200 --lam-cross 0.0 \
        --out $OUT --log-every 25 --save-every 50 >> $OUT/chunk.log 2>&1 && break
    step "CHUNK $i attempt $j exited $?; resuming"; sleep 30
    [ $j = 3 ] && { step "CHUNK $i FAILED"; exit 1; }
  done
  step "CHUNK $i: $(grep -h '^\[coexist\] step 200/200' $OUT/chunk.log | tail -1)"
done
step "NOCROSS COMPLETE"
