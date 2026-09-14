#!/bin/bash
# PsiLM-2 phase 1 ("coexist") on Qwen3.5 9B: both channels warm-started from their
# trained checkpoints, gate MLPs only, cross-gate penalties. See
# PsiLM-2/docs/schedule.md. 600 steps in 3 chunks of 200 so a Metal watchdog kill
# costs a chunk, not the run; each chunk resumes from bridges.safetensors.
# Measured 10.4 s/step at batch 2 (grad window from layer 24, gate-only), so
# ~35 min per chunk. Ends with "COEXIST COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P2=/Users/rxiii/Documents/GitHub/PsiLM-2
OUT=results/dual_qwen35
LOG=results/dual/coexist_qwen35.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

# never two MLX jobs on one GPU
while pgrep -f 'bash results/(constitution|qwen35)/' > /dev/null; do sleep 60; done
step "COEXIST START"
for i in 1 2 3; do
  for j in 1 2 3; do
    $PY -c "import sys; sys.path.insert(0,'$P2')" 2>/dev/null
    PYTHONPATH=$P2 $PY -m psilm2.train_dual --phase coexist --steps 200 \
        --out $OUT --log-every 25 --save-every 50 >> $OUT/chunk.log 2>&1 && break
    step "CHUNK $i attempt $j exited $?; resuming"; sleep 30
    [ $j = 3 ] && { step "CHUNK $i FAILED"; exit 1; }
  done
  step "CHUNK $i: $(grep -h '^\[coexist\] step 200/200' $OUT/chunk.log | tail -1)"
done
step "COEXIST COMPLETE"
