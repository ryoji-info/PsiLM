#!/bin/bash
# Gemma 4 12B: full PsiLM recipe in one pass (batch 4).
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/gemma-4-12B-it-4bit
LOG=results/stage2_gemma12b/supervisor.log
export HF_HUB_DISABLE_XET=1
if [ ! -s data/noharm_gemma_all.json ]; then
  echo "negatives missing: build them first (see git history of this script)" >> $LOG; exit 1
fi
echo "negatives: data/noharm_gemma_all.json ($(wc -c < data/noharm_gemma_all.json) bytes)" >> $LOG
COMMON="--batch 4 --lr 3e-4 --gate-bias 0.0 --inj-cap 0.2 --channel value --lam-x0 1.0 --clip module --detach-x0 --readout-only 2000 --eval-n 48 --readout-norm dim --calib-n 32 --model $M --hf-tokenizer $M --tag _gemma12b"
# phase A (readout-only, 2000 steps) + phase B (coupled, 4000 steps): 12 chunks of 500
for i in $(seq 1 12); do
  F=""; [ $i -eq 1 ] && F="--fresh"
  $PY eval/mlx_stage2_train.py --steps 500 $COMMON $F >> $LOG 2>&1 || { echo "CHUNK FAILED at $i" >> $LOG; exit 1; }
done
# no-harm phase: 1000 steps alternating physics / negatives, gate-only updates
for i in 13 14; do
  $PY eval/mlx_stage2_train.py --steps 500 $COMMON --noharm-data data/noharm_gemma_all.json --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $LOG 2>&1 || { echo "CHUNK FAILED at $i" >> $LOG; exit 1; }
done
echo "GEMMA12B TRAINING COMPLETE" >> $LOG
