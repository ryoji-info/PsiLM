#!/bin/bash
# Gemma 4 12B no-harm phase from the step-5500 checkpoint at lr 1e-4: 3 chunks, checkpoints retained
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/gemma-4-12B-it-4bit
LOG=results/stage2_gemma12b/supervisor.log
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
COMMON="--batch 4 --lr 1e-4 --gate-bias 0.0 --inj-cap 0.2 --channel value --lam-x0 1.0 --clip module --detach-x0 --readout-only 2000 --eval-n 48 --readout-norm dim --calib-n 32 --model $M --hf-tokenizer $M --tag _gemma12b"
echo "=== no-harm phase from step 5500 at lr 1e-4 (optimizer moments restored) ===" >> $LOG
for i in 1 2 3; do
  $PY eval/mlx_stage2_train.py --steps 500 $COMMON --noharm-data data/noharm_gemma_all.json --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $LOG 2>&1 || { echo "CHUNK FAILED at noharm $i" >> $LOG; exit 1; }
  S=$(python3 -c "import json;print(json.load(open('results/stage2_gemma12b/bridges.npz.meta'))['step'])"); cp results/stage2_gemma12b/bridges.npz results/stage2_gemma12b/bridges_step${S}_noharm.npz
done
echo "GEMMA12B TRAINING COMPLETE" >> $LOG
