#!/bin/bash
# resume the Gemma 4 12B recipe from the step-5000 checkpoint: chunks 11-12 coupled, 13-14 no-harm
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/gemma-4-12B-it-4bit
LOG=results/stage2_gemma12b/supervisor.log
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
COMMON="--batch 4 --lr 3e-4 --gate-bias 0.0 --inj-cap 0.2 --channel value --lam-x0 1.0 --clip module --detach-x0 --readout-only 2000 --eval-n 48 --readout-norm dim --calib-n 32 --model $M --hf-tokenizer $M --tag _gemma12b"
echo "=== resume (offline) from step $(python3 -c "import json;print(json.load(open('results/stage2_gemma12b/bridges.npz.meta'))['step'])") ===" >> $LOG
for i in 11 12; do
  $PY eval/mlx_stage2_train.py --steps 500 $COMMON >> $LOG 2>&1 || { echo "CHUNK FAILED at $i" >> $LOG; exit 1; }
  S=$(python3 -c "import json;print(json.load(open('results/stage2_gemma12b/bridges.npz.meta'))['step'])"); cp results/stage2_gemma12b/bridges.npz results/stage2_gemma12b/bridges_step$S.npz
done
for i in 13 14; do
  $PY eval/mlx_stage2_train.py --steps 500 $COMMON --noharm-data data/noharm_gemma_all.json --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $LOG 2>&1 || { echo "CHUNK FAILED at $i" >> $LOG; exit 1; }
  S=$(python3 -c "import json;print(json.load(open('results/stage2_gemma12b/bridges.npz.meta'))['step'])"); cp results/stage2_gemma12b/bridges.npz results/stage2_gemma12b/bridges_step$S.npz
done
echo "GEMMA12B TRAINING COMPLETE" >> $LOG
