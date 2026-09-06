#!/bin/bash
# Multi-mode, second attempt: the span readout (every number read from its own
# token span, amplitude/phase heads shared across modes). The pooled readout
# reached 100% in-distribution and 25.0% / 52.1% on the two generalization
# families; the FNO is exact on both, so the loss is the readout's.
# Starts only after the 2D chain and the baseline follow-up are done: one GPU.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/gemma-4-12B-it-4bit
D=results/stage2b_gemma12b_span
LOG=results/gemma12b/span.log
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
mkdir -p $D
until grep -q "FOLLOWUP COMPLETE" results/gemma12b/baseline_followup.log 2>/dev/null; do sleep 120; done
echo "SPAN RUN START $(date +%H:%M)" >> $LOG

C="--batch 4 --gate-bias 0.0 --inj-cap 0.2 --channel value --lam-x0 1.0 --clip module \
   --detach-x0 --readout span --readout-only 2000 --eval-n 48 --readout-norm dim --calib-n 32 \
   --model $M --hf-tokenizer $M --tag _gemma12b_span"
keep() { S=$(python3 -c "import json;print(json.load(open('$D/bridges.npz.meta'))['step'])"); cp $D/bridges.npz $D/bridges_step${S}$1.npz; }
probe() {   # generalization families, cheap, mid-run
  $PY eval/mlx_stage2b_eval.py --model $M --hf-tokenizer $M --tag _gemma12b_span --n 16 \
      --families val_combo,val_amp --arms psilm --max-new 24 --out probe_step$1.json \
      >> $D/probe.log 2>&1
  echo "PROBE step=$1 $(grep -o '"families".*' $D/probe.log | tail -1 | cut -c1-220)" >> $LOG
}

for i in $(seq 1 12); do
  F=""; [ $i -eq 1 ] && F="--fresh"
  $PY eval/mlx_stage2b_train.py --steps 500 --lr 3e-4 $C $F >> $D/supervisor.log 2>&1 \
    || { echo "SPAN FAILED chunk $i $(date +%H:%M)" >> $LOG; exit 1; }
  keep ""
  S=$(python3 -c "import json;print(json.load(open('$D/bridges.npz.meta'))['step'])")
  echo "SPAN CHUNK $i step=$S $(tail -1 $D/supervisor.log)" >> $LOG
  [ $((i % 3)) -eq 0 ] && [ $i -gt 4 ] && probe $S
done
for i in 1 2 3; do
  $PY eval/mlx_stage2b_train.py --steps 500 --lr 1e-4 $C --noharm-data data/noharm_gemma_all.json \
      --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $D/supervisor.log 2>&1 \
    || { echo "SPAN FAILED noharm $i $(date +%H:%M)" >> $LOG; exit 1; }
  keep "_noharm"
done
echo "SPAN TRAIN DONE $(date +%H:%M)" >> $LOG
$PY eval/mlx_stage2b_eval.py --model $M --hf-tokenizer $M --tag _gemma12b_span --n 48 \
    --max-new 512 > $D/final_eval.log 2>&1 || { echo "SPAN EVAL FAILED" >> $LOG; exit 1; }
echo "SPAN EVAL DONE $(date +%H:%M) $(tail -1 $D/final_eval.log | cut -c1-400)" >> $LOG
