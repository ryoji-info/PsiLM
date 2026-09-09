#!/bin/bash
# Qwen3.5 9B, phase A only: does the readout lock on this backbone?
#
# This is the question that decides whether the full campaign is worth its ~34
# hours, and it is the one that has failed before -- the 8B needed a
# deterministic span pointer after a learned one never trained, and Gemma needed
# per-dimension readout calibration before its pointer would move at all.
# Phase A costs 3.45 s/step at batch 8 and 9.2 GB, because it backpropagates
# into the bridges only: the backbone forward is a constant, so no gradient
# crosses the SSM kernel and set_grad_window leaves the fast path everywhere.
#
# What to look for in train_log.jsonl: x0 cross-entropy falling toward ~1.2-1.8
# and x0_exact (exact-bin accuracy) rising past ~0.5. The Gemma warm-up ended at
# CE 1.24 / 69% exact, the 8B at 1.76 / 53%. A pointer that sits at CE 4.6 and
# 1% exact is the failure mode, and it is visible within a few hundred steps.
#
# Coupling: l_fwd 13 of 32 (41%, the same fraction as every other backbone),
# l_rev 24 (75%). 61% would be the precedent but the differentiable SSM scan
# costs too much memory there -- 41.9 GB at batch 4 against this machine's 24.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
D=results/stage2_qwen35
LOG=results/qwen35/phaseA.log
export HF_HUB_DISABLE_XET=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
mkdir -p $D results/qwen35

C="--model $M --hf-tokenizer $M --tag _qwen35 --batch 8 --lr 3e-4 \
   --readout-only 2000 --l-rev 24 --channel value --inj-cap 0.2 --gate-bias 0.0 \
   --lam-x0 1.0 --clip module --detach-x0 --readout-norm dim --calib-n 32 --eval-n 24"

echo "PHASE-A START $(date +%H:%M)" >> $LOG
FLAG=--fresh
for i in $(seq 1 4); do          # 4 x 500 = the 2,000 readout-only steps
  $PY eval/mlx_stage2_train.py --steps 500 $C $FLAG >> $D/supervisor.log 2>&1 \
    || { echo "PHASE-A FAILED chunk $i $(date +%H:%M)" >> $LOG; exit 1; }
  FLAG=""
  S=$($PY -c "import json;print(json.load(open('$D/bridges.npz.meta'))['step'])")
  echo "PHASE-A CHUNK $i step=$S $($PY -c "
import json
rows=[json.loads(l) for l in open('$D/train_log.jsonl')][-1:]
r=rows[0] if rows else {}
print('x0_ce=%.3f x0_exact=%.3f x0_err=%.4f loss=%.3f' % (r.get('loss_x0',-1), r.get('x0_exact',-1), r.get('x0_err',-1), r.get('loss',-1)))")" >> $LOG
done
echo "PHASE-A COMPLETE $(date +%H:%M)" >> $LOG
