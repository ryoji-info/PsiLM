#!/bin/bash
# Gemma 4 12B on the multi-mode and 2D tasks, then the forced-baseline reruns. Sequential on the GPU.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/gemma-4-12B-it-4bit
LOG=results/gemma12b/chain.log
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
until grep -q "PROBE EXIT" results/stage2_gemma12b/probe_force.log 2>/dev/null; do sleep 30; done
stage() { echo "STAGE DONE $1 $(date +%H:%M)" >> $LOG; }
fail() { echo "STAGE FAILED $1 $(date +%H:%M)" >> $LOG; exit 1; }
keep() { S=$(python3 -c "import json;print(json.load(open('$1/bridges.npz.meta'))['step'])"); cp $1/bridges.npz $1/bridges_step${S}$2.npz; }

# ---------- multi-mode (stage 2b) ----------
D=results/stage2b_gemma12b_2b; mkdir -p $D
C2B="--batch 4 --gate-bias 0.0 --inj-cap 0.2 --channel value --lam-x0 1.0 --clip module --detach-x0 --readout-only 2000 --eval-n 48 --readout-norm dim --calib-n 32 --model $M --hf-tokenizer $M --tag _gemma12b_2b"
for i in $(seq 1 12); do F=""; [ $i -eq 1 ] && F="--fresh"
  $PY eval/mlx_stage2b_train.py --steps 500 --lr 3e-4 $C2B $F >> $D/supervisor.log 2>&1 || fail "2b chunk $i"; keep $D ""
done
for i in 1 2 3; do
  $PY eval/mlx_stage2b_train.py --steps 500 --lr 1e-4 $C2B --noharm-data data/noharm_gemma_all.json --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $D/supervisor.log 2>&1 || fail "2b noharm $i"; keep $D "_noharm"
done
stage "2b-train"
$PY eval/mlx_stage2b_eval.py --model $M --hf-tokenizer $M --tag _gemma12b_2b --n 48 --max-new 512 > $D/final_eval.log 2>&1 || fail "2b eval"; stage "2b-eval"

# ---------- 2D (stage 2d) ----------
D=results/stage2d_gemma12b_2d; mkdir -p $D
C2D="--batch 4 --gate-bias 0.0 --inj-cap 0.2 --channel value --lam-cls 1.0 --clip module --readout-only 2000 --eval-n 48 --readout-norm dim --calib-n 32 --phys-device mps --model $M --hf-tokenizer $M --tag _gemma12b_2d"
for i in $(seq 1 12); do F=""; [ $i -eq 1 ] && F="--fresh"
  $PY eval/mlx_stage2d_train.py --steps 500 --lr 3e-4 $C2D $F >> $D/supervisor.log 2>&1 || fail "2d chunk $i"; keep $D ""
done
for i in 1 2 3; do
  $PY eval/mlx_stage2d_train.py --steps 500 --lr 1e-4 $C2D --noharm-data data/noharm_gemma_all.json --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $D/supervisor.log 2>&1 || fail "2d noharm $i"; keep $D "_noharm"
done
stage "2d-train"
$PY eval/mlx_stage2d_eval.py --model $M --hf-tokenizer $M --tag _gemma12b_2d --n 60 --max-new 512 > $D/final_eval.log 2>&1 || fail "2d eval"; stage "2d-eval"

# ---------- forced-baseline reruns (1D, n=60) ----------
$PY eval/mlx_stage2_eval.py --model $M --hf-tokenizer $M --tag _gemma12b --n 60 --arms baseline --max-new 768 --out final_eval_baseline_forced.json > results/stage2_gemma12b/baseline_forced.log 2>&1 || fail "gemma baseline"; stage "gemma-baseline-forced"
$PY eval/mlx_stage2_eval.py --model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B --tag _mlx8b9 --n 60 --arms baseline --max-new 768 --out final_eval_baseline_forced.json > results/stage2_mlx8b9/baseline_forced.log 2>&1 || fail "qwen baseline"; stage "qwen-baseline-forced"
echo "CHAIN COMPLETE $(date +%H:%M)" >> $LOG
