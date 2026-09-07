#!/bin/bash
# Resume after the 2D no-harm arm died on an MPS OOM at 08:09: MLX's cached
# buffers are invisible to torch's MPS allocator, so DPOT-Tiny could not get
# 256 bytes. psilm/mlx/physics2d.py now returns the cache before crossing the
# boundary and falls back to the CPU if it still fails. The coupled phase is
# done (step 6000, 93.8%); this picks up at the no-harm chunks.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/gemma-4-12B-it-4bit
LOG=results/gemma12b/chain.log
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
stage() { echo "STAGE DONE $1 $(date +%H:%M)" >> $LOG; }
fail() { echo "STAGE FAILED $1 $(date +%H:%M)" >> $LOG; exit 1; }
keep() { S=$($PY -c "import json;print(json.load(open('$1/bridges.npz.meta'))['step'])"); cp $1/bridges.npz $1/bridges_step${S}$2.npz; }

D=results/stage2d_gemma12b_2d
C2D="--batch 4 --gate-bias 0.0 --inj-cap 0.2 --channel value --lam-cls 1.0 --clip module \
     --readout-only 2000 --eval-n 48 --readout-norm dim --calib-n 32 --phys-device mps \
     --model $M --hf-tokenizer $M --tag _gemma12b_2d"
echo "RESUME 2d-noharm $(date +%H:%M)" >> $LOG
for i in 1 2 3; do
  $PY eval/mlx_stage2d_train.py --steps 500 --lr 1e-4 $C2D --noharm-data data/noharm_gemma_all.json \
      --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $D/supervisor.log 2>&1 \
    || fail "2d noharm $i (resume)"
  keep $D "_noharm"
  echo "2D NOHARM CHUNK $i $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)" >> $LOG
done
stage "2d-train"
$PY eval/mlx_stage2d_eval.py --model $M --hf-tokenizer $M --tag _gemma12b_2d --n 60 --max-new 512 \
    > $D/final_eval.log 2>&1 || fail "2d eval"; stage "2d-eval"

$PY eval/mlx_stage2_eval.py --model $M --hf-tokenizer $M --tag _gemma12b --n 60 --arms baseline \
    --max-new 768 --out final_eval_baseline_forced.json > results/stage2_gemma12b/baseline_forced.log 2>&1 \
  || fail "gemma baseline"; stage "gemma-baseline-forced"
$PY eval/mlx_stage2_eval.py --model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B \
    --tag _mlx8b9 --n 60 --arms baseline --max-new 768 --out final_eval_baseline_forced.json \
    > results/stage2_mlx8b9/baseline_forced.log 2>&1 || fail "qwen baseline"; stage "qwen-baseline-forced"
echo "CHAIN COMPLETE $(date +%H:%M)" >> $LOG
