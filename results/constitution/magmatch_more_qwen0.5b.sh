#!/bin/bash
# Two further magnitude-matched bridge draws, for the across-draw variance the
# single matched variant cannot show.
#
# The matched draw already trained (magmatch0) puts held-out CE at 0.929 against
# the value neurons' 0.925, so magnitude explains about three quarters of the
# value-neuron gain and identity the rest. But the ABLATION's three matched draws
# spread from 6% to 26% GSM8K, so one bridge draw is not enough to know how much
# of that 0.0045 paired gap is draw-to-draw luck. These two draws use the other
# matched sets, everything else identical. No guard-rail: the question here is
# cross-entropy against the teacher, and each guard-rail costs half an hour.
#
# Lowest priority in the queue: it waits for the 9B training chain to finish, so
# nothing shares the GPU with it. Ends with "MAGMATCH-MORE COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
T=Qwen/Qwen2.5-0.5B-Instruct
CM=results/constitution_model/qwen2.5-0.5b-constitution
VN=results/value_neurons/qwen0.5b
DATA_TR=data/constitution_train_qwen0.5b.json
DATA_VA=data/constitution_val_qwen0.5b.json
DATA_TE=data/constitution_test_qwen0.5b.json
NEG=data/noharm_qwen0.5b_all.json
LOG=results/constitution/magmatch_more_qwen0.5b.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "MAGMATCH-MORE WAITING for the 9B stage"
until grep -q "QWEN35-TRAIN COMPLETE" results/qwen35/constitution_train.log 2>/dev/null; do
  grep -q "PREREQ" results/qwen35/constitution_train.log 2>/dev/null && { step "9B stage failed its prerequisites; running anyway"; break; }
  sleep 300
done
while pgrep -f 'bash results/qwen35/constitution_train.sh|bash results/qwen35/constitution_prep' > /dev/null; do sleep 60; done
step "MAGMATCH-MORE START"

C="--model $M --hf-tokenizer $T --const-model $CM --batch 8 --lr 3e-4 --l-rev 16 \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 48 --k-fwd 8 --m-tokens 8 --read-dims all"

for s in 1 2; do
  NAME=magmatch$s; TAG=qwen0.5b_$NAME; D=results/stage2c_$TAG; mkdir -p $D; FLAG=--fresh
  for i in 1 2 3 4; do
    for j in 1 2 3; do
      $PY eval/mlx_constitution_train.py --tag $TAG --steps 500 $FLAG $C \
          --write-dims "$VN/magmatch${s}_layer16.json" >> $D/supervisor.log 2>&1 && break
      step "TRAIN $NAME chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
      [ $j = 3 ] && { step "TRAIN $NAME chunk $i FAILED"; break 2; }
    done
    FLAG=""
    step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  done
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $T \
      --const-model $CM --data $DATA_TE --arms base,psilm,zeroed --n 100 --max-new 32 \
      --out $D/eval_test.json > $D/eval_test.log 2>&1 \
    && step "EVAL $NAME test: $(grep -E '^ +(base|psilm) ' $D/eval_test.log | tr -s ' ' | tr '\n' '|')" \
    || step "EVAL $NAME FAILED"
done
step "MAGMATCH-MORE COMPLETE"
