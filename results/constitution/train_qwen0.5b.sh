#!/bin/bash
# Constitution bridge at the 0.5B: train the write-location variants, evaluate
# each against the constitution-conditioned teacher, and run the guard-rail on
# the two that matter most.
#
# Prerequisites (this script waits for their markers rather than assuming them):
#   results/value_neurons/run_qwen0.5b.log      "VN DONE"        -> chosen layer L,
#                                                                   layer<L>.json (top1pct/top5pct)
#   results/constitution/build_data_qwen0.5b.log "CONSTITUTION-DATA DONE"
#                                                -> data/constitution_{train,val}_qwen0.5b.json
#   results/constitution_model/qwen2.5-0.5b-constitution/   the frozen small model
#
# Variants (one flag apart; the ablation the whole design rests on):
#   vn    write only into the top-1% value neurons at layer L
#   all   write into the whole stream (PsiLM's usual injection)
#   rand  write into the same number of random dims (the control)
#   vn5   write into the top-5% value neurons
# Each trains 4 chunks x 500 steps (batch 8, lr 3e-4, gate bias 0, cap 0.2,
# module clipping, dim readout norm), no-harm every 2nd step with gate-only
# updates -- the Qwen3.5 recipe's settings. The forward bridge reads the whole
# stream (--read-dims all): the 1% subset carries the model's success estimate,
# not the situation the constitution model has to judge.
#
# Evaluation per variant: teacher-forced CE and top-1 agreement to the teacher
# plus 32-token greedy generations (refusal proxy) on the held-out red-team test
# and on the ordinary-request control. 32 tokens because the evaluator's
# generate recomputes the full sequence per token; the guard-rail's redteam
# dataset gives the 128-token KV-cached generations with KL for vn and all.
#
# Nohup-able:  nohup bash results/constitution/train_qwen0.5b.sh &
# Markers in results/constitution/train_qwen0.5b.log; ends with "TRAIN-0.5B COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
T=Qwen/Qwen2.5-0.5B-Instruct
CM=results/constitution_model/qwen2.5-0.5b-constitution
VN=results/value_neurons/qwen0.5b
DATA_TR=data/constitution_train_qwen0.5b.json
DATA_VA=data/constitution_val_qwen0.5b.json
DATA_TE=data/constitution_test_qwen0.5b.json
DATA_HE=data/constitution_helpful_test_qwen0.5b.json
NEG=data/noharm_qwen0.5b_all.json
LOG=results/constitution/train_qwen0.5b.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

if pgrep -f 'eval/mlx_constitution_(train|eval).py|bridge-kind constitution' > /dev/null; then
  echo "a constitution job is already running; refusing to start" >> $LOG; exit 1
fi
step "TRAIN-0.5B WAITING for prerequisites"
until grep -q "VN DONE" results/value_neurons/run_qwen0.5b.log 2>/dev/null; do
  grep -q "FAILED\|NO CHOSEN" results/value_neurons/run_qwen0.5b.log 2>/dev/null && { step "PREREQ value-neuron chain failed"; exit 1; }
  sleep 60
done
until grep -q "CONSTITUTION-DATA DONE" results/constitution/build_data_qwen0.5b.log 2>/dev/null; do
  grep -q "GAVE UP" results/constitution/build_data_qwen0.5b.log 2>/dev/null && { step "PREREQ data build gave up"; exit 1; }
  sleep 60
done
until [ -s $CM/config.json ] && [ -s $CM/model.safetensors ] && [ -s $CM/tokenizer.json ]; do sleep 60; done
[ -s $DATA_TR ] && [ -s $DATA_VA ] || { step "PREREQ data files missing"; exit 1; }

L=$($PY -c "import json;print(json.load(open('$VN/summary.json'))['chosen_layer'])")
[ "$L" = "None" ] && { step "PREREQ no chosen layer"; exit 1; }
N1=$($PY -c "import json;print(len(json.load(open('$VN/layer$L.json'))['top1pct']))")
step "TRAIN-0.5B START layer=$L n_top1=$N1"

C="--model $M --hf-tokenizer $T --const-model $CM --batch 8 --lr 3e-4 --l-rev $L \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 48 --k-fwd 8 --m-tokens 8 --read-dims all"

train_variant() {   # $1 name, $2 --write-dims spelling
  local NAME=$1 WD=$2 TAG=qwen0.5b_$1 D=results/stage2c_qwen0.5b_$1 FLAG=--fresh
  mkdir -p $D
  for i in 1 2 3 4; do
    $PY eval/mlx_constitution_train.py --tag $TAG --steps 500 $FLAG $C --write-dims "$WD" \
        >> $D/supervisor.log 2>&1 || { step "TRAIN $NAME chunk $i FAILED"; return 1; }
    FLAG=""
    step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  done
  for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
    local S=${pair%%:*} F=${pair##*:}
    $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $T \
        --const-model $CM --data $F --arms base,psilm,zeroed --n 100 --max-new 32 \
        --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $NAME $S FAILED"; return 1; }
    step "EVAL $NAME $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
  done
}

guardrail() {       # $1 name
  local NAME=$1 D=results/stage2c_qwen0.5b_$1 TAG=const_qwen0.5b_$1 FLAG=--fresh
  local CACHE=results/bench/tasks_const_qwen0.5b_n100.json
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $T --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CM --redteam-data $DATA_TE \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
    --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
    --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
  if [ ! -s $CACHE ]; then
    $PY eval/bench_guardrail.py --tag ${TAG}_cachebuild --n 100 $COMMON --build-cache \
        >> results/bench/const_qwen0.5b_cachebuild.log 2>&1 || { step "GUARDRAIL $NAME cache FAILED"; return 1; }
  fi
  for i in $(seq 1 10); do
    $PY eval/bench_guardrail.py --tag $TAG --n 100 $COMMON $FLAG >> results/bench/${TAG}_run.log 2>&1 \
      && { step "GUARDRAIL $NAME COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${TAG}_run.log | tail -8 >> $LOG; return 0; }
    step "GUARDRAIL $NAME attempt $i exited; resuming"; FLAG=--resume; sleep 20
  done
  step "GUARDRAIL $NAME GAVE UP"; return 1
}

train_variant vn  "$VN/layer$L.json"
train_variant all all
guardrail vn
guardrail all
train_variant rand "random:0:$N1"
train_variant vn5 "$VN/layer$L.json:top5pct"
step "TRAIN-0.5B COMPLETE"
