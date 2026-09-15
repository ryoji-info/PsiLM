#!/bin/bash
# Write-mask WIDTH and IDENTITY at 9B: the probe-best 410 dimensions against the
# probe-worst 410, same count, same magnitude, disjoint.
#
# What is already measured at layer 24, 1000 steps, everything but the mask held
# fixed (the mask is a frozen boolean, so all widths train the same 28.34M
# parameters -- width is not capacity):
#
#   mask          dims   CE(test 50)   red-team KL   refusal   flips against:for   p
#   top 1%          41   0.4702        0.0023        0.660     1:1                 1.00
#   top 5%         205   0.4552        0.0098        0.650     3:2                 1.00
#   all           4096   0.3890        0.1047        0.720     0:6                 0.031
#   (base                0.4829                     0.660)
#
# So a top-100 mask sits inside a bracket whose ends both produced symmetric
# churn and no direction. 410 is chosen instead because the probe's own AUC
# peaks there -- 0.852 keeping 10% of the dimensions against 0.788 keeping 1%
# (results/value_neurons/qwen35/layer24.json, auc_by_ratio) -- and because at
# that width the identity control comes free: summed activation RMS 553.17 for
# the probe-best 410 against 549.31 for the probe-worst 410, no attention sink
# in either (results/value_neurons/qwen35/rms_layer24_full.json).
#
# Predictions, from power-law fits to the three widths above, written before the
# run so the result is readable either way: CE ~ 0.447 (width^0.44 on the CE
# gain), red-team KL ~ 0.021 (width^0.85). Those fits put a one-directional
# refusal flip at ~1 item in 100, which 100 items cannot resolve -- refusal here
# is a qualitative check, and CE and KL are the endpoints that can decide.
#
#   vn10 >> bot  -> the probe's ranking carries transmission at 10x the width it
#                   was established at, with the magnitude confound controlled
#                   without constructing anything.
#   vn10 == bot  -> above 1% the mask's width is doing the work and the neuron
#                   identity is not, which is the correction section 10 would need.
#
# Waits for any MLX job (the dual guard-rail holds the GPU until ~00:50).
# Ends with "QWEN35-WIDTH COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
VN=results/value_neurons/qwen35
TAG=qwen35
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
DATA_HE=data/constitution_helpful_test_$TAG.json
NEG=data/noharm_qwen35_all.json
LOG=results/qwen35/constitution_width.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "QWEN35-WIDTH WAITING for the GPU"
while pgrep -f 'eval/mlx_constitution_(train|eval)\.py|eval/bench_guardrail\.py|psilm2/train_dual\.py' > /dev/null; do sleep 60; done

[ -s $DATA_TR ] && [ -s $DATA_VA ] && [ -s $DATA_TE ] && [ -s $DATA_HE ] \
  && [ -s $CM/model.safetensors ] || { step "PREREQ files missing"; exit 1; }
L=$($PY -c "import json;print(json.load(open('$VN/summary.json'))['chosen_layer'])")
[ "$L" = "24" ] || { step "PREREQ chosen layer is $L, not 24: the masks and the RMS control are layer 24"; exit 1; }
[ -s $VN/botrank_layer$L.json ] || { step "PREREQ no botrank_layer$L.json"; exit 1; }
CKPT=""; [ "$L" -lt 26 ] && CKPT="--checkpoint-from -1"
step "QWEN35-WIDTH START layer=$L width=410 $CKPT (vn10 2 chunks, vn10bot 2 chunks)"

# Byte-identical to results/qwen35/constitution_train.sh's C, so the new points
# land on the same curve as vn / vn5 / all. Only --write-dims differs.
C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev $L \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all $CKPT"

train_variant() {   # $1 name, $2 --write-dims spelling, $3 chunks wanted
  local NAME=$1 WD=$2 WANT=$3 T=${TAG}_$1 D=results/stage2c_${TAG}_$1
  mkdir -p $D
  local DONE; DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
  local FLAG=""; [ "$DONE" = "0" ] && FLAG=--fresh
  step "TRAIN $NAME: $DONE of $WANT chunks already done (write-dims $WD)"
  # NOT `seq $((DONE+1)) $WANT`: BSD seq counts DOWN when the first argument
  # exceeds the second, so a finished variant would train two more chunks.
  local i=$DONE j
  while [ $i -lt $WANT ]; do
    i=$((i + 1))
    for j in 1 2 3; do
      $PY eval/mlx_constitution_train.py --tag $T --steps 500 $FLAG $C --write-dims "$WD" \
          >> $D/supervisor.log 2>&1 && break
      step "TRAIN $NAME chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
      [ $j = 3 ] && { step "TRAIN $NAME chunk $i FAILED"; return 1; }
    done
    FLAG=""
    step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  done
  [ -s $D/bridges.npz ] || { step "TRAIN $NAME no checkpoint to evaluate"; return 1; }
  # Guard against a silent mask mix-up: the checkpoint's own meta must carry 410.
  local NW; NW=$($PY -c "import json;print(len(json.load(open('$D/bridges.npz.meta'))['write_dims'] or []))")
  [ "$NW" = "410" ] || { step "TRAIN $NAME wrote $NW dims, not 410"; return 1; }
  for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
    local S=${pair%%:*} F=${pair##*:}
    [ -s $D/eval_$S.json ] && { step "EVAL $NAME $S already done"; continue; }
    $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
        --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
        --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $NAME $S FAILED"; return 1; }
    step "EVAL $NAME $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
  done
}

guardrail() {       # $1 name -- same 100 cached items as vn / vn5 / all
  local NAME=$1 D=results/stage2c_${TAG}_$1 GT=const_${TAG}_$1 FLAG=--fresh
  local CACHE=results/bench/tasks_const_${TAG}_n100.json
  [ -s $CACHE ] || { step "GUARDRAIL $NAME no task cache; refusing to build a different one"; return 1; }
  [ -s results/bench/${GT}_guardrail_summary.json ] && { step "GUARDRAIL $NAME already done"; return 0; }
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CM --redteam-data $DATA_TE \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
    --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
    --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
  local i
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 100 $COMMON $FLAG >> results/bench/${GT}_run.log 2>&1 \
      && { step "GUARDRAIL $NAME COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${GT}_run.log | tail -8 >> $LOG; return 0; }
    step "GUARDRAIL $NAME attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  step "GUARDRAIL $NAME GAVE UP"; return 1
}

train_variant vn10    "$VN/layer$L.json:top410"         2 && guardrail vn10
train_variant vn10bot "$VN/botrank_layer$L.json:top410" 2 && guardrail vn10bot
$PY eval/const_refusal_mcnemar.py \
  --tags const_${TAG}_vn,const_${TAG}_vn5,const_${TAG}_all,const_${TAG}_vn10,const_${TAG}_vn10bot \
  --out results/constitution/refusal_mcnemar_${TAG}.json >> $LOG 2>&1
step "QWEN35-WIDTH COMPLETE"
