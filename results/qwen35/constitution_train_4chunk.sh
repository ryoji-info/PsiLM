#!/bin/bash
# Constitution bridge, the 9B stage, part 2: train the write-location variants
# on Qwen3.5 9B, evaluate each against its constitution-conditioned teacher, and
# guard-rail each. Waits for results/qwen35/constitution_prep.sh ("QWEN35-PREP
# COMPLETE": the 9B value neurons in results/value_neurons/qwen35/ and the 9B
# teacher data in data/constitution_*_qwen35.json).
#
# Variants, in the order the 0.5B results argue for: vn (top-1% value neurons,
# 41 of 4096 dims at the chosen layer -- the requested design), all (the whole
# stream: at 0.5B it carried the teacher but leaked into GSM8K), vn5 (top-5%,
# 205 dims: the width the 0.5B trade-off points at). Each: 4 chunks x 500 steps
# at batch 2 (the Qwen3.5 physics campaign's batch; ~5 s/step, ~3 h per variant),
# lr 3e-4, the same gate/cap/clip/no-harm settings as the 0.5B chain, negatives
# data/noharm_qwen35_all.json (the physics campaign's). --checkpoint-from -1 when
# the chosen layer is below 26 (the memory cliff measured for this backbone).
# Eval: --n 50 --max-new 24 (the evaluator recomputes the full sequence per
# token; the guard-rail's redteam arm gives the 128-token cached generations).
# Ends with "QWEN35-TRAIN COMPLETE". Markers in results/qwen35/constitution_train.log.
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
LOG=results/qwen35/constitution_train.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "QWEN35-TRAIN WAITING for the prep chain"
until grep -q "QWEN35-PREP COMPLETE" results/qwen35/constitution_prep.log 2>/dev/null; do
  grep -q "FAILED\|GAVE UP\|NO CHOSEN" results/qwen35/constitution_prep.log 2>/dev/null && { step "PREREQ prep chain failed"; exit 1; }
  sleep 120
done
# 'bash ...': a bare path also matches a tail -f on the log. No '.sh': the prep
# chain may be running from its resume entry point (constitution_prep_resume.sh).
while pgrep -f 'bash results/qwen35/constitution_prep' > /dev/null; do sleep 30; done

# The magnitude-matched ablation that used to gate this step was cancelled on
# 2026-09-13: the 9B ablation damaged nothing (41 value neurons zeroed cost 3
# points at p=0.375), so there was no damage for a matched control to attribute.
# See results/qwen35/vn35_magmatch.log.
[ -s $DATA_TR ] && [ -s $DATA_VA ] && [ -s $CM/model.safetensors ] || { step "PREREQ files missing"; exit 1; }
L=$($PY -c "import json;print(json.load(open('$VN/summary.json'))['chosen_layer'])")
[ "$L" = "None" ] && { step "PREREQ no chosen layer"; exit 1; }
N1=$($PY -c "import json;print(len(json.load(open('$VN/layer$L.json'))['top1pct']))")
CKPT=""; [ "$L" -lt 26 ] && CKPT="--checkpoint-from -1"
step "QWEN35-TRAIN START layer=$L n_top1=$N1 $CKPT"

C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev $L \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all $CKPT"

train_variant() {   # $1 name, $2 --write-dims spelling
  local NAME=$1 WD=$2 T=${TAG}_$1 D=results/stage2c_${TAG}_$1 FLAG=--fresh
  mkdir -p $D
  for i in 1 2 3 4; do
    for j in 1 2 3; do
      $PY eval/mlx_constitution_train.py --tag $T --steps 500 $FLAG $C --write-dims "$WD" \
          >> $D/supervisor.log 2>&1 && break
      step "TRAIN $NAME chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
      [ $j = 3 ] && { step "TRAIN $NAME chunk $i FAILED"; return 1; }
    done
    FLAG=""
    step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  done
  for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
    local S=${pair%%:*} F=${pair##*:}
    $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
        --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
        --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $NAME $S FAILED"; return 1; }
    step "EVAL $NAME $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
  done
}

guardrail() {       # $1 name
  local NAME=$1 D=results/stage2c_${TAG}_$1 GT=const_${TAG}_$1 FLAG=--fresh
  local CACHE=results/bench/tasks_const_${TAG}_n100.json
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CM --redteam-data $DATA_TE \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
    --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
    --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
  if [ ! -s $CACHE ]; then
    $PY eval/bench_guardrail.py --tag ${GT}_cachebuild --n 100 $COMMON --build-cache \
        >> results/bench/const_${TAG}_cachebuild.log 2>&1 || { step "GUARDRAIL $NAME cache FAILED"; return 1; }
  fi
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 100 $COMMON $FLAG >> results/bench/${GT}_run.log 2>&1 \
      && { step "GUARDRAIL $NAME COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${GT}_run.log | tail -8 >> $LOG; return 0; }
    step "GUARDRAIL $NAME attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  step "GUARDRAIL $NAME GAVE UP"; return 1
}

train_variant vn  "$VN/layer$L.json" && guardrail vn
train_variant all all                && guardrail all
train_variant vn5 "$VN/layer$L.json:top5pct" && guardrail vn5
step "QWEN35-TRAIN COMPLETE"
