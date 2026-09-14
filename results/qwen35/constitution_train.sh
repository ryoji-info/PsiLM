#!/bin/bash
# Constitution bridge, the 9B stage, part 2 (TRIMMED, 2026-09-13 -- supersedes
# constitution_train_4chunk.sh, which is the original kept verbatim for the record).
#
# Why the trim. The original ran 4 chunks x 500 steps for each of the three
# variants on the estimate of ~5 s/step measured in the Qwen3.5 physics campaign.
# The constitution bridge on this backbone actually runs at 26.7 s/step -- peak
# 48.9 GB against 25.8 GB of physical RAM, so memory compression and ~1.6 GB of
# swap pay for the difference -- which makes a chunk 3h43m and the original plan
# ~44 h of training plus ~20 h of evaluation. At 0.5B the narrow writes had
# already converged after one chunk and only the full-width write kept improving:
#
#   vn  (9 dims)    0.9623 0.9607 0.9588 0.9615   gained 0.0008 after chunk 1
#   vn5 (45)        0.9270 0.9272 0.9286 0.9221   gained 0.0049
#   all (896)       0.6715 0.6440 0.6300 0.6283   gained 0.0432
#
# So chunks 3-4 of a narrow variant buy ~0.001 of held-out cross-entropy for
# 7.5 h of GPU here. This script gives the narrow variants (vn, vn5) 2 chunks --
# two points is enough to show the plateau rather than assume it. Evaluations and
# guard-rails are unchanged, so every reported number is still produced by the
# same harness on the same held-out hundreds.
#
# REVISED 2026-09-14: 'all' was given 4 chunks on the 0.5B precedent above, where
# the full-width write was the one that kept improving. On this backbone it does
# not: its chunk 1 -> 2 moved held-out CE by 0.0010 (0.3914 -> 0.3904), flatter
# than the 0.5B's full-width write was at chunk FOUR (-0.0017). So 'all' is
# trimmed to 2 as well, by the same measured criterion. Its 4-chunk allocation is
# recoverable -- raise the count and relaunch, and it resumes from bridges.npz.
#
# Chunks already on disk are counted from each variant's supervisor.log and not
# repeated, so this is safe to relaunch after an interruption. Variant order is
# unchanged (vn first: it is the requested design). Ends with the same
# "QWEN35-TRAIN COMPLETE" marker that results/constitution/magmatch_more_qwen0.5b.sh
# waits on, and lives at the same path so that script's pgrep guard still matches.
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

# Never two MLX jobs on one GPU: the Metal watchdog kills both. Wait out any
# trainer or evaluator still finishing from the superseded chain.
while pgrep -f 'eval/mlx_constitution_(train|eval)\.py|eval/bench_guardrail\.py' > /dev/null; do sleep 20; done

[ -s $DATA_TR ] && [ -s $DATA_VA ] && [ -s $DATA_TE ] && [ -s $DATA_HE ] \
  && [ -s $CM/model.safetensors ] || { step "PREREQ files missing"; exit 1; }
L=$($PY -c "import json;print(json.load(open('$VN/summary.json'))['chosen_layer'])")
[ "$L" = "None" ] && { step "PREREQ no chosen layer"; exit 1; }
N1=$($PY -c "import json;print(len(json.load(open('$VN/layer$L.json'))['top1pct']))")
CKPT=""; [ "$L" -lt 26 ] && CKPT="--checkpoint-from -1"
step "QWEN35-TRAIN TRIMMED RESUME layer=$L n_top1=$N1 $CKPT (vn 2, all 4, vn5 2 chunks)"

C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev $L \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all $CKPT"

train_variant() {   # $1 name, $2 --write-dims spelling, $3 chunks wanted
  local NAME=$1 WD=$2 WANT=$3 T=${TAG}_$1 D=results/stage2c_${TAG}_$1
  mkdir -p $D
  # Chunks already finished, counted from the log the trainer itself writes.
  local DONE; DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
  local FLAG=""; [ "$DONE" = "0" ] && FLAG=--fresh
  step "TRAIN $NAME: $DONE of $WANT chunks already done"
  # NOT `for i in $(seq $((DONE+1)) $WANT)`: BSD seq counts DOWN when the first
  # argument exceeds the second, so `seq 3 2` prints "3 2" here where GNU seq
  # prints nothing -- an already-complete variant would train two more chunks.
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
  for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
    local S=${pair%%:*} F=${pair##*:}
    [ -s $D/eval_$S.json ] && { step "EVAL $NAME $S already done"; continue; }
    $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
        --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
        --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $NAME $S FAILED"; return 1; }
    step "EVAL $NAME $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
  done
}

guardrail() {       # $1 name
  local NAME=$1 D=results/stage2c_${TAG}_$1 GT=const_${TAG}_$1 FLAG=--fresh
  local CACHE=results/bench/tasks_const_${TAG}_n100.json
  mkdir -p results/bench
  [ -s results/bench/${GT}_guardrail_summary.json ] && { step "GUARDRAIL $NAME already done"; return 0; }
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CM --redteam-data $DATA_TE \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
    --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
    --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
  if [ ! -s $CACHE ]; then
    $PY eval/bench_guardrail.py --tag ${GT}_cachebuild --n 100 $COMMON --build-cache \
        >> results/bench/const_${TAG}_cachebuild.log 2>&1 || { step "GUARDRAIL $NAME cache FAILED"; return 1; }
  fi
  local i
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 100 $COMMON $FLAG >> results/bench/${GT}_run.log 2>&1 \
      && { step "GUARDRAIL $NAME COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${GT}_run.log | tail -8 >> $LOG; return 0; }
    step "GUARDRAIL $NAME attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  step "GUARDRAIL $NAME GAVE UP"; return 1
}

train_variant vn  "$VN/layer$L.json"         2 && guardrail vn
train_variant all all                        2 && guardrail all
train_variant vn5 "$VN/layer$L.json:top5pct" 2 && guardrail vn5
step "QWEN35-TRAIN COMPLETE"
