#!/bin/bash
# Does the partner's KNOWLEDGE of the constitution matter at 9B?
#
# At 0.5B a plain-partner control was run and came back negative: a partner that
# has never read the constitution matched or slightly beat one fine-tuned on it
# (mean dCE -0.0040 at nine written dimensions, 47 of 100 items; -0.0096 at full
# width, 63 of 100). The reading was that the document enters through the
# TEACHER's system prompt -- the backbone itself, prompted -- and the partner is a
# carrier rather than a source.
#
# That control was never run at 9B. All three recorded 9B variants used
# results/constitution_model/qwen2.5-0.5b-constitution, so the carrier reading is
# measured at 0.5B and inherited at 9B, where the only significant behavioural
# result in the project lives (full width, red-team refusal 0.660 -> 0.720, six
# flips toward refusal and none away, exact McNemar p = 0.031).
#
# This is that arm: the twin of `all`, byte-identical in every flag, with
# --const-model pointing at the stock instruct model instead of the fine-tune.
# Qwen2.5-0.5B-Instruct has d_model 896, the same as the fine-tuned partner --
# it IS the same base model -- so every bridge shape is unchanged and the only
# difference is whether the partner has read the document.
#
# It also gates the question of whether a LARGER partner would help. If knowledge
# is irrelevant at 9B too, the only remaining reason to grow the carrier is
# representational width, which is a different experiment; if the plain partner is
# clearly worse here, the 0.5B reading was scale-specific and the partner is doing
# real work.
#
# LAUNCH A COPY, NOT THIS FILE. Ends with "ALLPLAIN COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=Qwen/Qwen2.5-0.5B-Instruct          # the STOCK model: never read the constitution
TAG=qwen35
V=allplain
D=results/stage2c_${TAG}_$V
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
DATA_HE=data/constitution_helpful_test_$TAG.json
NEG=data/noharm_qwen35_all.json
RT=data/redteam_qwen35_n400.json
CACHE=results/bench/tasks_rt400_qwen35_n400.json
LOG=results/qwen35/allplain.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "ALLPLAIN WAITING for the physics attribution arm"
MISS=0
until grep -q "GUARDRAIL-PHYS COMPLETE" results/dual/guardrail_phys.log 2>/dev/null; do
  grep -q 'GUARDRAIL-PHYS GAVE UP' results/dual/guardrail_phys.log 2>/dev/null \
    && { step "the physics arm gave up; proceeding"; break; }
  # The physics arm is launched by the replicate chain, so before that lands
  # nothing upstream is running yet -- count the whole upstream queue, not just
  # the immediate predecessor, or this escapes during a legitimate gap.
  if pgrep -f 'guardrail_phys.sh|contentless_run.sh|rt400_run.sh|replicates_run.sh' > /dev/null; then
    MISS=0
  else
    MISS=$((MISS + 1))
  fi
  [ $MISS -ge 3 ] && { step "nothing upstream is running or complete; proceeding"; break; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py' > /dev/null; do sleep 60; done

mkdir -p $D
step "ALLPLAIN START partner=$CM (twin of the all arm, 2 chunks)"

# Byte-identical to the `all` arm in results/qwen35/constitution_train.sh except
# --const-model. Layer 24 < 26, so --checkpoint-from -1 as that arm had.
C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev 24 \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all --checkpoint-from -1"

DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
FLAG=""; [ "$DONE" = "0" ] && FLAG=--fresh
i=$DONE
while [ $i -lt 2 ]; do
  i=$((i + 1))
  for j in 1 2 3; do
    $PY eval/mlx_constitution_train.py --tag ${TAG}_$V --steps 500 $FLAG $C --write-dims all \
        >> $D/supervisor.log 2>&1 && break
    step "TRAIN chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
    [ $j = 3 ] && { step "TRAIN chunk $i FAILED"; exit 1; }
  done
  FLAG=""
  step "TRAIN chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
done

# The checkpoint must name the stock partner, or this silently became the all arm.
CHK=$($PY -c "
import json; m=json.load(open('$D/bridges.npz.meta'))
print('OK' if 'constitution' not in m['const_model'] else 'BAD ' + m['const_model'])")
[ "$CHK" = "OK" ] || { step "meta names the wrong partner: $CHK"; exit 1; }
step "partner check: $CHK ($($PY -c "import json;print(json.load(open('$D/bridges.npz.meta'))['const_model'])"))"

for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
  S=${pair%%:*}; F=${pair##*:}
  [ -s $D/eval_$S.json ] && continue
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
      --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
      --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $S FAILED"; exit 1; }
  step "EVAL $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
done

# n=400 red-team on the same cache as the other 400-item arms, so the refusal
# comparison against `all` is item-for-item.
GT=const_${TAG}_${V}_rt400
if [ ! -s results/bench/${GT}_guardrail_summary.json ]; then
  FLAG=--fresh
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 400 $FLAG \
        --tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
        --ckpt $D/bridges.npz --const-model $CM --redteam-data $RT \
        --datasets redteam --arms base,psilm,zeroed \
        --max-new-redteam 128 --kl --seed 0 --print-every 10 --save-every 10 \
        >> results/bench/${GT}_run.log 2>&1 \
      && { step "GUARDRAIL COMPLETE (attempt $i)"; \
           $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; break; }
    step "GUARDRAIL attempt $i exited; resuming"; FLAG=--resume; sleep 30
    [ $i = 20 ] && step "GUARDRAIL GAVE UP"
  done
fi
$PY eval/const_refusal_mcnemar.py --pairs base:psilm \
  --tags const_${TAG}_all_rt400,const_${TAG}_${V}_rt400 \
  --out results/constitution/refusal_mcnemar_allplain.json >> $LOG 2>&1
step "ALLPLAIN COMPLETE"
