#!/bin/bash
# The plain-partner control at the two narrow widths, completing a 3x2 design.
#
# results/qwen35/allplain.sh runs the twin of `all` (4096 coordinates). This runs
# the twins of `vn` (41, the top 1% of value neurons) and `vn5` (205, the top 5%),
# so every recorded 9B width has a partner-read-it and a partner-never-read-it arm.
#
# Why width matters for this question and not only for the write. At 0.5B the
# partner effect was width-dependent and the direction was against the fine-tune:
# at nine written dimensions the plain partner had LOWER held-out CE by 0.0040
# with a bootstrap interval excluding zero, while at full width the difference was
# -0.0096 with an interval spanning it. If that structure repeats at 9B, the
# fine-tuned partner is actively costing something where the channel is most
# constrained, which is the opposite of what a carrier-with-more-knowledge is
# supposed to do, and it bears directly on whether a LARGER partner would help.
#
# Names follow this project's width-first convention rather than 0.5B's: the 0.5B
# `plainpartner` arm was its nine-dimension twin, which is `vnplain` here.
#
# No guard-rail on these two. The recorded vn and vn5 arms moved nothing
# behaviourally -- 1:1 and 3:2 churn at p = 1.00 -- so their twins cannot be
# compared on behaviour with any power, and CE (which arrives with training) is
# the better-powered comparison at these widths. If either shows a partner effect
# on CE, guard-railing it then is a separate 3.9h and worth it.
#
# Two chunks each, matching the recorded arms' step 1000. Not one chunk: drift
# from step 500 to 1000 is small but asymmetric between arms (-0.0007 to +0.0015
# across the five measured), and these are compared against step-1000 records.
#
# LAUNCH A COPY, NOT THIS FILE. Ends with "PLAINWIDTHS COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=Qwen/Qwen2.5-0.5B-Instruct          # the STOCK model: never read the constitution
VN=results/value_neurons/qwen35
TAG=qwen35
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
DATA_HE=data/constitution_helpful_test_$TAG.json
NEG=data/noharm_qwen35_all.json
LOG=results/qwen35/plainpartner_widths.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "PLAINWIDTHS WAITING for the full-width plain-partner arm"
MISS=0
until grep -q "ALLPLAIN COMPLETE" results/qwen35/allplain.log 2>/dev/null; do
  if pgrep -f 'allplain_run.sh|allplain.sh' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "the allplain chain is neither running nor complete; proceeding"; break; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py' > /dev/null; do sleep 60; done

step "PLAINWIDTHS START partner=$CM (vnplain 41, vn5plain 205)"

C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev 24 \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all --checkpoint-from -1"

variant() {   # $1 name, $2 --write-dims spelling, $3 expected dim count
  local V=$1 WD=$2 WANT=$3 D=results/stage2c_${TAG}_$1
  mkdir -p $D
  local DONE; DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
  local FLAG=""; [ "$DONE" = "0" ] && FLAG=--fresh
  step "TRAIN $V: $DONE of 2 chunks done (write-dims $WD)"
  local i=$DONE j
  while [ $i -lt 2 ]; do
    i=$((i + 1))
    for j in 1 2 3; do
      $PY eval/mlx_constitution_train.py --tag ${TAG}_$V --steps 500 $FLAG $C \
          --write-dims "$WD" >> $D/supervisor.log 2>&1 && break
      step "TRAIN $V chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
      [ $j = 3 ] && { step "TRAIN $V chunk $i FAILED"; return 1; }
    done
    FLAG=""
    step "TRAIN $V chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  done
  # Both halves of what makes this a control: the width it claims, and a partner
  # that has not read the document.
  local CHK; CHK=$($PY - "$D/bridges.npz.meta" "$WANT" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
n = len(m["write_dims"] or [])
cm = m["const_model"]
bad = []
if n != int(sys.argv[2]):
    bad.append(f"dims={n}")
if "constitution" in cm:
    bad.append(f"partner={cm}")
print("OK" if not bad else "BAD " + " ".join(bad))
PYEOF
)
  [ "$CHK" = "OK" ] || { step "TRAIN $V meta wrong: $CHK"; return 1; }
  for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
    local S=${pair%%:*} F=${pair##*:}
    [ -s $D/eval_$S.json ] && continue
    $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
        --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
        --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $V $S FAILED"; return 1; }
    step "EVAL $V $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
  done
}

variant vnplain  "$VN/layer24.json"         41
variant vn5plain "$VN/layer24.json:top5pct" 205
$PY eval/vn_partner_effect.py >> $LOG 2>&1
step "PLAINWIDTHS COMPLETE"
