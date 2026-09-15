#!/bin/bash
# Is the 9B write mask's width a DIMENSIONALITY limit or a BUDGET limit?
#
# The three recorded widths were read as "only full width transmits decisions".
# That reading does not survive looking at what the injection cap does. From
# MaskedGatedCrossAttentionMLX.__call__, the cap bounds the injection's
# per-written-dimension RMS, so it is indifferent to how many dimensions are
# written, and every recorded arm saturated it exactly:
#
#   arm   dims   ratio_gen/gate_gen   Sum rms^2 over written dims   energy vs full
#   vn      41   0.2000                286.5                        3.8%
#   vn5    205   0.2000                769.2                       10.2%
#   all   4096   0.2000               7527.8                      100.0%
#
# (magnitudes: results/value_neurons/qwen35/rms_layer24_full.json; the ratio is
# sigma * per-dim injection RMS, so ratio/gate recovers the cap.)
#
# Full width was therefore handed 26x the perturbation energy of the top-5%
# mask, and the widths were never compared at equal strength. Per unit of
# energy actually spent, the narrow masks did MORE: CE gain per unit relative
# energy 0.334 (41 dims) and 0.271 (205) against 0.094 for full width, at lower
# total KL. The value-neuron write is the efficient one; it was starved.
#
# So the useful pair is at MATCHED energy, cap raised to restore parity with
# full width -- 0.2 * sqrt(E_full / E_set):
#
#   vn10e     probe-best  410 dims, --inj-cap 0.5220   (energy 0.1468 of full)
#   vn10ebot  probe-worst 410 dims, --inj-cap 0.5514   (energy 0.1316 of full)
#
# Both then carry full width's total write energy, on 10% of the dimensions,
# with per-dimension writes 2.6x larger. 410 because the probe's AUC holds flat
# from 4096 down to ~200 dims and breaks below that at BOTH scales measured
# (9B keeps 0.82-0.85 down to 205 then falls to 0.77 at 82; 0.5B keeps
# 0.66-0.69 down to 179 then falls to 0.63 at 45) -- an absolute knee near
# 100-200 dimensions, not a constant fraction. And at 410 the identity control
# is free: summed RMS 553.17 against 549.31, disjoint, no attention sink.
#
# Reading the outcome, against all (4096 dims, cap 0.2, CE 0.3890, red-team KL
# 0.1047, refusal 0.720 = 6 flips toward refusal and 0 against, p = 0.031):
#
#   vn10e ~ all, vn10ebot << vn10e   10% of the dimensions suffice AND it has to
#                                    be the probe's dimensions. The narrow
#                                    bridge is real and the value neurons are why.
#   vn10e ~ vn10ebot ~ all           any 410 dimensions will do at this energy;
#                                    width was always budget, identity is inert.
#   vn10e << all                     dimensionality does bind, and full width is
#                                    not substitutable. The original reading holds.
#
# What could spoil it, and is measured rather than assumed: a 2.6x larger
# per-dimension write may leave the residual stream's own distribution, which
# would show up as collateral on GSM8K / MMLU / BoolQ, where every recorded arm
# sat at KL <= 4.5e-3. A narrow bridge that transmits by damaging the backbone
# is not tight alignment, and the guard-rail arms are what say which it is.
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
[ "$L" = "24" ] || { step "PREREQ chosen layer is $L, not 24: the masks and the energy parity are layer 24"; exit 1; }
[ -s $VN/botrank_layer$L.json ] || { step "PREREQ no botrank_layer$L.json"; exit 1; }
[ -s $VN/rms_layer${L}_full.json ] || { step "PREREQ no rms_layer${L}_full.json to check parity against"; exit 1; }
CKPT=""; [ "$L" -lt 26 ] && CKPT="--checkpoint-from -1"
step "QWEN35-WIDTH START layer=$L width=410 energy-matched $CKPT (vn10e 2 chunks, vn10ebot 2 chunks)"

# Byte-identical to results/qwen35/constitution_train.sh's C except --inj-cap,
# which each variant passes itself.
C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev $L \
  --gate-bias 0.0 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all $CKPT"

train_variant() {   # $1 name, $2 --write-dims spelling, $3 --inj-cap, $4 chunks wanted
  local NAME=$1 WD=$2 CAP=$3 WANT=$4 T=${TAG}_$1 D=results/stage2c_${TAG}_$1
  mkdir -p $D
  local DONE; DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
  local FLAG=""; [ "$DONE" = "0" ] && FLAG=--fresh
  step "TRAIN $NAME: $DONE of $WANT chunks done (write-dims $WD, inj-cap $CAP)"
  # NOT `seq $((DONE+1)) $WANT`: BSD seq counts DOWN when the first argument
  # exceeds the second, so a finished variant would train two more chunks.
  local i=$DONE j
  while [ $i -lt $WANT ]; do
    i=$((i + 1))
    for j in 1 2 3; do
      $PY eval/mlx_constitution_train.py --tag $T --steps 500 $FLAG $C \
          --write-dims "$WD" --inj-cap $CAP >> $D/supervisor.log 2>&1 && break
      step "TRAIN $NAME chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
      [ $j = 3 ] && { step "TRAIN $NAME chunk $i FAILED"; return 1; }
    done
    FLAG=""
    step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  done
  [ -s $D/bridges.npz ] || { step "TRAIN $NAME no checkpoint to evaluate"; return 1; }
  # The checkpoint's own meta must carry the mask and the cap this arm claims:
  # the evaluator and the guard-rail read both from it, so a silent fallback to
  # 0.2 would quietly turn this back into the starved experiment.
  local CHK; CHK=$($PY - "$D/bridges.npz.meta" "$CAP" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
n, cap = len(m["write_dims"] or []), float(m["inj_cap"])
print("OK" if n == 410 and abs(cap - float(sys.argv[2])) < 1e-9 else f"BAD dims={n} cap={cap}")
PYEOF
)
  [ "$CHK" = "OK" ] || { step "TRAIN $NAME meta wrong: $CHK"; return 1; }
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

train_variant vn10e    "$VN/layer$L.json:top410"         0.5220 2 && guardrail vn10e
train_variant vn10ebot "$VN/botrank_layer$L.json:top410" 0.5514 2 && guardrail vn10ebot
$PY eval/const_refusal_mcnemar.py \
  --tags const_${TAG}_vn,const_${TAG}_vn5,const_${TAG}_all,const_${TAG}_vn10e,const_${TAG}_vn10ebot \
  --out results/constitution/refusal_mcnemar_${TAG}.json >> $LOG 2>&1
step "QWEN35-WIDTH COMPLETE"
