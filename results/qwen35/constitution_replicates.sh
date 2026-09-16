#!/bin/bash
# An ERROR BAR on the identity effect at 410 coordinates.
#
# The identity comparison so far is one pair: the probe-best 410 (CE gain 0.0597
# at step 500) against the probe-worst 410 (0.0459), same count, same total write
# energy, disjoint. A single pair has no spread, so "+30% for choosing the probe's
# coordinates" is a point estimate with nothing under it.
#
# The draw that needs repeating is the CONTROL, and the probe-worst 410 is
# deterministic -- there is only one of it. What has a distribution is "410
# coordinates the probe did not pick", so that is what is sampled here.
#
# Uniform sampling was tried first and rejected: 410 coordinates drawn uniformly
# from outside the top 410 sum to 529-696 of RMS^2 against the probe-best 410's
# 1105, so energy parity would need caps of 0.66-0.75 against 0.5220, and the
# 410-coordinate arm established that MMLU collateral tracks per-coordinate
# magnitude rather than total energy. The probe's ranking is magnitude-biased at
# BOTH ends -- which is why the probe-worst 410 was a usable control (RMS^2 990.5)
# and a uniform draw is not. These three draws are matched one to one on
# activation RMS to the probe-best 410 instead: RMS^2 958-966, summed RMS within
# 1.8%, mean per-coordinate match error 0.027. The one coordinate at RMS 10.77
# cannot be matched (the pool's largest is 6.85) and accounts for about half the
# residual.
#
# So each arm matches the probe-best 410 on count, on total energy and on
# per-coordinate magnitude. Probe identity is the only difference left.
#
# 500 steps, one chunk, no guard-rail. The comparison is CE, which is precise and
# arrives with training; the behavioural comparison is underpowered at this effect
# size on 100 items and is not worth 5 hours per arm to restate. Four arms are
# already at step 500 on this schedule (base 0.4720, probe-best 0.4123,
# probe-worst 0.4261, whole stream 0.3914) and chunk-to-chunk movement past 500
# steps has been <= 0.0010 in all five arms measured, so step 500 is the
# comparison point.
#
# LAUNCH A COPY, NOT THIS FILE -- bash reads lazily by byte offset and editing a
# running chain makes it resume at a shifted position.
# Ends with "QWEN35-REPLICATES COMPLETE", then hands the GPU to the physics arm.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
VN=results/value_neurons/qwen35
TAG=qwen35
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
NEG=data/noharm_qwen35_all.json
LOG=results/qwen35/constitution_replicates.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "QWEN35-REPLICATES WAITING for the width chain"
until grep -q "QWEN35-WIDTH COMPLETE" results/qwen35/constitution_width.log 2>/dev/null; do
  grep -q "GAVE UP" results/qwen35/constitution_width.log 2>/dev/null \
    && { step "width chain gave up; running the replicates anyway"; break; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py' > /dev/null; do sleep 60; done

L=24
[ -s $VN/rms_layer${L}_full.json ] || { step "PREREQ no rms_layer${L}_full.json"; exit 1; }
step "QWEN35-REPLICATES START layer=$L width=410 (match0 0.5606, match1 0.5583, match2 0.5605)"

C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev $L \
  --gate-bias 0.0 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all --checkpoint-from -1"

replicate() {   # $1 seed, $2 cap
  local S=$1 CAP=$2 T=${TAG}_match$1 D=results/stage2c_${TAG}_match$1
  mkdir -p $D
  grep -q 'CHUNK DONE' $D/supervisor.log 2>/dev/null && { step "REP $S already trained"; return 0; }
  local j
  for j in 1 2 3; do
    $PY eval/mlx_constitution_train.py --tag $T --steps 500 --fresh $C \
        --write-dims "$VN/match${S}_layer$L.json:top410" --inj-cap $CAP \
        >> $D/supervisor.log 2>&1 && break
    step "REP $S attempt $j exited $?; resuming"; sleep 30
    [ $j = 3 ] && { step "REP $S FAILED"; return 1; }
  done
  local CHK; CHK=$($PY - "$D/bridges.npz.meta" "$CAP" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
n, cap = len(m["write_dims"] or []), float(m["inj_cap"])
print("OK" if n == 410 and abs(cap - float(sys.argv[2])) < 1e-9 else f"BAD dims={n} cap={cap}")
PYEOF
)
  [ "$CHK" = "OK" ] || { step "REP $S meta wrong: $CHK"; return 1; }
  step "REP $S: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  [ -s $D/eval_test.json ] || $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz \
      --model $M --hf-tokenizer $M --const-model $CM --data $DATA_TE \
      --arms base,psilm,zeroed --n 50 --max-new 24 --out $D/eval_test.json > $D/eval_test.log 2>&1
  step "REP $S test: $(grep -E '^ +(base|psilm) ' $D/eval_test.log | tr -s ' ' | tr '\n' '|')"
}

replicate 0 0.5606
replicate 1 0.5583
replicate 2 0.5605
$PY eval/vn_identity_spread.py >> $LOG 2>&1
step "QWEN35-REPLICATES COMPLETE"
# The physics attribution arm was holding the GPU queue behind this; hand it over.
nohup bash results/dual/guardrail_phys.sh > /dev/null 2>&1 &
