#!/bin/bash
# Is the near-zero collateral a property of "not the probe's coordinates", or of
# one particular set? -- now with the replicates brought to the same step as the
# arms they are compared against.
#
# The energy-matched pair says coordinate identity, not magnitude, decides
# collateral: the probe-worst 410 (cap 0.5514, written HARDER than the probe-best
# 410's 0.5220) produced MMLU KL 0.00024 against 0.02441. A 102x separation on a
# precise per-token measure. The three matched replicates decide whether that is
# the probe's coordinates or one coordinate set.
#
# Why a second chunk first. The replicates were trained for 500 steps; vn10e and
# vn10ebot were guard-railed at step 1000, and no arm anywhere has a step-500
# guard-rail to calibrate the gap. Drift from 500 to 1000 is small in CE but
# asymmetric between arms (-0.0007 to +0.0015 across five), and its sign on MMLU
# KL is unmeasured. Benching step-500 checkpoints against step-1000 references
# would put an uncalibrated confound under the decisive test. So each replicate
# resumes for one more chunk (~3.6h), the checkpoint meta is checked for step
# 1000, and only then is it guard-railed. Recommended by the 2026-09-17 review;
# the alternative was to caveat the comparison rather than make it.
#
# redteam and mmlu only: gsm8k at 384 tokens dominates a full guard-rail's wall
# clock and sits at 1e-4 in every arm, boolq likewise. The subset task cache's
# items were verified byte-identical in order to the 4-dataset cache's.
#
# LAUNCH A COPY, NOT THIS FILE. Ends with "REPGUARD COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
VN=results/value_neurons/qwen35
TAG=qwen35
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
NEG=data/noharm_qwen35_all.json
CACHE=results/bench/tasks_repguard_qwen35_n100.json
LOG=results/qwen35/replicate_guardrails.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "REPGUARD WAITING for the replicate chain"
MISS=0
until grep -q "QWEN35-REPLICATES COMPLETE" results/qwen35/constitution_replicates.log 2>/dev/null; do
  if pgrep -f 'replicates_run.sh|constitution_replicates.sh' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "the replicate chain is neither running nor complete; proceeding"; break; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py' > /dev/null; do sleep 60; done

[ -s $CACHE ] || { step "PREREQ no $CACHE"; exit 1; }
step "REPGUARD START (chunk 2 for match0/1/2, then redteam+mmlu guard-rails, n=100)"

# Byte-identical to results/qwen35/constitution_replicates.sh's C.
C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev 24 \
  --gate-bias 0.0 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all --checkpoint-from -1"

chunk2() {   # $1 seed, $2 cap -- resume the replicate for its second chunk
  local S=$1 CAP=$2 T=${TAG}_match$1 D=results/stage2c_${TAG}_match$1
  local DONE; DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
  [ "$DONE" -ge 2 ] && { step "CHUNK2 match$S already at step 1000"; return 0; }
  [ "$DONE" = "1" ] || { step "CHUNK2 match$S has $DONE chunks, expected 1; refusing"; return 1; }
  local j
  for j in 1 2 3; do
    $PY eval/mlx_constitution_train.py --tag $T --steps 500 $C \
        --write-dims "$VN/match${S}_layer24.json:top410" --inj-cap $CAP \
        >> $D/supervisor.log 2>&1 && break
    step "CHUNK2 match$S attempt $j exited $?; resuming"; sleep 30
    [ $j = 3 ] && { step "CHUNK2 match$S FAILED"; return 1; }
  done
  local STEP; STEP=$($PY -c "import json;print(json.load(open('$D/bridges.npz.meta'))['step'])")
  [ "$STEP" = "1000" ] || { step "CHUNK2 match$S meta says step $STEP, not 1000"; return 1; }
  step "CHUNK2 match$S: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
}

arm() {   # $1 variant
  local V=$1 D=results/stage2c_qwen35_$1 GT=const_qwen35_${1}_rg FLAG=--fresh
  [ -s $D/bridges.npz ] || { step "ARM $V no checkpoint; skipping"; return 0; }
  [ -s results/bench/${GT}_guardrail_summary.json ] && { step "ARM $V already done"; return 0; }
  local i
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 100 $FLAG \
        --tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
        --ckpt $D/bridges.npz --const-model $CM \
        --redteam-data data/constitution_test_qwen35.json \
        --datasets redteam,mmlu --arms base,psilm,zeroed \
        --max-new-mmlu 256 --max-new-redteam 128 --kl --seed 0 --print-every 10 --save-every 10 \
        >> results/bench/${GT}_run.log 2>&1 \
      && { step "ARM $V COMPLETE (attempt $i)"; \
           $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; return 0; }
    step "ARM $V attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  step "ARM $V GAVE UP"; return 1
}

chunk2 0 0.5606 && arm match0
chunk2 1 0.5583 && arm match1
chunk2 2 0.5605 && arm match2
$PY eval/vn_identity_spread.py --step 1000 \
  --out results/constitution/identity_spread_qwen35.json >> $LOG 2>&1
step "REPGUARD COMPLETE"
