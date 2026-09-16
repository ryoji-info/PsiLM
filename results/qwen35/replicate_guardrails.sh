#!/bin/bash
# Is the near-zero collateral a property of "not the probe's coordinates", or of
# one particular set?
#
# The energy-matched pair says coordinate identity, not magnitude, decides
# collateral: the probe-worst 410 (cap 0.5514, written HARDER than the probe-best
# 410's 0.5220) produced MMLU KL 0.00024 against 0.02441 and moved no MMLU item,
# while delivering nominally more refusal movement. That is a 102x separation on a
# precise per-token measure.
#
# The three matched replicates were deliberately left without guard-rails, on the
# argument that behaviour was underpowered at this effect size. That argument was
# about REFUSAL, where it still holds, and it was wrong about MMLU divergence,
# which separates arms by two orders of magnitude and is the most discriminating
# number in the campaign. So the replicates get the two datasets that discriminate.
#
# redteam and mmlu only: gsm8k at 384 tokens dominates a full guard-rail's wall
# clock and sits at 1e-4 in every arm, and boolq likewise. The subset builds its
# own task cache because the cache key includes the dataset list, and the items are
# verified byte-identical in order to the 4-dataset cache's for both datasets --
# checked with --build-cache, which loads no weights.
#
# LAUNCH A COPY, NOT THIS FILE. Ends with "REPGUARD COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
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
step "REPGUARD START (match0, match1, match2; redteam+mmlu, n=100)"

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

arm match0
arm match1
arm match2
$PY eval/vn_identity_spread.py --step 500 \
  --out results/constitution/identity_spread_step500.json >> $LOG 2>&1
step "REPGUARD COMPLETE"
