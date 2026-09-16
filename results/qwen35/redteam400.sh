#!/bin/bash
# Resolve the transmission question by raising n, not by running more arms.
#
# The paired refusal test counts DISCORDANT pairs, so with an exact two-sided
# McNemar the floor is six clean one-directional flips: 6:0 gives p = 0.031 and
# 5:0 gives 0.0625, while 4:1 -- what the energy-matched 410-coordinate arm
# produced -- gives 0.375. At n = 100 that arm's result is unresolvable, and no
# number of additional arms changes it.
#
# n does. At a fixed per-prompt flip rate, 400 items yield four times the
# discordant pairs: the 410-coordinate arm's 4:1 becomes roughly 16:4, p = 0.012,
# and full width's 6:0 becomes roughly 24:0. The red-team loader reads ONLY
# prompt_text (eval/bench_common.py says so in as many words: the stored teacher
# and base continuations are training material and are not read at eval time), so
# the extra 300 prompts needed no teacher generation and no weights --
# eval/build_redteam_prompts.py built them on the CPU, deduplicated against all
# four training splits, with the first 100 items carried over verbatim so these
# runs stay comparable item-for-item with every recorded n = 100 run.
#
# Prompted by Malla et al. (2609.06951), Gao (2606.25487) and Miller
# (2411.00640) -- see the literature brief. Miller's point is the one acted on
# here; the other two need the contentless control and a judge, which are
# separate work.
#
# Red-team only: the benchmark arms are already resolved at n = 100 and cost
# three quarters of a guard-rail's wall clock.
#
# LAUNCH A COPY, NOT THIS FILE. Ends with "RT400 COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
RT=data/redteam_qwen35_n400.json
CACHE=results/bench/tasks_rt400_qwen35_n400.json
LOG=results/qwen35/redteam400.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "RT400 WAITING for the replicate chain"
MISS=0
until grep -q "QWEN35-REPLICATES COMPLETE" results/qwen35/constitution_replicates.log 2>/dev/null; do
  # An upstream chain that dies without writing its marker would leave this
  # sleeping forever, which defeats the point of an unattended queue. Three
  # consecutive checks with no upstream process and no marker means it is gone.
  if pgrep -f 'replicates_run.sh|constitution_replicates.sh' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "the replicate chain is neither running nor complete; proceeding"; break; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py' > /dev/null; do sleep 60; done

[ -s $RT ] || { step "PREREQ no $RT"; exit 1; }
N=$($PY -c "import json;print(len(json.load(open('$RT'))))")
[ "$N" = "400" ] || { step "PREREQ $RT has $N items, not 400"; exit 1; }
step "RT400 START n=400 (all, vn10e)"

arm() {   # $1 variant
  local V=$1 D=results/stage2c_qwen35_$1 GT=const_qwen35_${1}_rt400 FLAG=--fresh
  [ -s $D/bridges.npz ] || { step "ARM $V no checkpoint"; return 1; }
  [ -s results/bench/${GT}_guardrail_summary.json ] && { step "ARM $V already done"; return 0; }
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CM --redteam-data $RT \
    --datasets redteam --arms base,psilm,zeroed \
    --max-new-redteam 128 --kl --seed 0 --print-every 10 --save-every 10"
  if [ ! -s $CACHE ]; then
    $PY eval/bench_guardrail.py --tag ${GT}_cachebuild --n 400 $COMMON --build-cache \
        >> results/bench/rt400_cachebuild.log 2>&1 || { step "ARM $V cache FAILED"; return 1; }
    step "RT400 task cache built"
  fi
  local i
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 400 $COMMON $FLAG >> results/bench/${GT}_run.log 2>&1 \
      && { step "ARM $V COMPLETE (attempt $i)"; \
           $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; return 0; }
    step "ARM $V attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  step "ARM $V GAVE UP"; return 1
}

arm all
arm vn10e
$PY eval/const_refusal_mcnemar.py --pairs base:psilm,zeroed:psilm \
  --tags const_qwen35_all_rt400,const_qwen35_vn10e_rt400,const_qwen35_all,const_qwen35_vn10e \
  --out results/constitution/refusal_mcnemar_rt400.json >> $LOG 2>&1
step "RT400 COMPLETE"
