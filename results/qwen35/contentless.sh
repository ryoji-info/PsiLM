#!/bin/bash
# The CONTENTLESS control: is the refusal shift attributable to the constitution?
#
# Malla et al. (2609.06951) find steering's off-target movement lands on a fixed
# set of default sinks -- chiefly refusal -- largely regardless of what is
# steered, with a magnitude-matched contentless direction moving the same
# behaviours in the same order, and the pull strongest below 10B. A 9B backbone
# with refusal as the readout is precisely the case that needs the control before
# 0.660 -> 0.720 is credited to what the constitution says.
#
# The zeroed arm does not answer this. Zeroed asks whether the channel wrote
# anything; its KL to base is exactly 0.00000. This asks whether what it wrote
# mattered: the same coordinates, the same gate (bit-identical, asserted in
# eval/contentless_self_test.py), the same per-coordinate magnitude, a direction
# drawn from a seed instead of computed from the partner.
#
# Cheap, because the n=400 runs already hold their base and psilm rows and the
# harness skips (dataset, qid, arm) keys it has done: --resume with the arm list
# extended runs ONLY the new arm. About 1.3h per variant rather than 3.9.
#
# LAUNCH A COPY, NOT THIS FILE. Ends with "CONTENTLESS COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
RT=data/redteam_qwen35_n400.json
CACHE=results/bench/tasks_rt400_qwen35_n400.json
LOG=results/qwen35/contentless.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "CONTENTLESS WAITING for the 400-item red-team arms"
until grep -q "RT400 COMPLETE" results/qwen35/redteam400.log 2>/dev/null; do
  grep -q "GAVE UP" results/qwen35/redteam400.log 2>/dev/null \
    && { step "RT400 gave up; nothing to extend"; exit 1; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py' > /dev/null; do sleep 60; done

$PY eval/contentless_self_test.py >> $LOG 2>&1 \
  || { step "CONTENTLESS self-test FAILED; refusing to run the arm"; exit 1; }
step "CONTENTLESS self-test passed; extending the n=400 arms"

arm() {   # $1 variant
  local V=$1 D=results/stage2c_qwen35_$1 GT=const_qwen35_${1}_rt400
  [ -s results/bench/${GT}_guardrail.json ] || { step "ARM $V: no n=400 run to extend"; return 1; }
  local i
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 400 --resume \
        --tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
        --ckpt $D/bridges.npz --const-model $CM --redteam-data $RT \
        --datasets redteam --arms base,psilm,zeroed,contentless \
        --max-new-redteam 128 --kl --seed 0 --print-every 10 --save-every 10 \
        >> results/bench/${GT}_run.log 2>&1 \
      && { step "ARM $V COMPLETE (attempt $i)"; \
           $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; return 0; }
    step "ARM $V attempt $i exited; resuming"; sleep 30
  done
  step "ARM $V GAVE UP"; return 1
}

arm all
arm vn10e
$PY eval/const_refusal_mcnemar.py --pairs base:psilm,base:contentless,contentless:psilm \
  --tags const_qwen35_all_rt400,const_qwen35_vn10e_rt400 \
  --out results/constitution/refusal_mcnemar_contentless.json >> $LOG 2>&1
step "CONTENTLESS COMPLETE"
