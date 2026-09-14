#!/bin/bash
# Constitution bridge, the 9B stage, part 1: everything that does not depend on
# how the 0.5B bridges turn out -- the value-neuron identification on the
# Qwen3.5 9B backbone and the 9B teacher data. Waits for the 0.5B chain
# (results/constitution/train_qwen0.5b.sh) to exit so the two never share the
# GPU: the Metal watchdog killed every 0.5B job that ran alongside another.
#
#   1. collect  800 GSM8K-TRAIN trajectories at the paper's sampler (temperature
#      1.0 / top-p 0.95): this backbone solves ~83% greedy, so sampling is what
#      balances the reward here (the 0.5B needed greedy for the opposite reason).
#      Layers 13 (the read depth) and 20..30 (candidate injection depths: the
#      physics campaign's memory probes put the cliff at 26 for batch 2, and the
#      constitution model adds ~1 GB on top). ~2 h, ~15 GB of float16 states.
#   2. probe    Monte-Carlo target, 30 epochs, candidates 24/26/28  (~1 h)
#   3. ablate   base / zeroV / zeroV5 / 3 random arms on GSM8K test n=100 (~1.5 h)
#   4. data     1,000 / 100 / 100 / 100 teacher-vs-base pairs at max-new 128 (~3.5 h)
# Training on the 9B is a separate script, launched after the 0.5B results are
# reviewed. Markers in results/qwen35/constitution_prep.log; ends with
# "QWEN35-PREP COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
TAG=qwen35
D=results/value_neurons/$TAG
LOG=results/qwen35/constitution_prep.log
LAYERS=13,20,22,24,26,28,30
CAND=24,26,28
export HF_HUB_DISABLE_XET=1          # deliberately NOT offline: see below
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
mkdir -p $D results/constitution
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

# Data-only entry point. The collect, probe and ablation stages are finished
# (results/value_neurons/qwen35/), so this runs the teacher-data build alone.
# It needs the hub reachable: in offline mode `datasets` cannot map
# data_dir="harmless-base" onto its cached config, because it looks for a
# config literally named default-data_dir=harmless-base while the cache holds
# hashes. That crash-looped this stage five times on 2026-09-13, and the 0.5B
# build only worked because its script never set the offline flags.
step "QWEN35-PREP DATA-ONLY RESUME"

step "DATA35 START"
ARGS="--tag $TAG --n-train 1000 --n-val 100 --n-test 100 --n-helpful-test 100 \
  --seed 7 --model $M --hf-tokenizer $M --max-new 128 \
  --excerpt data/constitution/system_excerpt.md --out-dir data --work-dir results/constitution --print-every 25"
for i in $(seq 1 30); do
  $PY eval/build_constitution_data.py $ARGS >> results/constitution/build_data_${TAG}_run.log 2>&1 && break
  step "DATA35 attempt $i exited $?; resuming"; sleep 30
  [ $i = 30 ] && { step "DATA35 GAVE UP"; exit 1; }
done
$PY -c "
import json
s = json.load(open('data/constitution_${TAG}_stats.json'))
for k, v in s['splits'].items():
    print('%-13s n=%-5d differs=%.3f refusal base=%.3f teacher=%.3f len base=%.1f teacher=%.1f'
          % (k, v['n'], v['teacher_differs_rate'], v['refusal_rate_base'],
             v['refusal_rate_teacher'], v['mean_len_base'], v['mean_len_teacher']))" >> $LOG 2>&1
step "DATA35 DONE"
step "QWEN35-PREP COMPLETE"
