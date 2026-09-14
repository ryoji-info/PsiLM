#!/bin/bash
# Constitution self-distillation data on the dev backbone (Qwen2.5-0.5B-4bit).
#
#   nohup bash results/constitution/build_data_qwen0.5b.sh &
#
# 2,400 prompts x two greedy continuations of up to 160 tokens. The smoke run
# measured 1.74 s/item at 94 tok/s with the teacher's 3,391-token system block
# prefilled once and copied per prompt, so budget ~70 minutes and expect the
# estimate in the stats file to be the one to trust on the day. It loads its own
# copy of the 0.5B, so it can share the machine, but it does queue GPU work for
# an hour: start it when nothing else needs the GPU.
#
# Resumable: each split appends to results/constitution/qwen0.5b_<split>.jsonl
# keyed by source, and a rerun with the same --tag picks up whatever is missing.
# The retry loop below exists because long MLX runs on this machine have been
# killed by memory pressure before; the jsonl is the checkpoint, and the JSON
# lists under data/ are reassembled from it every time.
#
# The four splits, and why: train/val are disjoint seeded samples of
# harmless-base TRAIN, test is harmless-base TEST (the guard-rail's redteam set)
# and helpful_test is helpful-base TEST, the ordinary-request control that says
# whether the bridges damage normal chat. Roughly 90% of harmless-base TEST
# opening turns also occur in TRAIN and are dropped, leaving ~210 clean ones, so
# --n-test cannot go much above 200.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
TAG=qwen0.5b
LOG=results/constitution/build_data_$TAG.log
export HF_HUB_DISABLE_XET=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
mkdir -p results/constitution data
echo "CONSTITUTION-DATA START $(date +%H:%M)" >> $LOG

ARGS="--tag $TAG --n-train 2000 --n-val 200 --n-test 100 --n-helpful-test 100 \
  --seed 7 --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --hf-tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-new 160 \
  --excerpt data/constitution/system_excerpt.md \
  --out-dir data --work-dir results/constitution --print-every 50"

for i in $(seq 1 20); do
  $PY eval/build_constitution_data.py $ARGS >> results/constitution/build_data_${TAG}_run.log 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    $PY -c "
import json
s = json.load(open('data/constitution_${TAG}_stats.json'))
for k, v in s['splits'].items():
    print('%-13s n=%-5d differs=%.3f refusal base=%.3f teacher=%.3f len base=%.1f teacher=%.1f'
          % (k, v['n'], v['teacher_differs_rate'], v['refusal_rate_base'],
             v['refusal_rate_teacher'], v['mean_len_base'], v['mean_len_teacher']))
print('cache_check', s['cache_check'], 'uncached', s['uncached_fallbacks'])
print('timing', s['timing'])" >> $LOG 2>&1
    echo "CONSTITUTION-DATA DONE (attempt $i) $(date +%H:%M)" >> $LOG
    exit 0
  fi
  echo "CONSTITUTION-DATA attempt $i exited $RC; resuming $(date +%H:%M)" >> $LOG
  sleep 20
done
echo "CONSTITUTION-DATA GAVE UP $(date +%H:%M)" >> $LOG
exit 1
