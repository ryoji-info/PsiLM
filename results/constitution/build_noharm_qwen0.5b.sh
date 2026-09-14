#!/bin/bash
# No-harm negatives for the constitution bridge at the 0.5B: GSM8K-train and
# MMLU-validation prompts paired with the FROZEN 0.5B BACKBONE'S OWN greedy
# continuation, half with the benchmark's "Answer:" nudge line and half without
# (the Qwen3.5 recipe's two passes), merged into one file. Same builder, same
# exclusion of every benchmark test item, as results/qwen35/coupled_recipe.sh.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
T=Qwen/Qwen2.5-0.5B-Instruct
LOG=results/constitution/build_noharm_qwen0.5b.log
export HF_HUB_DISABLE_XET=1 HF_HOME=/Users/rxiii/Documents/huggingface
echo "NOHARM-0.5B START $(date +%H:%M)" >> $LOG
for pair in "nudge:1.0" "nonudge:0.0"; do
  NAME=${pair%%:*}; PROB=${pair##*:}
  $PY eval/build_noharm.py --model $M --hf-tokenizer $T --nudge-prob $PROB \
      --n-gsm8k 400 --n-mmlu 200 --max-new 32 --out data/noharm_qwen0.5b_$NAME.json >> $LOG 2>&1 \
    || { echo "NOHARM-0.5B $NAME FAILED $(date +%H:%M)" >> $LOG; exit 1; }
  echo "NOHARM-0.5B $NAME done $(date +%H:%M)" >> $LOG
done
$PY - <<'PY' >> $LOG 2>&1
import json
a = json.load(open('data/noharm_qwen0.5b_nudge.json')) + json.load(open('data/noharm_qwen0.5b_nonudge.json'))
json.dump(a, open('data/noharm_qwen0.5b_all.json', 'w'))
print('merged', len(a), 'items -> data/noharm_qwen0.5b_all.json')
PY
echo "NOHARM-0.5B COMPLETE $(date +%H:%M)" >> $LOG
