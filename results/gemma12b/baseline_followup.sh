#!/bin/bash
# After the chain's forced-baseline reruns: assess each zero, and retry the ones
# that are zero under a deliberately generous protocol (four solved examples for
# format, twice the token budget, thinking on where the backbone has it) before
# any number goes into a model card.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
LOG=results/gemma12b/baseline_followup.log
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
until grep -q "CHAIN COMPLETE" results/gemma12b/chain.log 2>/dev/null; do sleep 60; done
echo "FOLLOWUP START $(date +%H:%M)" >> $LOG

acc_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['baseline']['acc'])" "$1"; }

run() {   # dir  model  tokenizer  tag  extra-flags
  D=$1; M=$2; T=$3; TAG=$4; shift 4
  F=$D/final_eval_baseline_forced.json
  [ -f "$F" ] || { echo "MISSING $F" >> $LOG; return; }
  echo "--- $TAG forced protocol" >> $LOG
  $PY eval/assess_baseline.py "$F" >> $LOG 2>&1
  A=$(acc_of "$F")
  echo "ACC $TAG forced = $A" >> $LOG
  if [ "$A" = "0.0" ] || [ "$A" = "0" ]; then
    echo "--- $TAG retry: 4 shots, 1536 tokens $*" >> $LOG
    $PY eval/mlx_stage2_eval.py --model "$M" --hf-tokenizer "$T" --tag "$TAG" --n 60 \
        --arms baseline --max-new 1536 --shots 4 "$@" \
        --out final_eval_baseline_strong.json >> $D/baseline_strong.log 2>&1 \
      || { echo "RETRY FAILED $TAG" >> $LOG; return; }
    $PY eval/assess_baseline.py "$D/final_eval_baseline_strong.json" >> $LOG 2>&1
    echo "ACC $TAG strengthened = $(acc_of $D/final_eval_baseline_strong.json)" >> $LOG
  fi
}

run results/stage2_gemma12b mlx-community/gemma-4-12B-it-4bit mlx-community/gemma-4-12B-it-4bit _gemma12b
run results/stage2_mlx8b9 mlx-community/Qwen3-8B-4bit Qwen/Qwen3-8B _mlx8b9 --thinking
echo "FOLLOWUP COMPLETE $(date +%H:%M)" >> $LOG
