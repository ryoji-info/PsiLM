#!/bin/bash
# Re-evaluate the three magnitude-matched draws' 50-item test split at step 1000.
# Their eval_test.json files were produced at step 500, before chunk 2; the
# step-1000 identity spread pooled them under a step-1000 heading until
# vn_identity_spread.py learned to check the eval's own recorded step.
# Runs beside rt400 (eval peak 9 GB, rt400 ~7.5 GB). Step-500 evals are kept
# as eval_test_step500.*. LAUNCH A COPY, NOT THIS FILE. Ends with "RETEST COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
LOG=results/qwen35/retest_matches.log
step(){ echo "$1 $(date '+%F %H:%M')" | tee -a $LOG; }
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
DATA_TE=data/constitution_test_qwen35.json
evstep(){ $PY -c "import json;print(json.load(open('$1')).get('checkpoint',{}).get('step'))" 2>/dev/null; }
step "RETEST START (match0/1/2 test split at step 1000, beside rt400)"
for S in 0 1 2; do
  D=results/stage2c_qwen35_match$S
  STEP=$($PY -c "import json;print(json.load(open('$D/bridges.npz.meta')).get('step'))")
  [ "$STEP" = "1000" ] || { step "RETEST match$S meta step $STEP, not 1000; skipping"; continue; }
  [ "$(evstep $D/eval_test.json)" = "1000" ] && { step "RETEST match$S already evaluated at 1000"; continue; }
  for f in json jsonl log; do
    [ -e $D/eval_test.$f ] && [ ! -e $D/eval_test_step500.$f ] && mv $D/eval_test.$f $D/eval_test_step500.$f
  done
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
      --const-model $CM --data $DATA_TE --arms base,psilm,zeroed --n 50 --max-new 24 \
      --out $D/eval_test.json > $D/eval_test.log 2>&1
  step "RETEST match$S (eval step $(evstep $D/eval_test.json)): $(grep -E '^ +(base|psilm) ' $D/eval_test.log | tr -s ' ' | tr '\n' '|')"
done
$PY eval/vn_identity_spread.py --step 1000 --out results/constitution/identity_spread_qwen35.json >> $LOG 2>&1
step "RETEST COMPLETE"
