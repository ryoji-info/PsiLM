#!/bin/bash
# Second follow-up: guard-rails for the two write-location variants the main
# chain evaluated but did not guard-rail (top-45 value neurons, random-9) and
# for the plain-partner control, so every variant has the same four-dataset
# record (GSM8K/MMLU/BoolQ item agreement, red-team KL and refusal). Runs after
# results/constitution/train_qwen0.5b_followup.sh; ends with "FOLLOWUP2 COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
T=Qwen/Qwen2.5-0.5B-Instruct
CM=results/constitution_model/qwen2.5-0.5b-constitution
DATA_TE=data/constitution_test_qwen0.5b.json
LOG=results/constitution/train_qwen0.5b_followup2.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

guardrail() {       # $1 variant name, $2 constitution model path
  local NAME=$1 CMP=$2 D=results/stage2c_qwen0.5b_$1 TAG=const_qwen0.5b_$1 FLAG=--fresh
  local CACHE=results/bench/tasks_const_qwen0.5b_n100.json
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $T --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CMP --redteam-data $DATA_TE \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
    --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
    --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
  [ -s $D/bridges.npz ] || { step "GUARDRAIL $NAME: no checkpoint, skipped"; return 0; }
  for i in $(seq 1 10); do
    $PY eval/bench_guardrail.py --tag $TAG --n 100 $COMMON $FLAG >> results/bench/${TAG}_run.log 2>&1 \
      && { step "GUARDRAIL $NAME COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${TAG}_run.log | tail -8 >> $LOG; return 0; }
    step "GUARDRAIL $NAME attempt $i exited; resuming"; FLAG=--resume; sleep 20
  done
  step "GUARDRAIL $NAME GAVE UP"; return 1
}

step "FOLLOWUP2 WAITING for the first follow-up"
until grep -q "FOLLOWUP COMPLETE\|FOLLOWUP FAILED" results/constitution/train_qwen0.5b_followup.log 2>/dev/null; do sleep 60; done
while pgrep -f 'results/constitution/train_qwen0.5b.sh|train_qwen0.5b_followup.sh' > /dev/null; do sleep 30; done
step "FOLLOWUP2 START"
guardrail vn5 $CM
guardrail rand $CM
guardrail plainpartner Qwen/Qwen2.5-0.5B-Instruct
step "FOLLOWUP2 COMPLETE"
