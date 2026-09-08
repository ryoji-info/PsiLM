#!/bin/bash
# Leaky-gate dose-response on the released 8B selective bridges, inference only:
# the gate floored at eps in {0.01, 0.05, 0.1, 0.2} against the base backbone and
# the trained selective gate, on physics + GSM8K + MMLU + BoolQ, n=100 each,
# with per-token KL(base || arm). eps=0.2 should reproduce the v8 collapse
# (gate open everywhere at the 20% cap: GSM8K 89 -> 34).
#
# Tasks are built once into a cache by a separate process, so the run process
# does no dataset work. Retry-with-resume is kept as a safety net: the benchmark
# appends every row as it lands and --resume skips what is already done.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
LOG=results/bench/leaky_sweep.log
export HF_HUB_DISABLE_XET=1
CACHE=results/bench/tasks_leaky_n100.json
COMMON="--tasks-cache $CACHE --model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B \
  --ckpt results/stage2_mlx8b9/bridges.npz --fno results/stage2/fno.pt --gate-bias -2.0 \
  --datasets physics,mmlu,gsm8k,boolq --arms base,psilm,leaky0.01,leaky0.05,leaky0.1,leaky0.2 \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 --gsm8k-nudge 1 \
  --kl --seed 0 --print-every 5 --save-every 5"

attempt() {   # tag  n  max_attempts  first_flag
  local TAG=$1 N=$2 MAX=$3 FLAG=$4 i=1
  while [ $i -le $MAX ]; do
    $PY eval/bench_guardrail.py --tag $TAG --n $N $COMMON $FLAG \
        >> results/bench/${TAG}.log 2>&1
    local RC=$?
    if [ $RC -eq 0 ]; then
      echo "LEAKY $TAG COMPLETE (attempt $i) $(date +%H:%M)" >> $LOG
      return 0
    fi
    echo "LEAKY $TAG attempt $i exited $RC$([ $RC -eq 137 ] && echo ' (SIGKILL)'); resuming $(date +%H:%M)" >> $LOG
    FLAG="--resume"          # every attempt after the first continues the rows file
    i=$((i + 1))
    sleep 20
  done
  echo "LEAKY $TAG GAVE UP after $MAX attempts $(date +%H:%M)" >> $LOG
  return 1
}

echo "LEAKY START $(date +%H:%M)" >> $LOG
$PY eval/bench_guardrail.py --tag cachebuild --n 100 $COMMON --build-cache \
    >> results/bench/cachebuild.log 2>&1 \
  || { echo "LEAKY cache build FAILED $(date +%H:%M)" >> $LOG; exit 1; }
echo "LEAKY cache built $(date +%H:%M)" >> $LOG
attempt leaky_8b 100 40 --fresh || exit 1
echo "LEAKY SWEEP COMPLETE $(date +%H:%M)" >> $LOG
grep -A 40 "^dataset" results/bench/leaky_8b.log | tail -45 >> $LOG
