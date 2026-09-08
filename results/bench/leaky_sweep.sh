#!/bin/bash
# Leaky-gate dose-response on the released 8B selective bridges, inference only:
# the gate floored at eps in {0.01, 0.05, 0.1, 0.2} against the base backbone and
# the trained selective gate, on physics + GSM8K + MMLU + BoolQ, n=100 each,
# with per-token KL(base || arm). eps=0.2 should reproduce the v8 collapse
# (gate open everywhere at the 20% cap: GSM8K 89 -> 34). Waits for the chain
# and the baseline follow-up: one GPU.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
LOG=results/bench/leaky_sweep.log
export HF_HUB_DISABLE_XET=1
until grep -q "FOLLOWUP COMPLETE" results/gemma12b/baseline_followup.log 2>/dev/null; do sleep 120; done
echo "LEAKY START $(date +%H:%M)" >> $LOG
COMMON="--model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B \
  --ckpt results/stage2_mlx8b9/bridges.npz --fno results/stage2/fno.pt --gate-bias -2.0 \
  --datasets physics,mmlu,gsm8k,boolq --arms base,psilm,leaky0.01,leaky0.05,leaky0.1,leaky0.2 \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 --gsm8k-nudge 1 \
  --kl --seed 0 --print-every 5 --save-every 5"
# fail fast: two questions per dataset through every arm
$PY eval/bench_guardrail.py --tag leaky_8b_smoke --n 2 $COMMON --fresh > results/bench/leaky_8b_smoke.log 2>&1 \
  || { echo "LEAKY SMOKE FAILED $(date +%H:%M)" >> $LOG; exit 1; }
echo "LEAKY SMOKE OK $(date +%H:%M)" >> $LOG
$PY eval/bench_guardrail.py --tag leaky_8b --n 100 $COMMON --fresh > results/bench/leaky_8b.log 2>&1 \
  || { echo "LEAKY SWEEP FAILED $(date +%H:%M)" >> $LOG; exit 1; }
echo "LEAKY SWEEP COMPLETE $(date +%H:%M)" >> $LOG
grep -A 40 "^dataset" results/bench/leaky_8b.log | head -60 >> $LOG
