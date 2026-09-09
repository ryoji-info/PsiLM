#!/bin/bash
# Leaky-gate sweep on Gemma 4 12B: does the Qwen result replicate on a backbone
# whose gate operates an order of magnitude lower?
#
# Why these epsilons. The dose that matters is the injection as a share of the
# residual stream, and the two backbones sit in very different places: Qwen's
# trained gate is 0.79 with the injection at ~15% of the stream, Gemma's is
# 0.15-0.19 at 3-4%. So eps = 0.2 is a QUARTER of Qwen's normal on-task dose but
# slightly ABOVE Gemma's -- applied on every prompt. Floors here bracket Gemma's
# own operating point (0.02, 0.05, 0.1, 0.2) and then jump to the fully open
# gate (1.0), which is the arm that collapsed Qwen from 89% to 8% on GSM8K.
#
# A two-question smoke already suggests Gemma is the more fragile of the two:
# eps = 0.2 broke GSM8K and eps = 1.0 broke every dataset including physics.
# If that holds at n=100, the safe range is a property of the backbone's own
# operating point, not a universal constant -- which is the useful finding.
#
# --no-parity-check: the benchmark's staged decoder differs from the stock Gemma
# by 1.2e-2 relative on the last-position logits (argmax identical), above the
# 1e-2 gate. The prior Gemma guard-rail ran the same way
# (results/bench/gemma12b_guardrail_summary.json, no_parity_check true). Greedy
# output is unaffected, and the KL compares two runs through the same staged
# decoder, so the offset cancels.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
G=mlx-community/gemma-4-12B-it-4bit
LOG=results/bench/leaky_sweep.log
CACHE=results/bench/tasks_gemma_n100.json
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "LEAKY-GEMMA START $(date +%H:%M)" >> $LOG

COMMON="--tasks-cache $CACHE --model $G --hf-tokenizer $G \
  --ckpt results/stage2_gemma12b/bridges.npz --fno results/stage2/fno.pt --gate-bias 0.0 \
  --datasets physics,mmlu,gsm8k,boolq \
  --arms base,psilm,leaky0.02,leaky0.05,leaky0.1,leaky0.2,leaky1.0 \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 --gsm8k-nudge 1 \
  --kl --seed 0 --no-parity-check --print-every 5 --save-every 5"

$PY eval/bench_guardrail.py --tag gemma_cachebuild --n 100 $COMMON --build-cache \
    >> results/bench/gemma_cachebuild.log 2>&1 \
  || { echo "LEAKY-GEMMA cache build FAILED $(date +%H:%M)" >> $LOG; exit 1; }
echo "LEAKY-GEMMA cache built $(date +%H:%M)" >> $LOG

FLAG=--fresh
for i in $(seq 1 40); do
  $PY eval/bench_guardrail.py --tag leaky_gemma --n 100 $COMMON $FLAG \
      >> results/bench/leaky_gemma.log 2>&1
  RC=$?
  [ $RC -eq 0 ] && { echo "LEAKY-GEMMA COMPLETE (attempt $i) $(date +%H:%M)" >> $LOG; exit 0; }
  echo "LEAKY-GEMMA attempt $i exited $RC; resuming $(date +%H:%M)" >> $LOG
  FLAG=--resume
  sleep 20
done
echo "LEAKY-GEMMA GAVE UP $(date +%H:%M)" >> $LOG
exit 1
