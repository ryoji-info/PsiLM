#!/bin/bash
# Leaky-gate sweep, upper range: eps in {0.3, 0.4, 0.5}.
#
# Why a second sweep. The floor multiplies an injection that inj_cap has already
# limited to 0.2 of the stream RMS, so eps = 0.2 puts roughly 4% of the residual
# stream through the channel -- a fifth of run 8's operating point, where the
# gate was open at the cap and GSM8K fell 89 -> 34. The first sweep therefore
# showed the channel is harmless in its range without finding where that stops.
# These three arms walk toward it.
#
# Comparability: same task cache (identical questions and prompts), same seed,
# greedy decoding. The base and selective arms are NOT rerun -- their rows are
# already recorded and deterministic; --base-gen-from reuses the base arm's own
# continuations so the KL reference is identical rather than merely comparable.
# eval/leaky_report.py --merge stitches the two runs into one dose-response table.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
LOG=results/bench/leaky_sweep.log
FIRST=results/bench/leaky_8b_guardrail.rows.jsonl
export HF_HUB_DISABLE_XET=1
until grep -q "LEAKY SWEEP COMPLETE" $LOG 2>/dev/null; do sleep 120; done
echo "LEAKY-HI START $(date +%H:%M)" >> $LOG

CACHE=results/bench/tasks_leaky_n100.json
COMMON="--tasks-cache $CACHE --model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B \
  --ckpt results/stage2_mlx8b9/bridges.npz --fno results/stage2/fno.pt --gate-bias -2.0 \
  --datasets physics,mmlu,gsm8k,boolq --arms leaky0.3,leaky0.4,leaky0.5 \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 --gsm8k-nudge 1 \
  --kl --base-gen-from $FIRST --seed 0 --print-every 5 --save-every 5"

FLAG=--fresh
for i in $(seq 1 40); do
  $PY eval/bench_guardrail.py --tag leaky_8b_hi --n 100 $COMMON $FLAG \
      >> results/bench/leaky_8b_hi.log 2>&1
  RC=$?
  [ $RC -eq 0 ] && { echo "LEAKY-HI COMPLETE (attempt $i) $(date +%H:%M)" >> $LOG; exit 0; }
  echo "LEAKY-HI attempt $i exited $RC; resuming $(date +%H:%M)" >> $LOG
  FLAG=--resume
  sleep 20
done
echo "LEAKY-HI GAVE UP $(date +%H:%M)" >> $LOG
exit 1
