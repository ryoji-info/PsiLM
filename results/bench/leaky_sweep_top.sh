#!/bin/bash
# Leaky-gate sweep, top of the range: eps in {0.7, 0.9, 1.0}.
#
# eps = 1.0 makes sigma_eff identically 1.0, which is run 8's operating point:
# gate fully open, injection at the 0.2 cap on every prompt. Run 8 lost 55
# points on GSM8K there (89 -> 34), but its gate was TRAINED open, so the
# collapse could have been the gate or the training. This arm separates them --
# same released run-9 bridges, gate forced open only at inference.
#
# Registered prediction, before the run: GSM8K collapses toward run 8's 34% at
# eps = 1.0. The eps = 0.5 arm already lost 8 points (89 -> 81, McNemar p =
# 0.039 vs the trained gate), so the curve is already bending.
#
# Same task cache, same seed, greedy; base and psilm arms are not rerun and
# --base-gen-from reuses the first sweep's base continuations as the KL
# reference. eval/leaky_report.py --merge stitches all three sweeps.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
LOG=results/bench/leaky_sweep.log
FIRST=results/bench/leaky_8b_guardrail.rows.jsonl
export HF_HUB_DISABLE_XET=1
echo "LEAKY-TOP START $(date +%H:%M)" >> $LOG

CACHE=results/bench/tasks_leaky_n100.json
COMMON="--tasks-cache $CACHE --model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B \
  --ckpt results/stage2_mlx8b9/bridges.npz --fno results/stage2/fno.pt --gate-bias -2.0 \
  --datasets physics,mmlu,gsm8k,boolq --arms leaky0.7,leaky0.9,leaky1.0 \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 --gsm8k-nudge 1 \
  --kl --base-gen-from $FIRST --seed 0 --print-every 5 --save-every 5"

FLAG=--fresh
for i in $(seq 1 40); do
  $PY eval/bench_guardrail.py --tag leaky_8b_top --n 100 $COMMON $FLAG \
      >> results/bench/leaky_8b_top.log 2>&1
  RC=$?
  [ $RC -eq 0 ] && { echo "LEAKY-TOP COMPLETE (attempt $i) $(date +%H:%M)" >> $LOG; exit 0; }
  echo "LEAKY-TOP attempt $i exited $RC; resuming $(date +%H:%M)" >> $LOG
  FLAG=--resume
  sleep 20
done
echo "LEAKY-TOP GAVE UP $(date +%H:%M)" >> $LOG
exit 1
