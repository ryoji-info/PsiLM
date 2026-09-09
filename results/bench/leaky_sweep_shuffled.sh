#!/bin/bash
# Content control for the leaky-gate sweep: same channel, same floor, same
# magnitude distribution, WRONG VALUE.
#
# The shuffled<eps> arms feed the value encoder the u_hat of a different
# question from the same dataset (a one-step rotation of the sorted qids: 400
# substitutions, zero fixed points, and the multiset of injected values is
# unchanged by construction). Everything else is identical -- the readout still
# runs on the real prompt, the gate still decides, only the number injected is
# another question's.
#
# The question. The floor shortens replies and lifts MMLU where the token budget
# binds. Is that the physics model's OUTPUT doing it, or is it any perturbation
# of that magnitude arriving through the channel?
#
# Registered prediction, before the run: the effects SURVIVE the shuffle -- the
# brevity is the perturbation, not the content. Evidence for it: off-task the
# injected value is a Burgers field value computed from parameters scraped out
# of a word problem (GSM8K mean -0.218, sd 0.095, semantically empty), and the
# correlation between |injected value| and tokens saved by the floor is only
# +0.09 on GSM8K and +0.18 on MMLU. If instead the shuffled arms revert toward
# the trained gate, the content matters and the physics framing survives.
#
# Two floors: 0.2, where Qwen showed the MMLU gain at no GSM8K cost, and 1.0,
# where it collapsed 89% -> 8%. Each has a matched leaky arm already measured.
#
# OUTCOME: the prediction held off-task and inverted on-task -- a double
# dissociation. Off-task the swap changes nothing (MMLU 0.66 both ways, GSM8K
# 0.88 both ways, every paired test p = 1.00, KL indistinguishable). On-task it
# is annihilating: physics 98% -> 9% at identical tokens and parse rate, with
# the spoken answer within tolerance of the INJECTED value on 99 of 100
# questions. Presence explains the off-task effects; content explains the
# on-task ones.
# NOTE for a fresh clone: the wait below keys on a log this repository does not
# ship (results/**/*.log is git-ignored), so it blocks indefinitely if the run it
# waits for has not happened here. Delete the wait, or touch the marker, when
# re-running these sweeps from scratch.
# The task cache this consumes is built by the first sweep and is git-ignored;
# build it before running this script standalone:
#   python eval/bench_guardrail.py --tag cachebuild --n 100 --tasks-cache $CACHE \
#       --build-cache <the same --model/--datasets/--max-new-* flags as below>
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
LOG=results/bench/leaky_sweep.log
FIRST=results/bench/leaky_8b_guardrail.rows.jsonl
export HF_HUB_DISABLE_XET=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface

# behind the Gemma sweep and the cache relocation: one GPU, and the move
# invalidates the old cache path for any process started after it
until grep -q "=== done ===" results/bench/relocate_hf.log 2>/dev/null; do sleep 120; done
sleep 30
echo "LEAKY-SHUF START $(date +%H:%M)" >> $LOG

CACHE=results/bench/tasks_leaky_n100.json
COMMON="--tasks-cache $CACHE --model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B \
  --ckpt results/stage2_mlx8b9/bridges.npz --fno results/stage2/fno.pt --gate-bias -2.0 \
  --datasets physics,mmlu,gsm8k,boolq --arms shuffled0.2,shuffled1.0 \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 --gsm8k-nudge 1 \
  --kl --base-gen-from $FIRST --shuffle-values-from $FIRST --seed 0 --print-every 5 --save-every 5"

FLAG=--fresh
for i in $(seq 1 40); do
  $PY eval/bench_guardrail.py --tag leaky_8b_shuf --n 100 $COMMON $FLAG \
      >> results/bench/leaky_8b_shuf.log 2>&1
  RC=$?
  [ $RC -eq 0 ] && { echo "LEAKY-SHUF COMPLETE (attempt $i) $(date +%H:%M)" >> $LOG; exit 0; }
  echo "LEAKY-SHUF attempt $i exited $RC; resuming $(date +%H:%M)" >> $LOG
  FLAG=--resume
  sleep 20
done
echo "LEAKY-SHUF GAVE UP $(date +%H:%M)" >> $LOG
exit 1
