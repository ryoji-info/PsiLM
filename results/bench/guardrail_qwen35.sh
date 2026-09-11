#!/bin/bash
# Guard-rail benchmark for the Qwen3.5 9B bridge: does attaching it damage the
# backbone, and is the gate selective?
#
# Three arms on the same weights and prompts: base (frozen LLM alone), psilm
# (bridges attached, gate free), zeroed (bridges attached, injection multiplied
# by zero -- so base-vs-zeroed is the numerical noise floor and psilm-vs-zeroed
# is the injection's actual effect). The leaky-gate sweep is a separate later
# question and is deliberately not run here; get the operating point first.
#
# No --no-parity-check, unlike the Gemma sweep. Gemma's staged decoder differs
# from stock by 1.2e-2 relative on the last-position logits and had to be waved
# through; this backbone's staged forward reproduces stock exactly (0.000e+00),
# so the check should pass on its own. If it does not, that is a real finding
# about the padding mask the linear-attention shim supplies and should be
# investigated, not suppressed.
#
# The task cache is per-backbone: it holds tokenized prompts, and this tokenizer
# shares no ids with Gemma's. Built by the first invocation below.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
LOG=results/bench/guardrail_qwen35.log
CACHE=results/bench/tasks_qwen35_n100.json
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
mkdir -p results/bench
echo "GUARDRAIL-QWEN35 START $(date +%H:%M)" >> $LOG

COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $M \
  --ckpt results/stage2_qwen35/bridges.npz --fno results/stage2/fno.pt \
  --l-fwd 13 --l-rev 26 --gate-bias 0.0 \
  --datasets physics,mmlu,gsm8k,boolq --arms base,psilm,zeroed \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 \
  --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"

if [ ! -s $CACHE ]; then
  $PY eval/bench_guardrail.py --tag qwen35_cachebuild --n 100 $COMMON --build-cache \
      >> results/bench/qwen35_cachebuild.log 2>&1 \
    || { echo "GUARDRAIL-QWEN35 cache build FAILED $(date +%H:%M)" >> $LOG; exit 1; }
  echo "GUARDRAIL-QWEN35 cache built $(date +%H:%M)" >> $LOG
fi

# the bench is resumable and long runs have been killed by memory pressure
# before, so retry from its own checkpoint rather than restarting the sweep
FLAG=--fresh
for i in $(seq 1 40); do
  $PY eval/bench_guardrail.py --tag guardrail_qwen35 --n 100 $COMMON $FLAG \
      >> results/bench/guardrail_qwen35_run.log 2>&1
  RC=$?
  [ $RC -eq 0 ] && { echo "GUARDRAIL-QWEN35 COMPLETE (attempt $i) $(date +%H:%M)" >> $LOG; exit 0; }
  echo "GUARDRAIL-QWEN35 attempt $i exited $RC; resuming $(date +%H:%M)" >> $LOG
  FLAG=--resume
  sleep 20
done
echo "GUARDRAIL-QWEN35 GAVE UP $(date +%H:%M)" >> $LOG
exit 1
