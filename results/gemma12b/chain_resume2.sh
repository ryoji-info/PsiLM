#!/bin/bash
# The Qwen forced baseline failed on load: the v9 bridges predate the readout
# calibration buffers the module now always defines. load_bridge_weights
# tolerates exactly that omission. Rerun it and close the chain.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
LOG=results/gemma12b/chain.log
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1
stage() { echo "STAGE DONE $1 $(date +%H:%M)" >> $LOG; }
echo "RESUME qwen-baseline $(date +%H:%M)" >> $LOG
$PY eval/mlx_stage2_eval.py --model mlx-community/Qwen3-8B-4bit --hf-tokenizer Qwen/Qwen3-8B \
    --tag _mlx8b9 --n 60 --arms baseline --max-new 768 --out final_eval_baseline_forced.json \
    > results/stage2_mlx8b9/baseline_forced.log 2>&1 \
  || { echo "STAGE FAILED qwen baseline (resume) $(date +%H:%M)" >> $LOG; exit 1; }
stage "qwen-baseline-forced"
echo "CHAIN COMPLETE $(date +%H:%M)" >> $LOG
