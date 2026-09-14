#!/bin/bash
# Magnitude-matched ablation controls for the Qwen3.5 9B value neurons.
#
# The prep chain's own ablation uses uniform random draws, which the 0.5B stage
# proved too weak: at layer 24 the 41 value neurons sum to 90.7 of activation RMS
# and three uniform draws sum to 40-45, so a uniform control is a half-sized
# perturbation and will flatter the value neurons whatever they encode. These
# three draws match each value neuron's activation RMS one to one, from outside
# the top 5% (the layer has no attention-sink dimension to exclude): summed RMS
# 87.0-87.8, zero overlap with the top 5%.
#
# Runs after the whole prep chain so nothing shares the GPU, and the 9B training
# chain waits for this script's marker in turn. Ends with "VN35-MAGMATCH COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
VN=results/value_neurons/qwen35
LOG=results/qwen35/vn35_magmatch.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "VN35-MAGMATCH WAITING for the prep chain"
until grep -q "QWEN35-PREP COMPLETE" results/qwen35/constitution_prep.log 2>/dev/null; do
  grep -qE "FAILED|GAVE UP|NO CHOSEN" results/qwen35/constitution_prep.log 2>/dev/null \
    && { step "VN35-MAGMATCH: prep chain failed; running the ablation anyway"; break; }
  sleep 120
done
while pgrep -f 'bash results/qwen35/constitution_prep' > /dev/null; do sleep 60; done
L=$($PY -c "import json;print(json.load(open('$VN/summary.json'))['chosen_layer'])")
step "VN35-MAGMATCH START layer=$L"

for s in 0 1 2; do
  ARMS=zeroV; [ $s = 0 ] && ARMS=base,zeroV
  for i in 1 2 3; do
    $PY eval/vn_ablate.py --tag qwen35 --model $M --hf-tokenizer $M \
      --neurons $VN/magmatch${s}_layer$L.json --n 100 --seed 0 --max-new 384 \
      --random-seeds 0 --arms $ARMS --out-suffix _magmatch$s >> $VN/ablate_magmatch.log 2>&1 && break
    step "VN35-MAGMATCH draw $s attempt $i exited $?; resuming"; sleep 30
    [ $i = 3 ] && { step "VN35-MAGMATCH draw $s FAILED"; break; }
  done
  step "VN35-MAGMATCH draw $s: $(grep -E '^(base|zeroV) ' $VN/ablate_magmatch.log | tail -2 | tr -s ' ' | tr '\n' '|')"
done
step "VN35-MAGMATCH COMPLETE"
