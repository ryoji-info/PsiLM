#!/bin/bash
# Constitution bridge, the 9B stage, part 1: everything that does not depend on
# how the 0.5B bridges turn out -- the value-neuron identification on the
# Qwen3.5 9B backbone and the 9B teacher data. Waits for the 0.5B chain
# (results/constitution/train_qwen0.5b.sh) to exit so the two never share the
# GPU: the Metal watchdog killed every 0.5B job that ran alongside another.
#
#   1. collect  800 GSM8K-TRAIN trajectories at the paper's sampler (temperature
#      1.0 / top-p 0.95): this backbone solves ~83% greedy, so sampling is what
#      balances the reward here (the 0.5B needed greedy for the opposite reason).
#      Layers 13 (the read depth) and 20..30 (candidate injection depths: the
#      physics campaign's memory probes put the cliff at 26 for batch 2, and the
#      constitution model adds ~1 GB on top). ~2 h, ~15 GB of float16 states.
#   2. probe    Monte-Carlo target, 30 epochs, candidates 24/26/28  (~1 h)
#   3. ablate   base / zeroV / zeroV5 / 3 random arms on GSM8K test n=100 (~1.5 h)
#   4. data     1,000 / 100 / 100 / 100 teacher-vs-base pairs at max-new 128 (~3.5 h)
# Training on the 9B is a separate script, launched after the 0.5B results are
# reviewed. Markers in results/qwen35/constitution_prep.log; ends with
# "QWEN35-PREP COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
TAG=qwen35
D=results/value_neurons/$TAG
LOG=results/qwen35/constitution_prep.log
LAYERS=13,20,22,24,26,28,30
CAND=24,26,28
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
mkdir -p $D results/constitution
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "QWEN35-PREP WAITING for the 0.5B chains to exit"
until grep -q "FOLLOWUP3 COMPLETE" results/constitution/train_qwen0.5b_followup3.log 2>/dev/null; do sleep 120; done
# Match only a bash-executed chain. A bare path pattern also matches this
# session's own `tail -f <that log>` monitors, which deadlocked this wait for
# an hour on 2026-09-12 after every 0.5B job had already finished.
while pgrep -f 'bash results/constitution/train_qwen0.5b|bash results/value_neurons/run_qwen0.5b' > /dev/null; do sleep 60; done
step "QWEN35-PREP START"

# 0. the 0.5B ablation's random controls, redrawn from a pool without the
#    attention-sink dims: one of the first three draws contained dim 62 (|h| 1,550
#    at position 0, 0.5 elsewhere) and scored 0% with no item ever stopping --
#    a measurement of the sink, not of the value neurons. Same base/zeroV rows are
#    regenerated into a separate file so the original record stays untouched. ~15 min.
step "VN05 SINKFREE CONTROLS START"
for i in 1 2 3; do
  $PY eval/vn_ablate.py --tag qwen0.5b --neurons results/value_neurons/qwen0.5b/layer16.json \
    --n 100 --seed 0 --max-new 384 --random-seeds 3 --random-seed-base 30000 \
    --random-arm-prefix rands --random-exclude-sink --out-suffix _sinkfree \
    --arms base,zeroV,rands0,rands1,rands2 >> results/value_neurons/qwen0.5b/ablate_sinkfree.log 2>&1 && break
  step "VN05 SINKFREE attempt $i exited $?; resuming"; sleep 30
done
sed -n '/^layer /,$p' results/value_neurons/qwen0.5b/ablate_sinkfree.log | tail -10 >> $LOG
step "VN05 SINKFREE CONTROLS DONE"

step "VN35 COLLECT START"
for i in 1 2 3 4 5 6; do
  $PY eval/vn_collect.py --tag $TAG --model $M --hf-tokenizer $M --n 800 --seed 0 \
    --max-new 320 --layers $LAYERS --temp 1.0 --top-p 0.95 >> $D/collect.log 2>&1 && break
  step "VN35 COLLECT attempt $i exited $?; resuming"; sleep 30
  [ $i = 6 ] && { step "VN35 COLLECT FAILED"; exit 1; }
done
step "VN35 COLLECT DONE $(grep 'VN-COLLECT DONE\|reward_rate' $D/collect.log | tail -1 | cut -c1-160)"

step "VN35 PROBE START"
for i in 1 2 3 4 5; do
  $PY eval/vn_probe.py --tag $TAG --layers $LAYERS --candidate-rev $CAND --epochs 30 \
    --target mc --resume >> $D/probe.log 2>&1 && break
  step "VN35 PROBE attempt $i exited $?; resuming"; sleep 30
  [ $i = 5 ] && { step "VN35 PROBE FAILED"; exit 1; }
done
sed -n '/ layer  auc_full/,$p' $D/probe.log | tail -12 >> $LOG
step "VN35 PROBE DONE"

L=$($PY -c "import json;print(json.load(open('$D/summary.json'))['chosen_layer'])")
if [ "$L" = "None" ]; then step "VN35 NO CHOSEN LAYER -- stopping"; exit 1; fi
step "VN35 ABLATE START layer=$L"
for i in 1 2 3 4 5; do
  $PY eval/vn_ablate.py --tag $TAG --model $M --hf-tokenizer $M --neurons $D/layer$L.json \
    --n 100 --seed 0 --max-new 384 --random-seeds 3 --also-top5pct --random-exclude-sink \
    >> $D/ablate.log 2>&1 && break
  step "VN35 ABLATE attempt $i exited $?; resuming"; sleep 30
  [ $i = 5 ] && { step "VN35 ABLATE FAILED"; exit 1; }
done
sed -n '/^layer /,$p' $D/ablate.log | tail -12 >> $LOG
step "VN35 ABLATE DONE"

step "DATA35 START"
ARGS="--tag $TAG --n-train 1000 --n-val 100 --n-test 100 --n-helpful-test 100 \
  --seed 7 --model $M --hf-tokenizer $M --max-new 128 \
  --excerpt data/constitution/system_excerpt.md --out-dir data --work-dir results/constitution --print-every 25"
for i in $(seq 1 30); do
  $PY eval/build_constitution_data.py $ARGS >> results/constitution/build_data_${TAG}_run.log 2>&1 && break
  step "DATA35 attempt $i exited $?; resuming"; sleep 30
  [ $i = 30 ] && { step "DATA35 GAVE UP"; exit 1; }
done
$PY -c "
import json
s = json.load(open('data/constitution_${TAG}_stats.json'))
for k, v in s['splits'].items():
    print('%-13s n=%-5d differs=%.3f refusal base=%.3f teacher=%.3f len base=%.1f teacher=%.1f'
          % (k, v['n'], v['teacher_differs_rate'], v['refusal_rate_base'],
             v['refusal_rate_teacher'], v['mean_len_base'], v['mean_len_teacher']))" >> $LOG 2>&1
step "DATA35 DONE"
step "QWEN35-PREP COMPLETE"
