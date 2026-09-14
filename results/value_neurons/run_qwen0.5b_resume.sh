#!/bin/bash
# The full value-neuron identification on the 0.5B dev backbone, end to end.
#
#   0. self-tests: the planted-signal recovery, in both the unscaled and the
#      heterogeneous-scale variant (the second is what justifies the default
#      --standardize zscore; it fails loudly with --standardize none)
#   1. collect  1000 GSM8K-TRAIN trajectories, 10 layer boundaries, max-new 320,
#      one GREEDY continuation per problem (~30 min, ~4.5 GB of float16 states
#      under traj/). The paper samples at temperature 1.0 / top-p 0.95, but on
#      this 0.5B that policy solves 6% of problems against 42% greedy (measured
#      in the smoke runs): a 6% positive rate leaves ~12 positive held-out
#      trajectories, too few for the AUC that picks the layer. Greedy is also
#      the policy every evaluation in this repo decodes with, so the value the
#      probe learns is the value under the deployed policy. Per-backbone choice:
#      a stronger backbone (the 9B) can afford the paper's sampler.
#   2. probe    the value probe per layer (Monte-Carlo target, see below),
#      ranking, pruning sweep, random control at 0.99, chosen injection depth
#      among 14..20 (~1.5 h at 30 epochs)
#   3. ablate   the causal check on GSM8K TEST at the chosen layer: base /
#      zeroV(top1%) / zeroV5(top5%) / 3 random-dims controls  (~15 min)
#
# Sequential on purpose: steps 1 and 3 each hold their own copy of the backbone
# and the machine has 24 GB (and other agents' jobs on the same GPU).
#
# Nohup-able:  nohup bash results/value_neurons/run_qwen0.5b.sh &
# Markers land in results/value_neurons/run_qwen0.5b.log; the chain ends with
# "VN DONE".
#
# The paper-faithful sampled variant (temperature 1.0 / top-p 0.95) can be
# collected into a second tag by dropping --greedy and using --tag
# qwen0.5b_sampled in step 1 and passing that tag to steps 2 and 3; do NOT mix
# the two in one tag.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
TAG=qwen0.5b
D=results/value_neurons/$TAG
LOG=results/value_neurons/run_qwen0.5b.log
LAYERS=4,6,8,10,12,14,15,16,18,20
CAND=14,15,16,18,20
export HF_HUB_DISABLE_XET=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
mkdir -p $D
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

# resume entry point: the collect is done, run the probe and the ablation only
if pgrep -f 'eval/vn_(probe|ablate).py' > /dev/null; then echo 'a vn_* job is already running; refusing to start' >> $LOG; exit 1; fi
step "VN PROBE START (resume)"
# --target mc: the paper's TD loss at 30 epochs left V nearly constant on this
# backbone (held-out AUC 0.42-0.60; the run is archived in $D/td30/), the
# Monte-Carlo target -- its gamma->1 fixed point -- reaches 0.67-0.69. --resume
# and the retry loop: the Metal watchdog kills long command buffers when other
# jobs share the GPU ("Impacting Interactivity"), and a layer takes ~8 min.
for i in 1 2 3 4 5; do
  $PY eval/vn_probe.py --tag $TAG --layers $LAYERS --candidate-rev $CAND --epochs 30 \
    --target mc --resume >> $D/probe.log 2>&1 && break
  step "PROBE attempt $i exited $?; resuming"; sleep 30
  [ $i = 5 ] && { step "PROBE FAILED"; exit 1; }
done
sed -n '/ layer  auc_full/,$p' $D/probe.log >> $LOG
step "VN PROBE DONE"

L=$($PY -c "import json;print(json.load(open('$D/summary.json'))['chosen_layer'])")
if [ "$L" = "None" ]; then step "NO CHOSEN LAYER (all AUC none) -- stopping"; exit 1; fi
step "VN ABLATE START layer=$L"
$PY eval/vn_ablate.py --tag $TAG --neurons $D/layer$L.json --n 100 --seed 0 \
  --max-new 384 --random-seeds 3 --also-top5pct \
  > $D/ablate.log 2>&1 || { step "ABLATE FAILED"; exit 1; }
sed -n '/^layer /,$p' $D/ablate.log >> $LOG
step "VN ABLATE DONE"
step "VN DONE"

# Optional, ~30 min: the paper trained the probe for 100 epochs. Re-probing only
# the chosen layer at 100 epochs sharpens the index list the bridge will use
# without paying 100 epochs on all ten layers (~5 h). It overwrites layer$L.json
# and summary.json, so keep the ten-layer table first; compare auc_by_ratio
# between the two before adopting the refined list, and re-run the ablation on it.
#   cp $D/summary.json $D/summary_30ep.json
#   $PY eval/vn_probe.py --tag $TAG --layers $L --candidate-rev $L --epochs 100 \
#     > $D/probe_100ep.log 2>&1
#   $PY eval/vn_ablate.py --tag $TAG --neurons $D/layer$L.json --n 100 --seed 0 \
#     --max-new 384 --random-seeds 3 --also-top5pct > $D/ablate_100ep.log 2>&1
