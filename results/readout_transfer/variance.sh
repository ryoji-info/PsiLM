#!/bin/bash
# The layer sweep came out non-monotonic (5: 0.979, 8: 0.302, 10: 0.740 on the
# combination family), which no context-mixing account explains. Before any of
# it sets the 12B run's depth, measure the run-to-run spread: two more seeds at
# each of the three layers, everything else identical.
cd /Users/rxiii/Documents/GitHub/PsiLM
until grep -q "SWEEP COMPLETE" results/readout_transfer/sweep_results.log 2>/dev/null; do sleep 60; done
for S in 1 2; do
  for L in 5 8 10; do
    F=results/readout_transfer/span_l${L}_s${S}.json
    .venv/bin/python eval/readout_transfer_probe.py --steps 2500 --batch 8 --n-eval 96 \
        --readouts span --l-fwd $L --seed $S --out $F >> results/readout_transfer/variance.log 2>&1
    echo "VAR l_fwd=$L seed=$S $(python3 -c "
import json;d=json.load(open('$F'))['readouts']['span']
print(' '.join(f\"{k.replace('val_','')}={d[k]['acc']:.3f}\" for k in ('val_iid','val_combo','val_amp')))")" \
        >> results/readout_transfer/variance_results.log
  done
done
echo "VARIANCE COMPLETE" >> results/readout_transfer/variance_results.log
