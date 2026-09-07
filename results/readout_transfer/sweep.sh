#!/bin/bash
# Where should the span readout read? Later layers have mixed more context
# between the two terms of a combination prompt, which is the only family the
# shared-head readout still misses.
cd /Users/rxiii/Documents/GitHub/PsiLM
for L in 5 8 14; do
  .venv/bin/python eval/readout_transfer_probe.py --steps 2500 --batch 8 --n-eval 96 \
      --readouts span --l-fwd $L --out results/readout_transfer/span_l$L.json \
      >> results/readout_transfer/sweep.log 2>&1
  echo "SWEEP l_fwd=$L $(python3 -c "
import json;d=json.load(open('results/readout_transfer/span_l$L.json'))['readouts']['span']
print({k: d[k]['acc'] for k in ('val_iid','val_combo','val_amp')}, 'amp_mae_combo', d['val_combo']['amp_mae'])")" \
      >> results/readout_transfer/sweep_results.log
done
echo "SWEEP COMPLETE" >> results/readout_transfer/sweep_results.log
