#!/bin/bash
# Second augmentation form: the distractor carries a small nonzero amplitude
# (U(0.02, 0.25)) with the answer recomputed by the solver, so the readout
# learns "read whatever is written in the second span" rather than "a second
# term contributes nothing". The held-out family's joint range is still unseen.
cd /Users/rxiii/Documents/GitHub/PsiLM
until grep -q "AUG COMPLETE" results/readout_transfer/aug_results.log 2>/dev/null; do sleep 60; done
for S in 0 1 2; do
  F=results/readout_transfer/span_l10_s${S}_aug25.json
  .venv/bin/python eval/readout_transfer_probe.py --steps 2500 --batch 8 --n-eval 96 \
      --readouts span --l-fwd 10 --seed $S --aug-zero-frac 0.5 --aug-amp-max 0.25 --out $F \
      >> results/readout_transfer/aug2.log 2>&1
  python3 - "$F" "$S" <<'PY' >> results/readout_transfer/aug_results.log
import json, sys
d = json.load(open(sys.argv[1]))["readouts"]["span"]
acc = " ".join(f"{k.replace('val_','')}={d[k]['acc']:.3f}" for k in ("val_iid","val_combo","val_amp"))
print(f"AUG25 seed={sys.argv[2]} {acc} combo_amp_mae={d['val_combo']['amp_mae']:.4f}")
PY
done
echo "AUG25 COMPLETE" >> results/readout_transfer/aug_results.log
