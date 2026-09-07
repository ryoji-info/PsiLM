#!/bin/bash
# Depth is settled as noise: layer 5 gave combination 0.979 on one seed and
# 0.396 on the next. The remaining candidate is context -- the readout never
# trains on a two-term prompt, so what it does with one is unconstrained.
# --aug-zero-frac writes the absent mode in at amplitude 0.0: identical physics
# and identical answers, two-term prompts, and the held-out family's joint
# amplitudes still unseen. Three seeds each, with and without, at one layer.
cd /Users/rxiii/Documents/GitHub/PsiLM
run() {  # seed  frac  tag
  F=results/readout_transfer/span_l10_s$1_aug$3.json
  .venv/bin/python eval/readout_transfer_probe.py --steps 2500 --batch 8 --n-eval 96 \
      --readouts span --l-fwd 10 --seed $1 --aug-zero-frac $2 --out $F \
      >> results/readout_transfer/aug.log 2>&1
  python3 - "$F" "$1" "$2" <<'PY' >> results/readout_transfer/aug_results.log
import json, sys
d = json.load(open(sys.argv[1]))["readouts"]["span"]
acc = " ".join(f"{k.replace('val_','')}={d[k]['acc']:.3f}" for k in ("val_iid","val_combo","val_amp"))
print(f"AUG seed={sys.argv[2]} frac={sys.argv[3]} {acc} combo_amp_mae={d['val_combo']['amp_mae']:.4f}")
PY
}
for S in 0 1 2; do run $S 0.5 0.5; done
for S in 1 2; do run $S 0.0 0.0; done      # seed 0 without augmentation is already recorded
echo "AUG COMPLETE" >> results/readout_transfer/aug_results.log
