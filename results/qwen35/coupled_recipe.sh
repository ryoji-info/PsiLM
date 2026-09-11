#!/bin/bash
# Qwen3.5 9B: the coupled phase, then the selective gate, resuming from phase A.
#
# Coupling l_fwd 13 / l_rev 26 of 32 -- 41% and 81%. The 41% matches every other
# backbone; the 81% does not (8B 61%, Gemma 62%), and it is chosen from a
# measured cost cliff rather than from principle. The backward pass has to cross
# the GatedDeltaNet layers, whose fast Metal kernel has no VJP, so each one above
# the injection falls back to a pure-ops scan that keeps its whole recurrence on
# the tape. At batch 2: l_rev 28 (3 such layers) 4.17 s/step, l_rev 26 (4) 4.69
# and 16.0 GB, l_rev 24 (6) 10.03 and 19.1 GB, l_rev 20 (9) 23.5 GB. The jump
# from 26 to 24 is 1.5x the work for 2.1x the time, because 19.1 GB is past this
# machine's GPU wired allowance and the buffers begin to spill. 26 is the deepest
# coupling that stays under it.
#
# That leaves only 6 layers above the injection, against 18 of 48 for Gemma and
# 14 of 36 for the 8B. It is the real risk in this run: a bridge that reads the
# pointer well -- phase A ended at CE 0.45 / 91% exact bins on the last hundred
# steps, the best warm-up pointer of any backbone here, though on twice the
# warm-up samples -- but has too little depth left to turn it into an answer.
# Watch eval_acc across chunks; if it stays at zero while x0_exact holds, depth
# is the suspect, not the readout.
#
# Batch 2 (batch 3 costs 14.19 s/step for 1.5x the samples), so 8,000 coupled
# steps to reach Gemma's 16,000 samples. eval-n is 16, not Gemma's 48: a rollout
# costs tens of seconds through the staged forward (on the ops-path scan, at the
# time; 48 of them added ~25 minutes to a 40-minute chunk). These rollouts track
# a trend, they do not publish a number -- the held-out evaluation and the
# guard-rail do that -- and Gemma's own eval_acc swung 25 points chunk to chunk
# at n=48, so read the trend across chunks, not any one chunk.
#
# The negatives are built first, not just because the no-harm phase needs them
# but because they take an hour and a half and a failure should surface now
# rather than ten hours in. They cannot be inherited: no-harm items are token id
# sequences paired with the backbone's own greedy continuation, and this
# tokenizer has a 248k vocab where Gemma's has 262k different ids.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
D=results/stage2_qwen35
LOG=results/qwen35/coupled.log
NEG=data/noharm_qwen35_all.json
export HF_HUB_DISABLE_XET=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface

# the last TRAINING record: train_log.jsonl interleaves eval records carrying
# only {step, eval_acc, eval_mae}, so a plain tail lands on the wrong kind and
# prints -1.000 for everything -- which is what phase A's log did. In the
# selective-gate phase the last record of a chunk is a NEGATIVES batch (every
# second step), so the NOHARM CHUNK lines show x0_ce/exact of 0.000 and the
# gate on non-physics prompts; the physics-side numbers are in train_log.jsonl.
last() { $PY -c "
import json
rows=[json.loads(l) for l in open('$D/train_log.jsonl')]
tr=[r for r in rows if 'loss_x0' in r]
ev=[r for r in rows if 'eval_acc' in r]
r=tr[-1] if tr else {}; e=ev[-1] if ev else {}
print('step=%d x0_ce=%.3f exact=%.3f ans=%.3f gate=%.4f inj=%.3f acc=%s' % (
    r.get('step',-1), r.get('loss_x0',-1), r.get('x0_exact',-1), r.get('loss_ans',-1),
    r.get('gate_ans', r.get('gate',-1)), r.get('inj_ratio_ans',-1), e.get('eval_acc','-')))"; }

# retain per-chunk checkpoints, tagging the no-harm ones: the export step splits
# phases on this suffix, because the no-harm arm replays step numbers
keep() { S=$($PY -c "import json;print(json.load(open('$D/bridges.npz.meta'))['step'])")
         cp $D/bridges.npz $D/bridges_step${S}$1.npz; echo $S; }

# --- negatives, from this backbone's own continuations -----------------------
# two passes: one keeping the benchmark's "Answer:" nudge line and one dropping
# it, merged, so the gate cannot learn the wrapper instead of relevance. Unlike
# Gemma's build (597 + 449 over different draws), both passes here use the same
# seed and so the same 597 prompts, and the stripper only knows GSM8K's nudge
# sentence, so the 197 MMLU prompts appear twice unchanged -- a reweighting.
# Runs alone -- it loads its own copy of the 8 GB model.
if [ ! -s $NEG ]; then
  echo "NOHARM BUILD START $(date +%H:%M)" >> $LOG
  for P in 1.0:nudge 0.0:nonudge; do
    P_PROB=${P%%:*}; P_NAME=${P##*:}
    $PY eval/build_noharm.py --model $M --hf-tokenizer $M --nudge-prob $P_PROB \
        --out data/noharm_qwen35_$P_NAME.json >> results/qwen35/build_noharm.log 2>&1 \
      || { echo "NOHARM BUILD FAILED ($P_NAME) $(date +%H:%M)" >> $LOG; exit 1; }
  done
  $PY -c "
import json
a=json.load(open('data/noharm_qwen35_nudge.json'))+json.load(open('data/noharm_qwen35_nonudge.json'))
json.dump(a, open('$NEG','w')); print(len(a))" >> $LOG 2>&1 \
    || { echo "NOHARM MERGE FAILED $(date +%H:%M)" >> $LOG; exit 1; }
  echo "NOHARM BUILT $($PY -c "import json;print(len(json.load(open('$NEG'))))") items $(date +%H:%M)" >> $LOG
fi

C="--model $M --hf-tokenizer $M --tag _qwen35 --batch 2 --l-rev 26 --channel value \
   --inj-cap 0.2 --gate-bias 0.0 --lam-x0 1.0 --clip module --detach-x0 \
   --readout-norm dim --calib-n 32 --readout-only 2000 --eval-n 16"

# --- coupled -----------------------------------------------------------------
echo "COUPLED START $(date +%H:%M)" >> $LOG
for i in $(seq 2 16); do   # chunk 1 done at step 2500
  T0=$(date +%s)
  $PY eval/mlx_stage2_train.py --steps 500 --lr 3e-4 $C >> $D/supervisor.log 2>&1 \
    || { echo "COUPLED FAILED chunk $i $(date +%H:%M)" >> $LOG; exit 1; }
  keep "" > /dev/null
  echo "COUPLED CHUNK $i $(last) [$(( ($(date +%s)-T0)/60 )) min]" >> $LOG
done
echo "COUPLED DONE $(date +%H:%M)" >> $LOG

# --- selective gate ----------------------------------------------------------
for i in 1 2 3; do
  T0=$(date +%s)
  $PY eval/mlx_stage2_train.py --steps 500 --lr 1e-4 $C --noharm-data $NEG \
      --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0 >> $D/supervisor.log 2>&1 \
    || { echo "NOHARM FAILED chunk $i $(date +%H:%M)" >> $LOG; exit 1; }
  keep "_noharm" > /dev/null
  echo "NOHARM CHUNK $i $(last) [$(( ($(date +%s)-T0)/60 )) min]" >> $LOG
done
echo "COUPLED+NOHARM COMPLETE $(date +%H:%M)" >> $LOG
