#!/bin/bash
# A constitution bridge on Ternary Bonsai 2 27B, for the PsiLM chat app's switch.
#
# The recipe is the Qwen3.5 9B full-width arm's (results/qwen35/constitution_train.sh,
# variant "all"), scaled to 64 layers: read the whole stream at layer 26 and write
# the whole stream at 48 (13/32 and 24/32 on the 9B), the fine-tuned 0.5B partner,
# gate bias 0, cap 0.2, a no-harm step every second step, 2 x 500 steps.
#
# The data are the 9B's PROMPTS, verbatim (the two tokenizers and chat templates
# are identical, checked), with Bonsai's OWN continuations: Bonsai with the
# constitution excerpt is the teacher, Bonsai without it the student. No
# `datasets` import happens in a process that loads the model.
#
# Memory decides the batch. The 9B at batch 2 peaked at 48.9 GB (swap-backed) on
# this 24 GB machine; Bonsai's backward window is 16 layers of width 5120 against
# the 9B's 8 of 4096. Short smoke runs measure the peak at batch 2, then 1, then
# batch 1 with the write moved to layer 56; the first under PEAK_MAX GB trains. If
# none fits, the chain stops and says so -- that is a decision for a person.
#
# LAUNCH A COPY, NOT THIS FILE. Waits for the KL rescoring (KL-RESCORE COMPLETE or
# INCOMPLETE in results/qwen35/kl_rescore.log). Ends with BONSAI-CONSTITUTION COMPLETE.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P=/Users/rxiii/Documents/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787
CM=results/constitution_model/qwen2.5-0.5b-constitution
TAG=bonsai27b
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
DATA_HE=data/constitution_helpful_test_$TAG.json
NEG=data/noharm_${TAG}_all.json
LOG=results/bonsai/constitution_bonsai.log
PEAK_MAX=${PEAK_MAX:-50}
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "BONSAI WAITING for the KL rescoring"
MISS=0
until grep -qE "KL-RESCORE (COMPLETE|INCOMPLETE)" results/qwen35/kl_rescore.log 2>/dev/null; do
  if pgrep -f 'kl_rescore_all\.py' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "the rescoring is neither running nor finished; proceeding"; break; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py|build_constitution_data\.py|build_noharm\.py' > /dev/null; do sleep 60; done
[ -s $P/model.safetensors ] && [ -s $CM/model.safetensors ] && [ -s data/constitution_train_qwen35.json ] \
  && [ -s data/noharm_qwen35_all.json ] || { step "PREREQ files missing"; exit 1; }
step "BONSAI START"

# ---- 1. data: the 9B's prompts, Bonsai's continuations ------------------------
if [ ! -s $DATA_TR ] || [ ! -s $DATA_HE ]; then
  for j in 1 2 3; do
    $PY eval/build_constitution_data.py --tag $TAG --model $P --hf-tokenizer $P --prompts-from qwen35 \
        --n-train 1000 --n-val 100 --n-test 100 --n-helpful-test 100 --max-new 128 \
        >> results/bonsai/build_data_run.log 2>&1 && break
    step "DATA attempt $j exited; resuming"; sleep 30
  done
  [ -s $DATA_TR ] && [ -s $DATA_VA ] && [ -s $DATA_TE ] && [ -s $DATA_HE ] || { step "DATA FAILED"; exit 1; }
  step "DATA DONE: $($PY -c "import json;d=json.load(open('data/constitution_${TAG}_stats.json'));t=d['timing'];print({k:t.get(k) for k in ('items_generated','sec_per_item')}, {s:{k:v.get(k) for k in ('n','differs_rate','teacher_refusal','base_refusal') if k in v} for s,v in d['splits'].items()})")"
fi
if [ ! -s $NEG ] || [ "$($PY -c "import json;print(len(json.load(open('$NEG'))))")" -lt 1190 ]; then
  for j in 1 2 3; do
    $PY eval/build_noharm.py --model $P --hf-tokenizer $P --prompts-from data/noharm_qwen35_all.json \
        --max-new 32 --out $NEG >> results/bonsai/build_noharm_run.log 2>&1 && break
    step "NOHARM attempt $j exited; resuming"; sleep 30
  done
  [ -s $NEG ] || { step "NOHARM FAILED"; exit 1; }
  step "NOHARM DONE: $($PY -c "import json;print(len(json.load(open('$NEG'))))") items"
fi

C="--model $P --hf-tokenizer $P --const-model $CM --lr 3e-4 --l-fwd 26 \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --k-fwd 8 --m-tokens 8 --read-dims all --write-dims all --checkpoint-from -1"

# ---- 2. memory decides the batch ----------------------------------------------
peak_of() { grep 'CHUNK DONE' results/stage2c_${TAG}_smoke$1/supervisor.log 2>/dev/null | tail -1 | sed -E 's/.*peak=([0-9.]+)GB.*/\1/'; }
sps_of()  { $PY -c "import json;r=[json.loads(l) for l in open('results/stage2c_${TAG}_smoke$1/train_log.jsonl') if 'sec_per_step' in l];print(r[-1]['sec_per_step'] if r else '')" 2>/dev/null; }
CHOSEN=""
for cfg in "b2l48:2:48" "b1l48:1:48" "b1l56:1:56"; do
  NAME=${cfg%%:*}; REST=${cfg#*:}; B=${REST%%:*}; L=${REST##*:}
  D=results/stage2c_${TAG}_smoke$NAME
  if [ -z "$(peak_of $NAME)" ] && ! grep -q "SMOKE $NAME TIMED OUT" $LOG; then
    mkdir -p $D
    # a configuration that overflows memory swaps rather than fails: give it an
    # hour (macOS has no `timeout`), then call it a configuration that does not fit
    $PY eval/mlx_constitution_train.py --tag ${TAG}_smoke$NAME --steps 6 --fresh $C --batch $B --l-rev $L \
        --calib-n 8 --eval-n 2 >> $D/supervisor.log 2>&1 &
    SP=$!; W=0
    while kill -0 $SP 2>/dev/null; do
      sleep 60; W=$((W + 1))
      if [ $W -ge ${SMOKE_MAX_MIN:-60} ]; then
        kill $SP; sleep 10; kill -9 $SP 2>/dev/null
        step "SMOKE $NAME TIMED OUT after $W min"
        break
      fi
    done
    wait $SP 2>/dev/null
  fi
  PK=$(peak_of $NAME); SPS=$(sps_of $NAME)
  step "SMOKE $NAME batch $B l_rev $L: peak ${PK:-none} GB, ${SPS:-?} s/step"
  if [ -n "$PK" ] && $PY -c "import sys;sys.exit(0 if float('$PK') <= $PEAK_MAX else 1)"; then
    CHOSEN="$B:$L"; break
  fi
done
[ -n "$CHOSEN" ] || { step "BONSAI STOPPED: no smoke configuration peaked under $PEAK_MAX GB; a person decides"; exit 1; }
B=${CHOSEN%%:*}; L=${CHOSEN##*:}
step "TRAIN CONFIG batch $B, read 26, write $L of 64"

# ---- 3. train, 2 x 500 steps, then the 50-item evaluations ----------------------
NAME=all T=${TAG}_all D=results/stage2c_${TAG}_all
mkdir -p $D
DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
FLAG=""; [ "$DONE" = "0" ] && FLAG=--fresh
i=$DONE
while [ $i -lt 2 ]; do
  i=$((i + 1))
  for j in 1 2 3; do
    $PY eval/mlx_constitution_train.py --tag $T --steps 500 $FLAG $C --batch $B --l-rev $L \
        --calib-n 32 --eval-n 32 >> $D/supervisor.log 2>&1 && break
    step "TRAIN chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
    [ $j = 3 ] && { step "TRAIN chunk $i FAILED"; exit 1; }
  done
  FLAG=""
  step "TRAIN chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
done
for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
  S=${pair%%:*}; F=${pair##*:}
  [ -s $D/eval_$S.json ] && continue
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $P --hf-tokenizer $P \
      --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
      --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $S FAILED"; exit 1; }
  step "EVAL $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
done
step "BONSAI-CONSTITUTION COMPLETE"
