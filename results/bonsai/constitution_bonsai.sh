#!/bin/bash
# A constitution bridge on Ternary Bonsai 2 27B, for the PsiLM chat app's switch.
#
# The recipe is the Qwen3.5 9B full-width arm's (results/qwen35/constitution_train.sh,
# variant "all"), scaled to 64 layers: read the whole stream at layer 26 and write
# the whole stream at 48 (13/32 and 24/32 on the 9B), the fine-tuned 0.5B partner,
# gate bias 0, cap 0.2, a no-harm step every second step, 2,000 examples (1,000
# steps at batch 2), then the 50-item evaluations and the guard-rail.
#
# The data are the 9B's PROMPTS, verbatim, with Bonsai's OWN continuations: Bonsai
# with the constitution excerpt is the teacher, Bonsai without it the student.
# The two tokenizers are identical; the chat templates differ as files but render
# the same ids for these single-turn, thinking-off prompts (precheck.py checks
# every prompt against the 9B's). No `datasets` import happens in a process that
# loads the model.
#
# Revised 2026-09-23 after a 33-agent review, before any training step ran:
#   - the smoke runs measured peak memory on steps 0-5, whose batches are about a
#     third as long as the longest ones training reaches (step 407's no-harm batch
#     is 489 tokens at batch 2); they now run on the longest training item and the
#     longest negative, so the peak they report is the training peak
#   - --checkpoint-from recomputed nothing on the GPU (mx.checkpoint let every
#     window layer's recompute run early); psilm/mlx/staged.py now orders each
#     recompute after its cotangent, and precheck.py measures the saving on the GPU
#   - the no-harm builder's resume was keyed by source, which is not unique
#   - chunks of 100 steps instead of 500, so a crash costs at most 100 steps
#   - batch 1 trains 2,000 steps (the same 2,000 examples), not 1,000
#   - a projected training time over TRAIN_H_MAX hours stops for a person
#   - a smoke that crashes is recorded and not rerun on a relaunch, and a relaunch
#     after training has started keeps the checkpoint's batch and write layer
#   - the partner path is absolute (the chat app reads it from the checkpoint)
#   - the guard-rail of the 9B recipe (GSM8K, MMLU, BoolQ, red-team) is added
#
# Memory decides the batch. The 9B at batch 2 peaked at 48.9 GB (swap-backed) on
# this 24 GB machine. Long-batch smokes measure the peak at batch 1 writing at 56,
# batch 1 at 48, then batch 2 at 48 (cheapest first, so an overflow is met on the
# smallest configuration); the first-preferred of b2l48, b1l48, b1l56 that peaks
# under PEAK_MAX GB trains. If none fits, the chain stops and says so -- that is a
# decision for a person.
#
# LAUNCH A COPY, NOT THIS FILE. Waits for the KL rescoring (KL-RESCORE COMPLETE or
# INCOMPLETE in results/qwen35/kl_rescore.log) and for any running builder, trainer
# or evaluator. Ends with BONSAI-CONSTITUTION COMPLETE.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
P=/Users/rxiii/Documents/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787
CM=$PWD/results/constitution_model/qwen2.5-0.5b-constitution
TAG=bonsai27b
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
DATA_HE=data/constitution_helpful_test_$TAG.json
NEG=data/noharm_${TAG}_all.json
LOG=results/bonsai/constitution_bonsai.log
PEAK_MAX=${PEAK_MAX:-50}
TRAIN_H_MAX=${TRAIN_H_MAX:-72}
EXAMPLES=2000                   # the 9B recipe: 2 x 500 steps at batch 2
CHUNK=100
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
jget() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): d=d[k]
print(d)" "$1" "$2" 2>/dev/null; }

step "BONSAI WAITING for the KL rescoring"
MISS=0
until grep -qE "KL-RESCORE (COMPLETE|INCOMPLETE)" results/qwen35/kl_rescore.log 2>/dev/null; do
  if pgrep -f 'kl_rescore_all\.py' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "the rescoring is neither running nor finished; proceeding"; break; }
  sleep 120
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py|build_constitution_data\.py|build_noharm\.py|bonsai/precheck\.py' > /dev/null; do sleep 60; done
[ -s $P/model.safetensors ] && [ -s $CM/model.safetensors ] && [ -s data/constitution_train_qwen35.json ] \
  && [ -s data/noharm_qwen35_all.json ] || { step "PREREQ files missing"; exit 1; }
step "BONSAI START"

# ---- 1. data: the 9B's prompts, Bonsai's continuations ------------------------
if [ ! -s $DATA_TR ] || [ ! -s $DATA_VA ] || [ ! -s $DATA_TE ] || [ ! -s $DATA_HE ]; then
  for j in 1 2 3; do
    $PY eval/build_constitution_data.py --tag $TAG --model $P --hf-tokenizer $P --prompts-from qwen35 \
        --n-train 1000 --n-val 100 --n-test 100 --n-helpful-test 100 --max-new 128 \
        >> results/bonsai/build_data_run.log 2>&1 && break
    step "DATA attempt $j exited; resuming"; sleep 30
  done
  [ -s $DATA_TR ] && [ -s $DATA_VA ] && [ -s $DATA_TE ] && [ -s $DATA_HE ] || { step "DATA FAILED"; exit 1; }
  step "DATA DONE"
fi
# (the python goes through a variable: bash 3.2 brace-expands a {a,b} inside "...$(...)...")
if ! grep -q "^DATA STATS" $LOG; then
  DS=$($PY -c "import json;d=json.load(open('data/constitution_${TAG}_stats.json'));t=d['timing'];print({k:t.get(k) for k in ('items_generated','sec_per_item')}, {s:{k:v.get(k) for k in ('n','teacher_differs_rate','refusal_rate_teacher','refusal_rate_base','mean_len_teacher')} for s,v in d['splits'].items()})")
  [ -n "$DS" ] && step "DATA STATS: $DS"
fi

nneg() { $PY -c "import json;print(len(json.load(open('$NEG'))))" 2>/dev/null || echo 0; }
NEG_N=$($PY -c "import json;print(len(json.load(open('data/noharm_qwen35_all.json'))))")
if [ "$(nneg)" -lt $NEG_N ]; then
  for j in 1 2 3; do
    $PY eval/build_noharm.py --model $P --hf-tokenizer $P --prompts-from data/noharm_qwen35_all.json \
        --max-new 32 --out $NEG >> results/bonsai/build_noharm_run.log 2>&1 && break
    step "NOHARM attempt $j exited; resuming"; sleep 30
  done
  N=$(nneg); [ "$N" -ge $NEG_N ] || { step "NOHARM FAILED ($N of $NEG_N items)"; exit 1; }
  step "NOHARM DONE: $N items"
fi

# ---- 2. the checks that must hold before a training step ---------------------
PC=results/bonsai/precheck.json
# (a full run with the model, and not older than the data it checked)
pc_ok() { [ "$(jget $PC ok)" = "True" ] && [ "$(jget $PC stream.ok)" = "True" ] && [ ! $NEG -nt $PC ] && [ ! $DATA_TR -nt $PC ]; }
if ! pc_ok; then
  $PY results/bonsai/precheck.py > results/bonsai/precheck.log 2>&1
  pc_ok || { step "PRECHECK FAILED ($PC, results/bonsai/precheck.log); a person decides"; exit 1; }
  PS=$($PY -c "import json;d=json.load(open('$PC'));print({k:[round(r['rel'],6) for r in v] for k,v in d['stream'].items() if k!='ok'}, 'grad', d['grad'], 'memory', d['memory'], 'packed', d['packed'])")
  step "PRECHECK OK: stream $PS"
fi

C="--model $P --hf-tokenizer $P --const-model $CM --lr 3e-4 --l-fwd 26 \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --k-fwd 8 --m-tokens 8 --read-dims all --write-dims all --checkpoint-from -1"
T=${TAG}_all D=results/stage2c_${TAG}_all
META=$D/bridges.npz.meta

# ---- 3. memory decides the batch, on the longest batches training will see ------
peak_of() { grep 'CHUNK DONE' results/stage2c_${TAG}_$1/supervisor.log 2>/dev/null | tail -1 | sed -E 's/.*peak=([0-9.]+)GB.*/\1/'; }
sps_of()  { $PY -c "import json;r=[json.loads(l) for l in open('results/stage2c_${TAG}_$1/train_log.jsonl') if 'sec_per_step' in l];print(r[-1]['sec_per_step'] if r else '')" 2>/dev/null; }
le() { $PY -c "import sys;sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$2"; }
smoke() {   # $1 name, $2 batch, $3 l_rev, $4 steps, the rest: extra trainer flags
  local NAME=$1 B=$2 L=$3 S=$4 SD=results/stage2c_${TAG}_$1; shift 4
  if [ -n "$(peak_of $NAME)" ] || grep -qE "SMOKE $NAME (TIMED OUT|FAILED)" $LOG; then return 0; fi
  mkdir -p $SD
  # a configuration that overflows memory swaps rather than fails: give it an
  # hour (macOS has no `timeout`), then call it a configuration that does not fit
  $PY eval/mlx_constitution_train.py --tag ${TAG}_$NAME --steps $S --fresh $C --batch $B --l-rev $L \
      --calib-n 8 --eval-n 2 "$@" >> $SD/supervisor.log 2>&1 &
  local SP=$! W=0 RC
  while kill -0 $SP 2>/dev/null; do
    sleep 60; W=$((W + 1))
    if [ $W -ge ${SMOKE_MAX_MIN:-60} ]; then
      kill $SP; sleep 10; kill -9 $SP 2>/dev/null
      step "SMOKE $NAME TIMED OUT after $W min"
      break
    fi
  done
  wait $SP 2>/dev/null; RC=$?
  if [ -z "$(peak_of $NAME)" ] && ! grep -q "SMOKE $NAME TIMED OUT" $LOG; then
    step "SMOKE $NAME FAILED (exit $RC; see $SD/supervisor.log)"
  fi
}

if [ -s $META ]; then
  # training has started: its batch and write layer are fixed, whatever a smoke says now
  B=$(jget $META args.batch); L=$(jget $META l_rev)
  [ -n "$B" ] && [ -n "$L" ] || { step "BONSAI STOPPED: $META unreadable"; exit 1; }
  step "TRAIN CONFIG from the checkpoint: batch $B, write $L"
else
  LT=results/bonsai/smoke_long_train.json LN=results/bonsai/smoke_long_noharm.json
  $PY -c "
import json
t = json.load(open('$DATA_TR')); n = json.load(open('$NEG'))
t.sort(key=lambda r: -(len(r['prompt_ids']) + len(r['teacher_ids'])))
n.sort(key=lambda r: -(len(r['prompt_ids']) + len(r['target_ids'])))
json.dump(t[:1] * 2, open('$LT', 'w')); json.dump(n[:1] * 2, open('$LN', 'w'))   # both draws are the longest
print('longest positive', len(t[0]['prompt_ids']) + len(t[0]['teacher_ids']),
      'longest negative', len(n[0]['prompt_ids']) + len(n[0]['target_ids']))
" >> results/bonsai/smoke_long.txt 2>&1 || { step "BONSAI STOPPED: could not write the long-batch smoke data"; exit 1; }
  grep -q "^SMOKE DATA" $LOG || step "SMOKE DATA: $(tail -1 results/bonsai/smoke_long.txt)"
  # step 0 is a positive batch, step 1 a no-harm batch: two steps cover both longest cases
  # (each file holds its longest item twice, so batch 1 draws it too)
  LONG="--data $LT --noharm-data $LN"
  declare -a FIT=()
  for cfg in "b1l56:1:56" "b1l48:1:48" "b2l48:2:48"; do
    NAME=${cfg%%:*}; REST=${cfg#*:}; B=${REST%%:*}; L=${REST##*:}
    if [ "$NAME" = "b2l48" ] && [ -n "$(peak_of longb1l48)" ]; then
      # batch 2 doubles the activations: skip it if batch 1's peak already says it cannot fit
      P1=$(peak_of longb1l48)
      PROJ=$($PY -c "print(round(9.5 + 2 * ($P1 - 9.5), 1))")
      if ! le "$PROJ" "$(echo "$PEAK_MAX * 1.25" | bc)"; then
        grep -q "SMOKE longb2l48 SKIPPED" $LOG || step "SMOKE longb2l48 SKIPPED: batch 1 peaked at $P1 GB, batch 2 projects to $PROJ"
        continue
      fi
    fi
    smoke long$NAME $B $L 2 $LONG
    PK=$(peak_of long$NAME)
    grep -q "^SMOKE long$NAME batch" $LOG || step "SMOKE long$NAME batch $B l_rev $L (longest batches): peak ${PK:-none} GB"
    if [ -z "$PK" ] && ! grep -q "SMOKE long$NAME TIMED OUT" $LOG \
       && ! grep -qiE "malloc|resource limit|out of memory|insufficient memory" results/stage2c_${TAG}_long$NAME/supervisor.log; then
      # a crash that is not a memory error says nothing about fit: do not let it move the recipe
      step "BONSAI STOPPED: smoke long$NAME crashed without a memory error (results/stage2c_${TAG}_long$NAME/supervisor.log); a person decides (to rerun it, delete its FAILED line from this log)"
      exit 1
    fi
    if [ -n "$PK" ] && le "$PK" "$PEAK_MAX"; then FIT+=("$B:$L"); else break; fi
  done
  CHOSEN=""
  for want in "2:48" "1:48" "1:56"; do
    for f in "${FIT[@]}"; do [ "$f" = "$want" ] && CHOSEN=$want; done
    [ -n "$CHOSEN" ] && break
  done
  [ -n "$CHOSEN" ] || { step "BONSAI STOPPED: no configuration peaked under $PEAK_MAX GB on the longest batches (or every smoke failed; see the SMOKE lines); a person decides"; exit 1; }
  B=${CHOSEN%%:*}; L=${CHOSEN##*:}
  # time: the ordinary first six steps give the typical step
  smoke timeb${B}l${L} $B $L 6
  SPS=$(sps_of timeb${B}l${L})
  [ -n "$SPS" ] || { step "BONSAI STOPPED: the timing smoke for batch $B write $L produced no s/step"; exit 1; }
  HOURS=$($PY -c "print(round($SPS * $EXAMPLES / $B / 3600, 1))")
  step "SMOKE timeb${B}l${L}: $SPS s/step -> about $HOURS h for $((EXAMPLES / B)) steps"
  le "$HOURS" "$TRAIN_H_MAX" || { step "BONSAI STOPPED: batch $B write $L fits but projects to $HOURS h (> $TRAIN_H_MAX); a person decides"; exit 1; }
  [ "$L" = "48" ] || step "NOTE: the write moves to layer 56 of 64 (the 9B's recipe is 24/32 = 48/64); a deviation"
fi
TOTAL=$((EXAMPLES / B))
step "TRAIN CONFIG batch $B, read 26, write $L of 64, $TOTAL steps in chunks of $CHUNK"

# ---- 4. train in chunks of 100 steps, then the 50-item evaluations ---------------
mkdir -p $D
cur() { local s; s=$(jget $META step); echo ${s:-0}; }
while [ "$(cur)" -lt $TOTAL ]; do
  S0=$(cur); NS=$((TOTAL - S0)); [ $NS -gt $CHUNK ] && NS=$CHUNK
  OK=""
  for j in 1 2 3; do
    S1=$(cur); [ "$S1" -ge $((S0 + NS)) ] && { OK=1; break; }   # saved, then exited non-zero
    FLAG=""; [ -s $META ] || FLAG=--fresh
    if [ -s $META ]; then
      [ "$(jget $META l_rev)" = "$L" ] && [ "$(jget $META args.batch)" = "$B" ] \
        || { step "BONSAI STOPPED: the checkpoint's batch/write differ from $B/$L"; exit 1; }
    fi
    $PY eval/mlx_constitution_train.py --tag $T --steps $((S0 + NS - S1)) $FLAG $C --batch $B --l-rev $L \
        --calib-n 32 --eval-n 32 >> $D/supervisor.log 2>&1 && { OK=1; break; }
    step "TRAIN steps $S0+$NS attempt $j exited; resuming from step $(cur)"; sleep 30
  done
  [ -n "$OK" ] || { step "TRAIN FAILED at step $(cur)"; exit 1; }
  step "TRAIN $(cur)/$TOTAL: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
done

for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
  S=${pair%%:*}; F=${pair##*:}
  [ -s $D/eval_$S.json ] && continue
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $P --hf-tokenizer $P \
      --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
      --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $S FAILED"; exit 1; }
  step "EVAL $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
done

# ---- 5. the guard-rail of the 9B recipe ----------------------------------------
# The 9B's task cache restores under Bonsai's tokenizer (same fingerprint, no
# `datasets` import); the red-team prompts are the 9B's test split, which are this
# run's test prompts verbatim.
GT=const_${TAG}_all
if [ ! -s results/bench/${GT}_guardrail_summary.json ]; then
  G="--tasks-cache results/bench/tasks_const_qwen35_n100.json --model $P --hf-tokenizer $P \
    --bridge-kind constitution --ckpt $D/bridges.npz --const-model $CM \
    --redteam-data data/constitution_test_qwen35.json --datasets redteam,gsm8k,mmlu,boolq \
    --arms base,psilm,zeroed --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 \
    --max-new-redteam 128 --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
  FLAG=--fresh; [ -s results/bench/${GT}_guardrail.rows.jsonl ] && FLAG=--resume
  for i in 1 2 3 4 5 6; do
    $PY eval/bench_guardrail.py --tag $GT --n 100 $G $FLAG >> results/bench/${GT}_run.log 2>&1 \
      && { $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; break; }
    step "GUARDRAIL attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  [ -s results/bench/${GT}_guardrail_summary.json ] || { step "GUARDRAIL FAILED"; exit 1; }
  step "GUARDRAIL COMPLETE: $(grep -A5 '^dataset ' results/bench/${GT}_run.log | tail -4 | tr -s ' ' | tr '\n' '|')"
fi
step "BONSAI-CONSTITUTION COMPLETE"
