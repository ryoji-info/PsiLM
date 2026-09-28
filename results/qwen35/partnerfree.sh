#!/bin/bash
# Was the constitution partner ever needed? The 9B full-width bridge, trained with none.
#
# The recipe is the full-width arm's (results/qwen35/constitution_train.sh, variant
# "all"): same data, same negatives, same steps and batch, read 13, write 24 of 32,
# gate bias 0, cap 0.2, a no-harm step every second step. ONE thing is changed:
# --const-model constant:0 puts a fixed draw where the partner's features were, so
# nothing the forward bridge reads and nothing a partner computes reaches the
# write. What trains is the reverse bridge's map of a constant and the gated write.
#
# What the result is read against, fixed in advance
# (results/constitution/partnerfree_preregistration.json): the partner's own arm and
# its plain-partner twin (stage2c_qwen35_allplain), the only other full-width run of
# this recipe. No seed replicate of the full-width recipe exists, so the twin's gap
# to the partner arm (keyword 4:4 discordant, adjudicated 19:3 against 21:2) is the
# nearest thing to a run-to-run spread, and it mixes noise with the change of
# partner. A result between "match" and "partner needed" calls for a seed replicate
# before anything is claimed.
#
# The order puts the decisive comparison first: train, the 50-item evaluations, the
# 400 red-team prompts, then the guard-rail and the teacher ceiling. The base arm of
# the 400 is borrowed from the recorded run if the fixed-token chain showed today's
# harness regenerates it token for token (results/qwen35/reproduction.json);
# otherwise it is generated. `zeroed` is left out: it is the base arm by
# construction.
#
# Training is in 100-step chunks (a crash costs at most 100 steps; the batches are
# a function of the step, so the run is the 2 x 500 one). The backward pass is
# recomputed (--checkpoint-from -1), which since PsiLM 474105f saves the memory it
# was meant to: the peak is measured on the longest batches first.
#
# LAUNCH A COPY NAMED partnerfree_run.sh, NOT THIS FILE. Waits for the fixed-token
# chain (FIXED-RT400 COMPLETE or ENDED in results/qwen35/fixed_rt400.log), then for
# the GPU lock the two chains share (results/qwen35/.gpu.lock). Ends with
# PARTNERFREE COMPLETE; any other exit writes PARTNERFREE ENDED.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
TAG=qwen35
V=nopartner
SPEC=constant:0
T=${TAG}_$V D=results/stage2c_${TAG}_$V
REF=results/stage2c_${TAG}_all
META=$D/bridges.npz.meta
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
DATA_HE=data/constitution_helpful_test_$TAG.json
NEG=data/noharm_${TAG}_all.json
RT=data/redteam_qwen35_n400.json
RTCACHE=results/bench/tasks_rt400_qwen35_n400.json
LOG=results/qwen35/partnerfree.log
PEAK_MAX=${PEAK_MAX:-50}
TOTAL=1000 CHUNK=100 B=2 L=24
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
jget() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): d=d[k]
print(d)" "$1" "$2" 2>/dev/null; }
le() { $PY -c "import sys;sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$2"; }
LOCK=results/qwen35/.gpu.lock
HELD="" SP=""
bye() { local rc=$?; [ -n "$SP" ] && kill $SP 2>/dev/null; [ -n "$HELD" ] && rm -rf $LOCK
        step "PARTNERFREE ENDED rc=$rc"; }
trap bye EXIT
trap 'exit 143' TERM; trap 'exit 129' HUP; trap 'exit 130' INT
# anchored on the interpreter: a monitor or an editor that merely NAMES a program is not a GPU job
GPU='^[^ ]*[Pp]ython[0-9.]* [^ ]*(mlx_constitution_(train|eval)|bench_guardrail|teacher_ceiling|constitution_compress|constitution_fixed_tokens)\.py'
gpu_wait() {   # until no GPU program runs and the lock is ours
  local w=0 o
  while :; do
    if ! pgrep -f "$GPU" > /dev/null; then
      if mkdir $LOCK 2>/dev/null; then echo $$ > $LOCK/pid; HELD=1; return 0; fi
      o=$(cat $LOCK/pid 2>/dev/null)
      [ -n "$o" ] && ! kill -0 "$o" 2>/dev/null && { step "stale GPU lock of $o removed"; rm -rf $LOCK; continue; }
    fi
    w=$((w + 1))
    [ $w = 10 ] && step "GPU BUSY: $(pgrep -fl "$GPU" | head -1 | cut -c1-120) lock $(cat $LOCK/pid 2>/dev/null)"
    sleep 60
  done
}

step "PARTNERFREE WAITING for the fixed-token chain"
marks() { local n; n=$(grep -cE "FIXED-RT400 (COMPLETE|ENDED)" results/qwen35/fixed_rt400.log 2>/dev/null); echo ${n:-0}; }
N0=$(marks)
MISS=0
until [ "$(marks)" -gt "$N0" ]; do
  if pgrep -f '^(/bin/)?bash [^ ]*fixed_rt400_run\.sh' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "the fixed-token chain is neither running nor finished; proceeding"; break; }
  sleep 120
done
gpu_wait
[ -s $DATA_TR ] && [ -s $DATA_VA ] && [ -s $DATA_TE ] && [ -s $DATA_HE ] && [ -s $NEG ] && [ -s $RT ] \
  && [ -s $RTCACHE ] && [ -s $M/model.safetensors ] && [ -s $REF/base_eval.json ] \
  || { step "PREREQ files missing"; exit 1; }
for t in constitution_fixed_tokens const_arm_compare bench_seed_rows; do
  $PY eval/$t.py --self-test >> results/qwen35/partnerfree_selftest.log 2>&1 \
    || { step "SELF-TEST FAILED: eval/$t.py (results/qwen35/partnerfree_selftest.log)"; exit 1; }
done
step "PARTNERFREE START ($SPEC)"

C="--model $M --hf-tokenizer $M --const-model $SPEC --batch $B --lr 3e-4 --l-fwd 13 --l-rev $L \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim \
  --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --k-fwd 8 --m-tokens 8 --read-dims all --write-dims all --checkpoint-from -1"
FULL="--data $DATA_TR --val $DATA_VA --noharm-data $NEG --calib-n 32 --eval-n 32"

# ---- 1. the peak on the longest batches, before any training step ---------------
peak_of() { grep 'CHUNK DONE' results/stage2c_${TAG}_$1/supervisor.log 2>/dev/null | tail -1 | sed -E 's/.*peak=([0-9.]+)GB.*/\1/'; }
if [ ! -s $META ]; then
  SM=${V}_long SD=results/stage2c_${TAG}_${V}_long
  if [ -z "$(peak_of $SM)" ] && ! grep -qE "SMOKE (TIMED OUT|FAILED)" $LOG; then
    mkdir -p $SD
    $PY -c "
import json
t = json.load(open('$DATA_TR')); n = json.load(open('$NEG'))
t.sort(key=lambda r: -(len(r['prompt_ids']) + len(r['teacher_ids'])))
n.sort(key=lambda r: -(len(r['prompt_ids']) + len(r['target_ids'])))
json.dump(t[:1] * 2, open('$SD/long_train.json', 'w')); json.dump(n[:1] * 2, open('$SD/long_noharm.json', 'w'))
print('longest positive', len(t[0]['prompt_ids']) + len(t[0]['teacher_ids']),
      'longest negative', len(n[0]['prompt_ids']) + len(n[0]['target_ids']))
" > $SD/long.txt 2>&1 || { step "PARTNERFREE STOPPED: could not write the long-batch data"; exit 1; }
    $PY eval/mlx_constitution_train.py --tag ${TAG}_$SM --steps 2 --fresh $C --data $SD/long_train.json \
        --val $DATA_VA --noharm-data $SD/long_noharm.json --calib-n 8 --eval-n 2 >> $SD/supervisor.log 2>&1 &
    SP=$!; W=0
    while kill -0 $SP 2>/dev/null; do
      sleep 60; W=$((W + 1))
      [ $W -ge 60 ] && { kill $SP; sleep 10; kill -9 $SP 2>/dev/null; step "SMOKE TIMED OUT after $W min"; break; }
    done
    wait $SP 2>/dev/null; RC=$?
    [ -z "$(peak_of $SM)" ] && ! grep -q "SMOKE TIMED OUT" $LOG && step "SMOKE FAILED (exit $RC; $SD/supervisor.log)"
  fi
  PK=$(peak_of $SM)
  step "SMOKE batch $B write $L on the longest batches ($(cat $SD/long.txt 2>/dev/null)): peak ${PK:-none} GB"
  [ -n "$PK" ] && le "$PK" "$PEAK_MAX" \
    || { step "PARTNERFREE STOPPED: the smoke did not finish under $PEAK_MAX GB; a person decides (to measure again, delete $SD/supervisor.log and the SMOKE lines of this log; to accept this peak, relaunch with PEAK_MAX=<GB>)"; exit 1; }
fi
SP=""

# ---- 2. train, 1000 steps in chunks of 100 ---------------------------------------
mkdir -p $D
cur() { local s; s=$(jget $META step); echo ${s:-0}; }
mine() {   # the checkpoint in $D was trained by THIS recipe, every argument of it
  $PY -c "
import json, sys
m = json.load(open('$META')); a = m['args']
want = {'const_model': '$SPEC', 'model': '$M', 'batch': $B, 'lr': 3e-4, 'l_fwd': 13, 'l_rev': $L,
        'gate_bias': 0.0, 'inj_cap': 0.2, 'clip': 'module', 'readout_norm': 'dim', 'noharm_every': 2,
        'noharm_gate_only': 1, 'lam_gate': 1.0, 'k_fwd': 8, 'm_tokens': 8, 'read_dims': 'all',
        'write_dims': 'all', 'checkpoint_from': -1, 'data': '$DATA_TR', 'val': '$DATA_VA',
        'noharm_data': '$NEG'}
bad = {k: a.get(k) for k, v in want.items() if a.get(k) != v}
if m.get('const_model') != '$SPEC:896':
    bad['meta.const_model'] = m.get('const_model')
print(json.dumps(bad))
sys.exit(1 if bad else 0)" 2>&1; }
gate() {   # the drift gate: the base arm on the validation prompts is the partner run's
  local DR
  DR=$($PY -c "
import json
a = json.load(open('$D/base_eval.json')); b = json.load(open('$REF/base_eval.json'))
ka, kb = next(iter(a)), next(iter(b))
print('ok' if ka == kb and abs(a[ka]['ce'] - b[kb]['ce']) <= 1e-3 else 'DRIFT', a[ka]['ce'], b[kb]['ce'])" 2>&1)
  step "DRIFT GATE base CE on the validation prompts, this run and the partner's: $DR"
  case "$DR" in ok*) return 0;; esac
  step "PARTNERFREE STOPPED: the base arm is not the partner run's; a person decides"
  exit 1
}
gated() { grep -q '^DRIFT GATE .*: ok' $LOG; }
if [ -s $META ]; then
  WHY=$(mine) || { step "PARTNERFREE STOPPED: the checkpoint in $D is not this run's: $WHY"; exit 1; }
  [ "$(cur)" -ge $CHUNK ] && ! gated && gate       # a relaunch does not get past a gate that stopped it
fi
while [ "$(cur)" -lt $TOTAL ]; do
  S0=$(cur); NS=$((TOTAL - S0)); [ $NS -gt $CHUNK ] && NS=$CHUNK
  OK=""
  for j in 1 2 3; do
    S1=$(cur); [ "$S1" -ge $((S0 + NS)) ] && { OK=1; break; }
    FLAG=""; [ -s $META ] || FLAG=--fresh
    if [ -s $META ]; then
      WHY=$(mine) || { step "PARTNERFREE STOPPED: the checkpoint in $D is not this run's: $WHY"; exit 1; }
    fi
    $PY eval/mlx_constitution_train.py --tag $T --steps $((S0 + NS - S1)) $FLAG $C $FULL \
        >> $D/supervisor.log 2>&1 && { OK=1; break; }
    [ $j = 3 ] && break
    step "TRAIN steps $S0+$NS attempt $j exited; resuming from step $(cur)"; sleep 30
  done
  [ -n "$OK" ] || { step "TRAIN FAILED at step $(cur)"; exit 1; }
  step "TRAIN $(cur)/$TOTAL: $(grep "CHUNK DONE step=$(cur) " $D/supervisor.log | tail -1)"
  gated || gate
done
WHY=$(mine) || { step "PARTNERFREE STOPPED: the trained checkpoint is not this run's: $WHY"; exit 1; }
gated || gate

# ---- 3. the 50-item evaluations ---------------------------------------------------
for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
  S=${pair%%:*}; F=${pair##*:}
  [ -s $D/eval_$S.json ] && continue
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
      --data $F --arms base,psilm --n 50 --max-new 24 \
      --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $S FAILED"; exit 1; }
  step "EVAL $S: $(grep -E '^ +(base|psilm) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
done

bench() {   # $1 tag, $2 n, the rest: flags. --fresh only when the run has no rows.
  local GT=$1 N=$2 FLAG=--fresh i; shift 2
  [ -s results/bench/${GT}_guardrail_summary.json ] && return 0
  [ -s results/bench/${GT}_guardrail.rows.jsonl ] && FLAG=--resume
  for i in 1 2 3 4 5 6; do
    $PY eval/bench_guardrail.py --tag $GT --n $N $FLAG --model $M --hf-tokenizer $M \
        --bridge-kind constitution --ckpt $D/bridges.npz --arms base,psilm \
        --kl --seed 0 --print-every 10 --save-every 10 "$@" >> results/bench/${GT}_run.log 2>&1 \
      && { $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1 \
             && [ -s results/bench/${GT}_guardrail_summary.json ] && return 0
           step "SUMMARY FAILED ($GT)"; return 1; }
    [ $i = 6 ] && break
    step "BENCH $GT attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  return 1
}

# ---- 4. the 400 red-team prompts: the comparison this run exists for ---------------
NP=const_${T}_rt400
RP=const_qwen35_all_rt400_repro
GT=const_qwen35_all_rt400_fixed
REC=const_qwen35_all_rt400
R=$(jget results/qwen35/reproduction.json reproduction.ok)
rows_of() { grep -c "\"arm\": \"$2\"" results/bench/${1}_guardrail.rows.jsonl 2>/dev/null; }
# the base arm: the recorded run's if today's harness regenerates it, else the one
# the fixed-token run generated today, else generated here
if [ ! -s results/bench/${NP}_guardrail.rows.jsonl ]; then
  FROM=""
  if [ "$R" = "True" ] && [ "$(rows_of $RP base)" = "400" ]; then FROM=$RP
  elif [ "$(rows_of $GT base)" = "400" ]; then FROM=$GT; fi
  if [ -n "$FROM" ]; then
    $PY eval/bench_seed_rows.py --from-tag $FROM --arms base --tasks-cache $RTCACHE --out-tag $NP >> $LOG 2>&1 \
      && step "the base arm of the 400 is borrowed from $FROM" || step "SEED FAILED: generating the base arm"
  else
    step "no base arm to borrow (reproduction ${R:-missing}): generating it"
  fi
fi
bench $NP 400 --tasks-cache $RTCACHE --redteam-data $RT --datasets redteam --max-new-redteam 128 \
  || { step "RT400 FAILED"; exit 1; }
# the reference: the partner's psilm arm, on a harness shown to be today's
PS=""
if [ "$(rows_of $GT psilm)" = "400" ]; then PS=$GT
elif [ "$R" = "True" ]; then PS=$REC; fi
if [ -n "$PS" ]; then
  $PY eval/const_arm_compare.py --base $NP:base --ref $PS:psilm --arms $NP:psilm \
      --identity-reference $REF/compress.rows.jsonl \
      --out results/constitution/nopartner_rt400.json > results/qwen35/nopartner_report.txt 2>&1
  if [ $? = 0 ]; then
    while IFS= read -r line; do step "RT400 $line"; done < results/qwen35/nopartner_report.txt
  else
    step "RT400 REPORT FAILED (results/qwen35/nopartner_report.txt)"
  fi
else
  step "RT400 NO VALIDATED REFERENCE (reproduction ${R:-missing}, no psilm arm generated today): the paired comparison is not made"
fi
step "RT400 COMPLETE"

# ---- 5. the guard-rail of the recipe -------------------------------------------------
bench const_${T} 100 --tasks-cache results/bench/tasks_const_qwen35_n100.json --redteam-data $DATA_TE \
    --datasets redteam,gsm8k,mmlu,boolq --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 \
    --max-new-redteam 128 --gsm8k-nudge 1 || { step "GUARDRAIL FAILED"; exit 1; }
step "GUARDRAIL COMPLETE: $(grep -A5 '^dataset ' results/bench/const_${T}_run.log | tail -4 | tr -s ' ' | tr '\n' '|')"

# ---- 6. the teacher ceiling, temperature-controlled ---------------------------------
if ! grep -q temperature_crossfit $D/teacher_ceiling.json 2>/dev/null; then
  $PY eval/teacher_ceiling.py --ckpt $D/bridges.npz --data $DATA_TE,$DATA_HE --n 50 --fresh \
      > $D/teacher_ceiling.log 2>&1 || step "CEILING exited non-zero ($D/teacher_ceiling.log)"
fi
trap - EXIT
rm -rf $LOCK
step "PARTNERFREE COMPLETE"
