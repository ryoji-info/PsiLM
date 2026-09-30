#!/bin/bash
# A physics bridge on Ternary Bonsai 2 27B, for the PsiLM chat app's Physics switch.
#
# The recipe is the Qwen3.5 9B physics bridge's (results/qwen35/phaseA_recipe.sh and
# coupled_recipe.sh, eval/mlx_stage2_train.py): a readout warm-up of 2,000 steps at
# batch 8, 8,000 coupled steps at batch 2, then 1,500 selective-gate steps in which
# every second step is a batch of negatives that may move the gate only; the value
# channel, 8 soft tokens, injection capped at 0.2, gate bias 0. Then the sixty-item
# held-out evaluation and the guard-rail. What differs from the 9B, the stops and the
# criteria were written down before any step ran: results/bonsai/physics_preregistration.json.
#
# Depths: read at 26 and write at 48 of 64, the depths of the Bonsai constitution
# bridge (results/bonsai/constitution_bonsai.sh) and the window results/bonsai/precheck.json
# checked. The 16 layers above the write are recomputed in the backward pass
# (--checkpoint-from -1), so they do not set the memory; the longest negative batch does.
#
# Nothing is inherited from the 9B but its prompts: the questions are tokenized at run
# time (precheck_physics.py checks that Bonsai's tokenizer gives the 9B's ids), and the
# negatives are Bonsai's own continuations (data/noharm_bonsai27b_all.json, built for the
# constitution bridge from the 9B's 1,194 prompts).
#
# Chunks are 500 steps, as on the 9B, with the checkpoint also written every 100 steps
# inside a chunk, so a crash costs at most 100 steps. A chunk whose log stops growing
# for two hours is killed and resumed. Smoke runs go to their own directories: the run
# directory's supervisor.log must hold this run's coupling line only (the guard-rail and
# the chat app read the depths from its last one).
#
# LAUNCH A COPY, NOT THIS FILE (bash reads a script as it runs it). One GPU job at a
# time: the chain takes results/qwen35/.gpu.lock and waits for any trainer, evaluator or
# chat server to go. Relaunching resumes: smokes are not repeated once training has
# started, and every stage skips what is already on disk. Ends with
# BONSAI-PHYSICS COMPLETE, or with a BONSAI-PHYSICS STOPPED line that says why.
cd /Users/rxiii/Documents/GitHub/PsiLM || exit 1
PY=.venv/bin/python
P=/Users/rxiii/Documents/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787
TAG=_bonsai27b
D=results/stage2$TAG
META=$D/bridges.npz.meta
LOG=results/bonsai/physics_bonsai.log
NEG=data/noharm_bonsai27b_all.json
LONGNEG=results/bonsai/smoke_long_noharm.json
CACHE=results/bench/tasks_qwen35_n100.json
GT=guardrail_bonsai27b
READ=26
WRITE=48
A_END=2000            # readout warm-up
C_END=10000           # + 8,000 coupled steps
N_END=11500           # + 1,500 selective-gate steps
CHUNK=500
SAVE=100
PEAK_MAX=${PEAK_MAX:-50}
TRAIN_H_MAX=${TRAIN_H_MAX:-96}
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/Users/rxiii/Documents/huggingface
mkdir -p $D results/bench
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
stop() { step "BONSAI-PHYSICS STOPPED: $1"; exit 1; }
jget() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): d=d[k]
print(d)" "$1" "$2" 2>/dev/null; }
le() { $PY -c "import sys;sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$2"; }

LOCK=results/qwen35/.gpu.lock
HELD=""
CHILD=""
bye() { local rc=$?; [ -n "$CHILD" ] && kill $CHILD 2>/dev/null; [ -n "$HELD" ] && rm -rf $LOCK; step "BONSAI-PHYSICS ENDED rc=$rc"; }
trap bye EXIT
trap 'exit 143' TERM; trap 'exit 129' HUP; trap 'exit 130' INT
# anchored on the interpreter: a monitor or an editor that merely NAMES a program is not a GPU job
GPU='^[^ ]*[Pp]ython[0-9.]* [^ ]*(mlx_stage2[bd]?_(train|eval)|mlx_constitution_(train|eval)|bench_guardrail|build_noharm|precheck(_physics)?|stage2_train_selftest|affect_collect|teacher_ceiling|constitution_compress|constitution_fixed_tokens)\.py'
CHAT='^[^ ]*[Pp]ython[0-9.]* .*(psilm_chat[./](server|cli)|[ /]psilm-chat(-cli)?( |$))'
gpu_wait() {   # until no GPU program runs, no chat server holds a model, and the lock is ours
  local w=0 o
  while :; do
    if ! pgrep -f "$GPU" > /dev/null && ! pgrep -f "$CHAT" > /dev/null; then
      if mkdir $LOCK 2>/dev/null; then echo $$ > $LOCK/pid; HELD=1; return 0; fi
      o=$(cat $LOCK/pid 2>/dev/null)
      [ -n "$o" ] && ! kill -0 "$o" 2>/dev/null && { step "stale GPU lock of $o removed"; rm -rf $LOCK; continue; }
    fi
    w=$((w + 1))
    [ $((w % 10)) = 1 ] && step "GPU BUSY: $( (pgrep -fl "$GPU"; pgrep -fl "$CHAT") | head -1 | cut -c1-110) lock $(cat $LOCK/pid 2>/dev/null) (a running chat app holds the model: quit it)"
    sleep 60
  done
}

# a chat app opened while the chain runs would share the 24 GB with the next job: wait for it to go
# (the lock stays ours; an app opened in the middle of a job still collides with it)
chat_wait() {
  pgrep -f "$CHAT" > /dev/null || return 0
  step "CHAT APP OPEN: waiting for it to quit"
  while pgrep -f "$CHAT" > /dev/null; do sleep 60; done
  step "CHAT APP CLOSED"
}

# run "$@" with its output appended to $2, and kill it when $2 has not grown for $1 minutes
# (a stalled GPU prints nothing; macOS has no `timeout`). Returns the command's status, 124 when killed.
watched() {
  local STALL=$1 OUT=$2 W RC NOW MT; shift 2
  "$@" >> $OUT 2>&1 &
  CHILD=$!
  while kill -0 $CHILD 2>/dev/null; do
    sleep 60
    NOW=$(date +%s); MT=$(stat -f %m $OUT 2>/dev/null || echo $NOW)
    if [ $(( (NOW - MT) / 60 )) -ge $STALL ]; then
      kill $CHILD 2>/dev/null; sleep 15; kill -9 $CHILD 2>/dev/null
      wait $CHILD 2>/dev/null; CHILD=""
      step "STALLED: nothing written to $OUT for $STALL min; killed"
      return 124
    fi
  done
  wait $CHILD; RC=$?; CHILD=""
  return $RC
}

step "BONSAI-PHYSICS WAITING for the GPU"
gpu_wait
[ -s $P/model.safetensors ] && [ -s data/stage2_qa_train.json ] && [ -s data/stage2_qa_val.json ] && [ -s $NEG ] \
  && [ -s $LONGNEG ] && [ -s results/stage2/fno.pt ] && [ -s $CACHE ] && [ -s results/bonsai/precheck.json ] \
  && [ -s results/bonsai/physics_preregistration.json ] || stop "prerequisite files missing"
step "BONSAI-PHYSICS START (read $READ, write $WRITE of 64)"

# ---- 1. the checks that must hold before a training step ------------------------
if ! grep -q "^SELF-TESTS OK" $LOG; then
  ST=results/bonsai/physics_selftest.log
  : > $ST
  $PY eval/stage2_train_selftest.py >> $ST 2>&1 || stop "self-test failed: eval/stage2_train_selftest.py ($ST)"
  $PY results/bonsai/physics_verdict.py --self-test >> $ST 2>&1 || stop "self-test failed: physics_verdict.py ($ST)"
  $PY eval/bench_guardrail.py --tag selftest --self-test >> $ST 2>&1 || stop "self-test failed: bench_guardrail.py ($ST)"
  $PY eval/checkpoint_selftest.py >> $ST 2>&1 || stop "self-test failed: checkpoint_selftest.py ($ST)"
  step "SELF-TESTS OK"
fi
PC=results/bonsai/precheck_physics.json
pc_ok() { [ "$(jget $PC ok)" = "True" ] && [ "$(jget $PC stream.ok)" = "True" ] && [ ! $NEG -nt $PC ]; }
if ! pc_ok; then
  $PY results/bonsai/precheck_physics.py > results/bonsai/precheck_physics.log 2>&1
  pc_ok || stop "precheck failed ($PC, results/bonsai/precheck_physics.log); a person decides"
  step "PRECHECK OK: $($PY -c "import json;d=json.load(open('$PC'));print('qa', d['qa']['train']['same_ids_and_labels_as_9b'], d['qa']['val']['same_ids_and_labels_as_9b'], 'fno', d['fno']['max_abs_difference'], 'stream', [round(r['rel'], 6) for k, v in d['stream'].items() if isinstance(v, list) for r in v])")"
fi
# the guard-rail's task cache must restore under this tokenizer and these flags: found now, not after training
G="--tasks-cache $CACHE --model $P --hf-tokenizer $P --ckpt $D/bridges.npz --fno results/stage2/fno.pt \
  --l-fwd $READ --l-rev $WRITE --gate-bias 0.0 --datasets physics,mmlu,gsm8k,boolq --arms base,psilm,zeroed \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-physics 32 --max-new-boolq 16 \
  --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
if ! grep -q "^TASK CACHE OK" $LOG; then
  $PY eval/bench_guardrail.py --tag ${GT}_dry --n 100 $G --dry-run > results/bonsai/physics_guardrail_dry.log 2>&1 \
    && grep -q "^\[tasks\] 400 restored" results/bonsai/physics_guardrail_dry.log \
    || stop "the guard-rail's task cache does not restore for this run (results/bonsai/physics_guardrail_dry.log)"
  rm -f results/bench/${GT}_dry_guardrail_dryrun.json
  step "TASK CACHE OK: 400 tasks restored from $CACHE"
fi

C="--model $P --hf-tokenizer $P --l-fwd $READ --l-rev $WRITE --channel value --inj-cap 0.2 --gate-bias 0.0 \
  --lam-x0 1.0 --clip module --detach-x0 --readout-norm dim --calib-n 32 --checkpoint-from -1"
NH="--noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0"

# ---- 2. smokes: memory on the longest negative batch, and the time of each phase ---
peak_of() { grep 'CHUNK DONE' results/stage2${TAG}_smoke$1/supervisor.log 2>/dev/null | tail -1 | sed -E 's/.*peak=([0-9.]+)GB.*/\1/'; }
sps_of()  { $PY -c "import json;r=[json.loads(l) for l in open('results/stage2${TAG}_smoke$1/train_log.jsonl') if 'sec_per_step' in l];print(r[-1]['sec_per_step'] if r else '')" 2>/dev/null; }
smoke() {   # $1 name, then the trainer flags; a smoke that ended (well or badly) is not run again
  local NAME=$1 SD=results/stage2${TAG}_smoke$1 T0 RC; shift
  if [ -n "$(peak_of $NAME)" ] || grep -qE "^SMOKE $NAME (FAILED|STALLED)" $LOG; then return 0; fi
  rm -rf $SD; mkdir -p $SD
  chat_wait
  T0=$(date +%s)
  watched 60 $SD/supervisor.log $PY eval/mlx_stage2_train.py --tag ${TAG}_smoke$NAME --fresh --init-seed 0 $C "$@"
  RC=$?
  echo $(( $(date +%s) - T0 )) > $SD/wall_sec
  rm -f $SD/bridges.npz $SD/opt.npz           # 0.8 GB a smoke; the logs are what is kept
  if [ $RC = 124 ]; then step "SMOKE $NAME STALLED"; elif [ -z "$(peak_of $NAME)" ]; then step "SMOKE $NAME FAILED (exit $RC; $SD/supervisor.log)"; fi
}
if [ ! -s $META ]; then
  smoke long --batch 2 --steps 2 --lr 1e-4 --readout-only 0 --eval-n 1 --noharm-data $LONGNEG --noharm-every 2 --noharm-gate-only 1 --lam-gate 1.0
  PK=$(peak_of long)
  [ -n "$PK" ] || stop "the memory smoke (longest negative batch, 513 tokens at batch 2) did not finish (results/stage2${TAG}_smokelong/supervisor.log); a person decides"
  grep -q "^SMOKE long:" $LOG || step "SMOKE long: batch 2, write $WRITE, longest negative batch: peak $PK GB"
  le "$PK" "$PEAK_MAX" || stop "the longest negative batch peaks at $PK GB (> $PEAK_MAX) at batch 2; a person decides"
  smoke A --batch 8 --steps 25 --lr 3e-4 --readout-only $A_END --eval-n 1
  smoke B --batch 2 --steps 25 --lr 3e-4 --readout-only 0 --eval-n 2
  smoke N --batch 2 --steps 26 --lr 1e-4 --readout-only 0 --eval-n 1 $NH
  SA=$(sps_of A); SB=$(sps_of B); SN=$(sps_of N)
  [ -n "$SA" ] && [ -n "$SB" ] && [ -n "$SN" ] && [ -n "$(peak_of A)" ] && [ -n "$(peak_of B)" ] && [ -n "$(peak_of N)" ] \
    || stop "a timing smoke did not finish (see the SMOKE lines and results/stage2${TAG}_smoke*/supervisor.log); a person decides"
  for nm in A B N; do le "$(peak_of $nm)" "$PEAK_MAX" || stop "smoke $nm peaks at $(peak_of $nm) GB (> $PEAK_MAX); a person decides"; done
  # a chunk's rollouts: what smoke B spent beyond its 25 steps was the load, the calibration and 2 rollouts;
  # 8 rollouts a chunk is at most four times that (an upper bound: the load is in it)
  WB=$(cat results/stage2${TAG}_smokeB/wall_sec)
  HOURS=$($PY -c "
a, b, n, wb = $SA, $SB, $SN, $WB
roll = max(0.0, wb - 25 * b) * 4
t = a * $A_END + b * ($C_END - $A_END) + n * ($N_END - $C_END) + roll * (($N_END - $A_END) // $CHUNK)
print(round(t / 3600, 1))")
  grep -q "^SMOKE times:" $LOG || step "SMOKE times: warm-up $SA s/step (peak $(peak_of A) GB), coupled $SB s/step (peak $(peak_of B) GB), selective gate $SN s/step (peak $(peak_of N) GB) -> about $HOURS h of training"
  le "$HOURS" "$TRAIN_H_MAX" || stop "training projects to $HOURS h (> $TRAIN_H_MAX); a person decides"
else
  [ "$(jget $META l_fwd)" = "$READ" ] || stop "the checkpoint reads at layer $(jget $META l_fwd), not $READ"
  step "RESUMING from step $(jget $META step)"
fi

# ---- 3. train ------------------------------------------------------------------
cur() { local s; s=$(jget $META step); echo ${s:-0}; }
last() { $PY -c "
import json, math, sys
rows=[json.loads(l) for l in open('$D/train_log.jsonl')]
tr=[r for r in rows if 'loss_x0' in r]; ev=[r for r in rows if 'eval_acc' in r]
r=tr[-1] if tr else {}; e=ev[-1] if ev else {}
print('step=%d phase=%s x0_ce=%.3f exact=%.3f ans=%.3f gate=%.4f inj=%.3f s/step=%s acc=%s mae=%s' % (
    r.get('step',-1), r.get('phase','?'), r.get('loss_x0',-1), r.get('x0_exact',-1), r.get('loss_ans',-1),
    r.get('gate_ans', r.get('gate',-1)), r.get('inj_ratio_ans',-1), r.get('sec_per_step','-'), e.get('eval_acc','-'), e.get('eval_mae','-')))
sys.exit(0 if all(math.isfinite(float(r.get(k, 0))) for k in ('loss_x0','loss_ans','loss_u','loss_param')) else 3)"; }
train_to() {   # $1 target step, $2 batch, $3 lr, $4 suffix of the kept copy (- for none), then extra flags
  local TARGET=$1 B=$2 LR=$3 SUF=$4 S0 S1 NEXT FLAG T0 j; shift 4
  [ "$SUF" = "-" ] && SUF=""
  # a relaunch checks the checkpoint it resumes from too (a chunk that ended non-finite saved what it had)
  if [ "$(cur)" -gt 0 ]; then LAST=$(last) || stop "a non-finite loss at step $(cur): $LAST"; fi
  while [ "$(cur)" -lt $TARGET ]; do
    S0=$(cur); NEXT=$(( (S0 / CHUNK + 1) * CHUNK )); [ $NEXT -gt $TARGET ] && NEXT=$TARGET
    T0=$(date +%s)
    for j in 1 2 3; do
      S1=$(cur); [ "$S1" -ge $NEXT ] && break                 # saved, then exited non-zero
      FLAG=""; [ -s $META ] || FLAG="--fresh --init-seed 0"
      if [ -s $META ]; then
        [ "$(jget $META l_fwd)" = "$READ" ] && [ "$(jget $META l_rev)" = "$WRITE" ] || stop "the checkpoint's depths are not $READ/$WRITE"
      fi
      chat_wait
      watched 120 $D/supervisor.log $PY eval/mlx_stage2_train.py --tag $TAG --steps $((NEXT - S1)) --batch $B --lr $LR \
          $C --readout-only $A_END --eval-n 8 --save-every $SAVE $FLAG "$@" && break
      step "TRAIN to $NEXT attempt $j exited; resuming from step $(cur)"; sleep 30
    done
    # the checkpoint decides: a chunk that saved its last step is done, however its process ended
    [ "$(cur)" -ge $NEXT ] || stop "training failed at step $(cur) (three attempts; $D/supervisor.log)"
    LAST=$(last) || stop "a non-finite loss at step $(cur): $LAST"
    [ $NEXT -gt $A_END ] && cp $D/bridges.npz $D/bridges_step${NEXT}${SUF}.npz
    step "CHUNK $LAST $(grep 'CHUNK DONE' $D/supervisor.log | tail -1 | sed -E 's/.*(peak=[0-9.]+GB).*/\1/') [$(( ($(date +%s) - T0) / 60 )) min]"
  done
}

train_to $A_END 8 3e-4 -
if ! grep -q "^PHASE-A POINTER OK" $LOG; then
  EX=$($PY -c "
import json
tr=[json.loads(l) for l in open('$D/train_log.jsonl')]
tr=[r for r in tr if r.get('phase') == 'A' and r.get('step', 0) <= $A_END][-4:]
print(round(sum(r['x0_exact'] for r in tr) / len(tr), 4), round(sum(r['loss_x0'] for r in tr) / len(tr), 3))")
  step "PHASE-A POINTER exact ${EX% *}, x0 cross-entropy ${EX#* } (last four records; the stop is below 0.5 exact)"
  # a bare relaunch meets the same stop again; going on past it is a person's decision, said with PTR_GO_ON=1
  le 0.5 "${EX% *}" || [ "${PTR_GO_ON:-}" = 1 ] \
    || stop "the pointer did not lock in the warm-up (exact ${EX% *} < 0.5); a person decides (to go on anyway, relaunch with PTR_GO_ON=1)"
  [ "$(cur)" = "$A_END" ] && cp $D/bridges.npz $D/bridges_step${A_END}_phaseA.npz
  step "PHASE-A POINTER OK${PTR_GO_ON:+ (overridden by PTR_GO_ON)}"
fi
train_to $C_END 2 3e-4 -
grep -q "^COUPLED DONE" $LOG || step "COUPLED DONE"
train_to $N_END 2 1e-4 _noharm $NH
[ "$(cur)" = "$N_END" ] || stop "the run is at step $(cur), not $N_END"
[ "$(jget $META args.noharm_data)" = "$NEG" ] || stop "the last chunk was not a selective-gate chunk"
cp $D/bridges.npz $D/bridges_final_step${N_END}_noharm.npz
grep -q "^TRAINING COMPLETE" $LOG || step "TRAINING COMPLETE at step $N_END"

# ---- 4. the held-out evaluation (Metal scan, then the training numerics) ----------
heldout() {   # $1 output file, $2 log, then the evaluator's flags
  local OUT=$1 EL=$2 j; shift 2
  [ -s $D/$OUT ] && [ "$(jget $D/$OUT summary.step)" = "$N_END" ] && return 0
  for j in 1 2 3; do
    : > $EL
    chat_wait
    watched 240 $EL $PY eval/mlx_stage2_eval.py --model $P --hf-tokenizer $P --tag $TAG --n 60 --out $OUT "$@" \
      && [ "$(jget $D/$OUT summary.step)" = "$N_END" ] && return 0
    step "HELD-OUT ($OUT) attempt $j exited; starting it again"; sleep 30
  done
  return 1
}
heldout final_eval.json $D/final_eval.log --arms baseline,oracle,psilm --max-new 768 || stop "the held-out evaluation failed ($D/final_eval.log)"
grep -q "^HELD-OUT (Metal scan)" $LOG || step "HELD-OUT (Metal scan) $(grep '^FINAL' $D/final_eval.log | tail -1 | cut -c1-420)"
heldout final_eval_ops.json $D/final_eval_ops.log --arms psilm --ops-path || stop "the held-out evaluation on the ops scan failed ($D/final_eval_ops.log)"
grep -q "^HELD-OUT (ops scan)" $LOG || step "HELD-OUT (ops scan) $(grep '^FINAL' $D/final_eval_ops.log | tail -1 | cut -c1-300)"

# ---- 5. the guard-rail ---------------------------------------------------------------
# (the bench writes <tag>_guardrail.json; the summary the repo tracks and the verdict reads is that report
# without its rows, written by results/constitution/summarize.py --track-tags)
track() { $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; }
[ -s results/bench/${GT}_guardrail.json ] && [ "$(jget results/bench/${GT}_guardrail.json ckpt_step)" = "$N_END" ] \
  && [ "$(jget results/bench/${GT}_guardrail.json n_rows)" = "1200" ] && track
if [ ! -s results/bench/${GT}_guardrail_summary.json ] || [ "$(jget results/bench/${GT}_guardrail_summary.json ckpt_step)" != "$N_END" ]; then
  FLAG=--fresh; [ -s results/bench/${GT}_guardrail.rows.jsonl ] && grep -q "^GUARDRAIL STARTED" $LOG && FLAG=--resume
  grep -q "^GUARDRAIL STARTED" $LOG || step "GUARDRAIL STARTED"
  DONE=""
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    chat_wait
    watched 120 results/bench/${GT}_run.log $PY eval/bench_guardrail.py --tag $GT --n 100 $G $FLAG && { track; DONE=1; break; }
    step "GUARDRAIL attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  [ -n "$DONE" ] && [ -s results/bench/${GT}_guardrail_summary.json ] || stop "the guard-rail failed (results/bench/${GT}_run.log)"
  step "GUARDRAIL COMPLETE"
fi

# ---- 6. the verdict, by the criteria written before training -------------------------
$PY results/bonsai/physics_verdict.py > results/bonsai/physics_verdict.log 2>&1 || stop "the verdict could not be computed (results/bonsai/physics_verdict.log)"
step "VERDICT: $(jget results/bonsai/physics_verdict.json verdict)"
step "BONSAI-PHYSICS COMPLETE"
