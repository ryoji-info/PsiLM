#!/bin/bash
# Does the 9B full-width bridge's generated behaviour survive taking away what it reads?
#
# eval/constitution_compress.py found, teacher-forced, that the bridge writes the
# same thing whatever prompt it read (KL 0.0003 with another prompt's tokens, at
# the half-precision noise floor, against 0.065 for the write itself). This runs
# the 400 red-team prompts the paper's behavioural result stands on with two more
# arms: the same bridges, gate and write, fed ONE stored set of tokens --
#   fixed       the mean over the 100 validation prompts (none is among the 400)
#   fixedzero   the tokens of a partner fed zeros: made from no prompt at all
# In both, the forward bridge, the partner and the reverse bridge are not evaluated.
#
# The base and psilm arms are the recorded run's (const_qwen35_all_rt400). Before
# they are borrowed, the first 40 prompts are generated again, both arms, and must
# be the recorded ones token for token; if they are not, nothing is borrowed and
# all four arms are generated.
#
# The verdict is on aggregate behaviour against base, keyword here and blind
# adjudication afterwards, by criteria fixed in advance
# (results/constitution/partnerfree_preregistration.json) -- NOT on how many
# generations are identical to the psilm arm's, which greedy decoding makes small
# for any change at all.
#
# LAUNCH A COPY NAMED fixed_rt400_run.sh, NOT THIS FILE (the partner-free chain
# looks for that name). Waits for the arms run (ARMS COMPLETE in
# results/qwen35/arms_controls.log), then for the GPU lock it shares with the
# partner-free chain (results/qwen35/.gpu.lock), so a relaunch never starts a
# second 9B job beside a running one. Ends with FIXED-RT400 COMPLETE; any other
# exit writes FIXED-RT400 ENDED.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
D=results/stage2c_qwen35_all
RT=data/redteam_qwen35_n400.json
CACHE=results/bench/tasks_rt400_qwen35_n400.json
REC=const_qwen35_all_rt400
RP=const_qwen35_all_rt400_repro
GT=const_qwen35_all_rt400_fixed
LOG=results/qwen35/fixed_rt400.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
jget() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): d=d[k]
print(d)" "$1" "$2" 2>/dev/null; }
LOCK=results/qwen35/.gpu.lock
HELD=""
bye() { local rc=$?; [ -n "$HELD" ] && rm -rf $LOCK; step "FIXED-RT400 ENDED rc=$rc"; }
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

step "FIXED-RT400 WAITING for the arms run"
MISS=0
until grep -q "ARMS COMPLETE" results/qwen35/arms_controls.log 2>/dev/null; do
  if pgrep -f '^(/bin/)?bash [^ ]*arms_run\.sh' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "the arms run is neither running nor complete; proceeding"; break; }
  sleep 120
done
gpu_wait
[ -s $D/bridges.npz ] && [ -s $RT ] && [ -s $CACHE ] && [ -s data/constitution_val_qwen35.json ] \
  && [ -s results/bench/${REC}_guardrail.rows.jsonl ] && [ -s results/bench/${REC}_klpool_guardrail.rows.jsonl ] \
  && [ -s $D/compress.rows.jsonl ] || { step "PREREQ files missing"; exit 1; }
step "FIXED-RT400 START"

for t in constitution_fixed_tokens const_arm_compare bench_seed_rows; do
  $PY eval/$t.py --self-test >> results/qwen35/fixed_rt400_selftest.log 2>&1 \
    || { step "SELF-TEST FAILED: eval/$t.py (results/qwen35/fixed_rt400_selftest.log)"; exit 1; }
done

bench() {   # $1 tag, $2 arms, the rest: flags. --fresh only when the run has no rows.
  local GT=$1 ARMS=$2 FLAG=--fresh i; shift 2
  [ -s results/bench/${GT}_guardrail.rows.jsonl ] && FLAG=--resume
  for i in 1 2 3 4 5 6; do
    $PY eval/bench_guardrail.py --tag $GT --n 400 $FLAG \
        --tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
        --ckpt $D/bridges.npz --const-model $CM --redteam-data $RT \
        --datasets redteam --arms $ARMS --max-new-redteam 128 --kl --seed 0 \
        --print-every 10 --save-every 10 "$@" >> results/bench/${GT}_run.log 2>&1 && return 0
    [ $i = 6 ] && break
    step "RUN $GT attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  return 1
}

# ---- 1. the two stored sets of tokens -----------------------------------------------
if [ ! -s $D/fixed_tokens_mean.json ] || [ ! -s $D/fixed_tokens_softzero.json ]; then
  # never under rows that already injected a set: a rebuilt file could differ from it
  grep -q '"fixed_tokens"' results/bench/${GT}_guardrail.rows.jsonl 2>/dev/null \
    && { step "FIXED-RT400 STOPPED: a token set is missing but $GT already holds fixed rows; a person decides"; exit 1; }
  $PY eval/constitution_fixed_tokens.py --ckpt $D/bridges.npz --data data/constitution_val_qwen35.json \
      --n 100 --kind mean,softzero --contrast-data data/constitution_helpful_test_qwen35.json \
      > $D/fixed_tokens.log 2>&1 || { step "TOKENS FAILED ($D/fixed_tokens.log)"; exit 1; }
fi
TS=$($PY -c "import json;d=json.load(open('$D/fixed_tokens_mean.json'))['spread_of_the_prompts_own_tokens'];print(json.dumps(d))")
step "TOKENS: the prompts' own tokens against their mean: $TS"

# ---- 2. does today's harness regenerate the recorded run? the first 40 prompts ----------
if [ ! -s results/qwen35/reproduction.json ]; then
  if [ ! -s results/bench/${RP}_guardrail.rows.jsonl ]; then
    $PY eval/bench_seed_rows.py --from-tag $REC --arms base,psilm \
        --klpool-rows results/bench/${REC}_klpool_guardrail.rows.jsonl \
        --tasks-cache $CACHE --skip-first 40 --out-tag $RP >> $LOG 2>&1 || { step "SEED FAILED"; exit 1; }
  fi
  bench $RP base,psilm || { step "REPRODUCTION RUN GAVE UP"; exit 1; }
  $PY eval/const_arm_compare.py --base $RP:base --ref $RP:psilm --arms $RP:psilm --repro $RP,$REC \
      --repro-klpool results/bench/${REC}_klpool_guardrail.rows.jsonl \
      --out results/qwen35/reproduction.json > results/qwen35/reproduction.txt 2>&1 \
    || { step "REPRODUCTION REPORT FAILED (results/qwen35/reproduction.txt)"; exit 1; }
fi
REPRO=$(jget results/qwen35/reproduction.json reproduction.ok)
RKL=$(jget results/qwen35/reproduction.json reproduction.kl_ok)
RB=$(jget results/qwen35/reproduction.json reproduction.base)
RPS=$(jget results/qwen35/reproduction.json reproduction.psilm)
step "REPRODUCTION generations ok=$REPRO, KLs on the corrected read ok=$RKL: base $RB psilm $RPS"
[ "$RKL" = "True" ] || step "NOTE: today's psilm KLs are not the rescored ones: compare KLs within this run only"

# ---- 3. the 400 prompts ---------------------------------------------------------------
if [ ! -s results/bench/${GT}_guardrail_summary.json ]; then
  if [ ! -s results/bench/${GT}_guardrail.rows.jsonl ]; then
    if [ "$REPRO" = "True" ]; then
      cp results/bench/${RP}_guardrail.rows.jsonl results/bench/${GT}_guardrail.rows.jsonl
      step "base and psilm are the recorded run's; generating the two fixed arms"
    else
      step "the recorded run is NOT regenerated token for token: generating all four arms"
    fi
  fi
  bench $GT base,psilm,fixed,fixedzero \
      --fixed-tokens fixed=$D/fixed_tokens_mean.npz,fixedzero=$D/fixed_tokens_softzero.npz \
    || { step "RUN GAVE UP"; exit 1; }
  $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1 \
    && [ -s results/bench/${GT}_guardrail_summary.json ] || { step "SUMMARY FAILED ($GT)"; exit 1; }
  step "RUN COMPLETE"
fi

# ---- 4. the comparison, on the keyword instrument ---------------------------------------
$PY eval/const_arm_compare.py --base $GT:base --ref $GT:psilm --arms $GT:fixed,$GT:fixedzero \
    --repro $RP,$REC --repro-klpool results/bench/${REC}_klpool_guardrail.rows.jsonl \
    --identity-reference $D/compress.rows.jsonl \
    --out results/constitution/fixed_tokens_rt400.json > results/qwen35/fixed_tokens_report.txt 2>&1 \
  || { step "REPORT FAILED (results/qwen35/fixed_tokens_report.txt)"; exit 1; }
while IFS= read -r line; do step "REPORT $line"; done < results/qwen35/fixed_tokens_report.txt
trap - EXIT
rm -rf $LOCK
step "FIXED-RT400 COMPLETE"
