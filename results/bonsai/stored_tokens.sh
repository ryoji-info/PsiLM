#!/bin/bash
# May the chat app run this bridge on ONE stored set of tokens, with no partner?
#
# On the 9B the stored mean tokens reproduced, on 400 red-team prompts adjudicated
# blind, the withholding of the full system (PsiLM-2 paper, "What the channel
# reads"); no benchmark was run on them, and nothing at all on the 27B, where only
# the teacher-forced controls say the write ignores what was read. This asks the
# question of ONE full-width bridge, named by NAME:
#
#   NAME=bonsai27b   Ternary Bonsai 2 27B   results/stage2c_bonsai27b_all
#   NAME=qwen35      Qwen3.5 9B             results/stage2c_qwen35_all
#   NAME=qwen0.5b    Qwen2.5 0.5B           results/stage2c_qwen0.5b_all
#                    (its write DOES use what it reads, about a sixth of its
#                    effect: the bridge where the stored set is expected to fail)
#
#   1. the tokens      the mean over the 100 validation prompts, and the zero-fed
#                      set (eval/constitution_fixed_tokens.py); the 9B's exist
#   2. teacher-forced  the stored sets as controls of eval/constitution_compress.py,
#                      held to the thresholds a compressed variant is held to
#   3. reproduction    the first 40 red-team prompts of the recorded guard-rail,
#                      generated again, base and psilm: token for token or not
#   4. the guard-rail  100 red-team prompts, GSM8K, MMLU and BoolQ with the arm
#                      `fixed` (the stored mean set). Base and psilm are the
#                      recorded run's rows if step 3 reproduced them, else generated.
#   5. the verdict     eval/stored_tokens_report.py, by criteria written before any
#                      of this ran (results/constitution/stored_tokens_criteria.json)
#
# LAUNCH A COPY, NOT THIS FILE, with NAME in the environment. Waits for the GPU
# lock the chains share (results/qwen35/.gpu.lock). Writes
# results/bonsai/stored_tokens_$NAME.log and ends with STORED COMPLETE; any other
# exit writes STORED ENDED.
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
NAME=${NAME:?name the bridge: NAME=bonsai27b, NAME=qwen35 or NAME=qwen0.5b}
CACHE=results/bench/tasks_const_qwen35_n100.json
RT=data/constitution_test_qwen35.json
case "$NAME" in
  qwen0.5b)  M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
             CM=results/constitution_model/qwen2.5-0.5b-constitution
             KLPOOL=results/bench/const_qwen0.5b_all_klpool_guardrail.rows.jsonl
             CACHE=results/bench/tasks_const_qwen0.5b_n100.json
             RT=data/constitution_test_qwen0.5b.json ;;
  bonsai27b) M=/Users/rxiii/Documents/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787
             CM=$PWD/results/constitution_model/qwen2.5-0.5b-constitution
             KLPOOL="" ;;
  qwen35)    M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
             CM=results/constitution_model/qwen2.5-0.5b-constitution
             KLPOOL=results/bench/const_qwen35_all_klpool_guardrail.rows.jsonl ;;
  *) echo "NAME is bonsai27b, qwen35 or qwen0.5b" >&2; exit 2 ;;
esac
D=results/stage2c_${NAME}_all
REC=const_${NAME}_all
RP=const_${NAME}_all_stored_repro
GT=const_${NAME}_all_stored
VAL=data/constitution_val_$NAME.json
TE=data/constitution_test_$NAME.json
HE=data/constitution_helpful_test_$NAME.json
NEG=data/noharm_${NAME}_all.json
LOG=results/bonsai/stored_tokens_$NAME.log
REPRO=results/bonsai/reproduction_$NAME.json
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
jget() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): d=d[k]
print(d)" "$1" "$2" 2>/dev/null; }
LOCK=results/qwen35/.gpu.lock
HELD=""
bye() { local rc=$?; [ -n "$HELD" ] && rm -rf $LOCK; step "STORED ENDED rc=$rc"; }
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

step "STORED $NAME WAITING for the GPU"
gpu_wait
[ -s $D/bridges.npz ] && [ -s $D/bridges.npz.meta ] && [ -s $VAL ] && [ -s $TE ] && [ -s $HE ] && [ -s $NEG ] \
  && [ -s $RT ] && [ -s $CACHE ] && [ -s results/bench/${REC}_guardrail.rows.jsonl ] \
  && [ -s results/constitution/stored_tokens_criteria.json ] && [ -s $CM/model.safetensors ] \
  || { step "PREREQ files missing"; exit 1; }
[ -z "$KLPOOL" ] || [ -s "$KLPOOL" ] || { step "PREREQ the rescored KLs are not at $KLPOOL"; exit 1; }
for t in constitution_fixed_tokens constitution_compress const_arm_compare bench_seed_rows stored_tokens_report; do
  $PY eval/$t.py --self-test >> results/bonsai/stored_tokens_selftest.log 2>&1 \
    || { step "SELF-TEST FAILED: eval/$t.py (results/bonsai/stored_tokens_selftest.log)"; exit 1; }
done
step "STORED START ($NAME)"

# ---- 1. the two stored sets -----------------------------------------------------------
if [ ! -s $D/fixed_tokens_mean.json ] || [ ! -s $D/fixed_tokens_softzero.json ]; then
  # never under rows that already injected a set: a rebuilt file could differ from it
  grep -q '"fixed_tokens"' results/bench/${GT}_guardrail.rows.jsonl 2>/dev/null \
    && { step "STORED STOPPED: a token set is missing but $GT already holds fixed rows; a person decides"; exit 1; }
  $PY eval/constitution_fixed_tokens.py --ckpt $D/bridges.npz --model $M --const-model $CM --data $VAL \
      --n 100 --kind mean,softzero --contrast-data $HE \
      > $D/fixed_tokens.log 2>&1 || { step "TOKENS FAILED ($D/fixed_tokens.log)"; exit 1; }
fi
TS=$($PY -c "import json;d=json.load(open('$D/fixed_tokens_mean.json'))['spread_of_the_prompts_own_tokens'];print(json.dumps(d))")
step "TOKENS: the prompts' own tokens against their mean: $TS"

# ---- 2. teacher-forced: the stored sets against the trained system --------------------------
if [ ! -s $D/stored_tokens.json ]; then
  $PY eval/constitution_compress.py --ckpt $D/bridges.npz --model $M --const-model $CM --data $TE,$HE \
      --noharm $NEG --heldout-rows results/bench/${REC}_guardrail.rows.jsonl --tasks-cache $CACHE \
      --variants fp32+native --no-real \
      --stored-tokens mean=$D/fixed_tokens_mean.npz,softzero=$D/fixed_tokens_softzero.npz \
      --out $D/stored_tokens.json > $D/stored_tokens.log 2>&1 \
    || { step "TEACHER-FORCED FAILED ($D/stored_tokens.log)"; exit 1; }
fi
step "TEACHER-FORCED: $(grep -E '^stored:' $D/stored_tokens.log | tr -s ' ' | tr '\n' '|' | cut -c1-600)"

bench() {   # $1 tag, $2 arms, the rest: flags. --fresh only when the run has no rows.
  local T=$1 ARMS=$2 FLAG=--fresh i; shift 2
  [ -s results/bench/${T}_guardrail.rows.jsonl ] && FLAG=--resume
  for i in 1 2 3 4 5 6; do
    $PY eval/bench_guardrail.py --tag $T --n 100 $FLAG \
        --tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
        --ckpt $D/bridges.npz --const-model $CM --redteam-data $RT \
        --datasets redteam,gsm8k,mmlu,boolq --arms $ARMS --max-new-gsm8k 384 --max-new-mmlu 256 \
        --max-new-boolq 16 --max-new-redteam 128 --gsm8k-nudge 1 --kl --seed 0 \
        --print-every 10 --save-every 10 "$@" >> results/bench/${T}_run.log 2>&1 && return 0
    [ $i = 6 ] && break
    step "RUN $T attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  return 1
}

# ---- 3. does today's code regenerate the recorded run? the first 40 red-team prompts ------------
if [ ! -s $REPRO ]; then
  if [ ! -s results/bench/${RP}_guardrail.rows.jsonl ]; then
    if [ -n "$KLPOOL" ]; then K="--klpool-rows $KLPOOL"; else K=""; fi
    $PY eval/bench_seed_rows.py --from-tag $REC --arms base,psilm $K \
        --tasks-cache $CACHE --skip-first 40 --out-tag $RP >> $LOG 2>&1 || { step "SEED FAILED"; exit 1; }
  fi
  bench $RP base,psilm || { step "REPRODUCTION RUN GAVE UP"; exit 1; }
  if [ -n "$KLPOOL" ]; then K="--repro-klpool $KLPOOL"; else K=""; fi
  $PY eval/const_arm_compare.py --base $RP:base --ref $RP:psilm --arms $RP:psilm --repro $RP,$REC $K \
      --out $REPRO > results/bonsai/reproduction_$NAME.txt 2>&1 \
    || { step "REPRODUCTION REPORT FAILED (results/bonsai/reproduction_$NAME.txt)"; exit 1; }
fi
R=$(jget $REPRO reproduction.ok)
step "REPRODUCTION generations ok=$R: base $(jget $REPRO reproduction.base) psilm $(jget $REPRO reproduction.psilm)"

# ---- 4. the guard-rail with the stored set ---------------------------------------------------
if [ ! -s results/bench/${GT}_guardrail_summary.json ]; then
  if [ ! -s results/bench/${GT}_guardrail.rows.jsonl ]; then
    if [ "$R" = "True" ]; then
      cp results/bench/${RP}_guardrail.rows.jsonl results/bench/${GT}_guardrail.rows.jsonl
      step "base and psilm are the recorded run's; generating the stored-token arm"
    else
      step "the recorded run is NOT regenerated token for token: generating all three arms"
    fi
  fi
  bench $GT base,psilm,fixed --fixed-tokens fixed=$D/fixed_tokens_mean.npz || { step "RUN GAVE UP"; exit 1; }
  $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1 \
    && [ -s results/bench/${GT}_guardrail_summary.json ] || { step "SUMMARY FAILED ($GT)"; exit 1; }
  step "RUN COMPLETE: $(grep -A5 '^dataset ' results/bench/${GT}_run.log | tail -4 | tr -s ' ' | tr '\n' '|')"
fi

# ---- 5. the verdict ---------------------------------------------------------------------------
$PY eval/stored_tokens_report.py --name $NAME --run-dir $D --tag $GT --recorded $REC \
    > results/bonsai/stored_tokens_$NAME.txt 2>&1 \
  || { step "REPORT FAILED (results/bonsai/stored_tokens_$NAME.txt)"; exit 1; }
while IFS= read -r line; do step "REPORT $line"; done < results/bonsai/stored_tokens_$NAME.txt
trap - EXIT
rm -rf $LOCK
step "STORED COMPLETE"
