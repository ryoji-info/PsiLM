#!/bin/bash
# The temperature control of the PsiLM-2 paper, on the nine 0.5B bridges.
#
# It was run on the 9B bridges only ("What the channel reads, and what its fit
# measures"); three 0.5B comparisons "have not been checked against it". How each
# is read was written before this ran:
#   results/constitution/tempcontrol_qwen0.5b_preregistration.json
#
#   1. eval/teacher_ceiling.py on each bridge, both splits, the evaluator's 100 items.
#      Every arm is measured; one whose checks fail is reported as not ok and nothing
#      is read from it. Only a crash stops the chain.
#   2. results/constitution/arms_controls.py --set qwen0.5b   the table
#   3. eval/tempcontrol_reading.py                             the rules applied
#
# The rules as applied are ONE commit: the pre-registration, the three programs
# and this chain. The chain refuses to start, to measure an arm, and to build the
# table or the reading, unless each is as that commit has it, what the measurement
# imports from this repository is too, and the running copy is this file as
# committed. An arm begun under another commit is kept or resumed only if the
# measurement and its imports hold the same content in both
# (results/stage2c_qwen0.5b_<arm>/teacher_ceiling.commit). Otherwise, and when an
# arm has rows or a result and no such record, the chain stops; the arm is
# measured again only after its rows and its result are removed.
#
# LAUNCH A COPY, NOT THIS FILE. Waits for the GPU lock the chains share
# (results/qwen35/.gpu.lock). Writes results/constitution/tempcontrol_qwen0.5b.log
# and ends with TEMPCONTROL COMPLETE; any other exit writes TEMPCONTROL ENDED.
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
ARMS="all allplain vn plainpartner vn5 magmatch magmatch1 magmatch2 rand"
TE=data/constitution_test_qwen0.5b.json
HE=data/constitution_helpful_test_qwen0.5b.json
LOG=results/constitution/tempcontrol_qwen0.5b.log
CHAIN=results/constitution/tempcontrol_qwen0.5b.sh
RULES="results/constitution/tempcontrol_qwen0.5b_preregistration.json eval/teacher_ceiling.py results/constitution/arms_controls.py eval/tempcontrol_reading.py $CHAIN"
# what eval/teacher_ceiling.py imports from this repository
USES="eval/bench_common.py eval/build_constitution_data.py psilm/__init__.py psilm/mlx/__init__.py psilm/mlx/constitution.py psilm/mlx/bridges.py psilm/mlx/model.py psilm/mlx/staged.py psilm/mlx/gemma_loader.py psilm/mlx/qwen35_loader.py psilm/mlx/bonsai_loader.py"
SAME="git -c core.fileMode=false diff --quiet"     # against the working tree: a changed mode alone is not a changed file
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
LOCK=results/qwen35/.gpu.lock
HELD=""
bye() { local rc=$?; [ -n "$HELD" ] && rm -rf $LOCK; step "TEMPCONTROL ENDED rc=$rc"; }
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
AT=$(git rev-parse HEAD)
frozen() {   # every rule file and import is as commit $AT has it, and the running copy is the committed chain
  local f
  for f in $RULES $USES; do
    git cat-file -e "$AT:$f" 2>/dev/null && $SAME "$AT" -- "$f" \
      || { step "PREREQ $f is not committed as it is here"; return 1; }
  done
  git show "$AT:$CHAIN" | cmp -s - "$SELF" \
    || { step "PREREQ the running copy $SELF is not the committed chain"; return 1; }
}
same_measurement() {   # $1 is a full commit id, and the measurement and its imports hold there what they hold at $AT
  local f                # (blob ids: the content, not the mode)
  [ -n "$1" ] && [ "$(git rev-parse -q --verify "$1^{commit}" 2>/dev/null)" = "$1" ] || return 1
  for f in eval/teacher_ceiling.py $USES; do
    [ "$(git rev-parse -q --verify "$1:$f" 2>/dev/null)" = "$(git rev-parse -q --verify "$AT:$f" 2>/dev/null)" ] || return 1
  done
}
# ok / notok: a complete result (both splits, 100 items each) that says so; anything else: partial
DONE='import json,sys
d=json.load(open(sys.argv[1])); s=d["splits"]
whole=len(s)==2 and all(v["n"]==100 and "temperature_crossfit" in v["all"] for v in s.values())
print(("ok" if d.get("ok") is True else "notok" if d.get("ok") is False else "partial") if whole else "partial")'
state() { $PY -c "$DONE" "$1" 2>/dev/null || echo partial; }

step "TEMPCONTROL WAITING for the GPU"
gpu_wait
[ -s $TE ] && [ -s $HE ] && [ -s data/constitution_qwen0.5b_stats.json ] && [ -s data/constitution/system_excerpt.md ] \
  || { step "PREREQ files missing"; exit 1; }
for a in $ARMS; do
  [ -s results/stage2c_qwen0.5b_$a/bridges.npz ] && [ -s results/stage2c_qwen0.5b_$a/bridges.npz.meta ] \
    || { step "PREREQ no checkpoint for $a"; exit 1; }
done
frozen || exit 1
step "the rules as applied are commit $(git log -1 --format='%h %cI' $AT)"
for t in eval/teacher_ceiling.py eval/tempcontrol_reading.py; do
  $PY $t --self-test >> results/constitution/tempcontrol_qwen0.5b_selftest.log 2>&1 \
    || { step "SELF-TEST FAILED: $t (results/constitution/tempcontrol_qwen0.5b_selftest.log)"; exit 1; }
done
step "TEMPCONTROL START"

for a in $ARMS; do
  frozen || exit 1                    # no arm is measured unless the rules are as at $AT
  D=results/stage2c_qwen0.5b_$a
  # an arm begun under another commit is resumed or kept only if the measurement is the same
  if [ -e $D/teacher_ceiling.rows.jsonl ] || [ -e $D/teacher_ceiling.json ]; then
    B=$(cat $D/teacher_ceiling.commit 2>/dev/null)
    same_measurement "$B" \
      || { step "PREREQ $a was begun under another measurement (${B:-no record of its commit}): remove its rows and result to measure it again"; exit 1; }
  else
    echo "$AT" > $D/teacher_ceiling.commit
  fi
  S=$(state $D/teacher_ceiling.json)
  if [ "$S" = partial ]; then
    rm -f $D/teacher_ceiling.json
    $PY eval/teacher_ceiling.py --ckpt $D/bridges.npz --data $TE,$HE --n 100 > $D/teacher_ceiling.log 2>&1; rc=$?
    S=$(state $D/teacher_ceiling.json)
    # the script exits 0 with ok true and 1 with ok false (its checks failed); anything else is a crash
    { [ $rc = 0 ] && [ "$S" = ok ]; } || { [ $rc = 1 ] && [ "$S" = notok ]; } \
      || { mv -f $D/teacher_ceiling.json $D/teacher_ceiling.failed.json 2>/dev/null
           step "CEILING FAILED: $a rc=$rc state=$S ($D/teacher_ceiling.log)"; exit 1; }
  fi
  [ "$S" = notok ] && step "CEILING NOT OK: $a (reported as not ok; nothing is read from it)"
  step "CEILING $a: $($PY -c "
import json
d=json.load(open('$D/teacher_ceiling.json'))
print('ok', d['ok'], '|', ' | '.join('%s gain %.4f, at best temperatures %+.4f' % (s.replace('constitution_','').replace('_qwen0.5b',''), v['all']['ce_gain'], v['all']['temperature_crossfit']['ce_gain_at_best_tau']) for s, v in d['splits'].items()))")"
done

frozen || exit 1
$PY results/constitution/arms_controls.py --set qwen0.5b > results/constitution/arms_controls_qwen0.5b.txt 2>&1 \
  || { step "TABLE FAILED (results/constitution/arms_controls_qwen0.5b.txt)"; exit 1; }
frozen || exit 1
$PY eval/tempcontrol_reading.py > results/constitution/tempcontrol_qwen0.5b_reading.txt 2>&1 \
  || { step "READING FAILED (results/constitution/tempcontrol_qwen0.5b_reading.txt)"; exit 1; }
while IFS= read -r line; do step "READING $line"; done < results/constitution/tempcontrol_qwen0.5b_reading.txt
trap - EXIT
rm -rf $LOCK
step "TEMPCONTROL COMPLETE"
