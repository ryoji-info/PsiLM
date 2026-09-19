#!/bin/bash
# The two narrow parity arms on the 400 red-team prompts.
#
# vn1e (41 value neurons at energy parity) and vn5e (the top 5%, 205) were
# guard-railed on the 100-item set, where the keyword count did not move (vn1e
# 2:2). The wide write's withholding was invisible at n=100 too (adjudicated
# 1:1) and appeared only on the 300 prompts added later (21:2 at n=400), so the
# question the paper asks of a narrow write -- can it carry the disposition? --
# has to be asked on the same 400 prompts. This runs the same guard-rail
# redteam400.sh ran for all / vn10e / vn10ebot: the shared 400-item cache, arms
# base/psilm/zeroed, 128 generated tokens, KL to base, seed 0. Adjudication is a
# separate judge pass on the rows.
#
# Queue (reordered 2026-09-19, after vn1e landed): parity arms (PARITY
# COMPLETE) -> THIS -> parity_controls.sh (match41, match205) -> allplain.sh ->
# plainpartner_widths.sh. No training here: two checkpoints, two guard-rails,
# about 3.5 h each.
#
# LAUNCH A COPY, NOT THIS FILE (bash reads scripts lazily; editing a running
# script resumes it mid-file). Ends with "RT400-NARROW COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
TAG=qwen35
RT=data/redteam_${TAG}_n400.json
CACHE=results/bench/tasks_rt400_${TAG}_n400.json
LOG=results/qwen35/rt400_narrow.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "RT400-NARROW WAITING for the parity arms (vn1e, vn5e)"
MISS=0
until grep -q "PARITY COMPLETE" results/qwen35/parity_widths.log 2>/dev/null; do
  # Count the whole upstream queue as alive, not just the predecessor, or a
  # legitimate gap between chains looks like a crash.
  if pgrep -f 'parity_run.sh|phys_run.sh|contentless_run.sh|rt400_run.sh' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "nothing upstream is running or complete; proceeding"; break; }
  sleep 60
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py|psilm2/(train_dual|bench)\.py' > /dev/null; do sleep 60; done

[ -s $RT ] && [ -s $CACHE ] && [ -s $CM/model.safetensors ] || { step "PREREQ files missing ($RT, $CACHE, partner)"; exit 1; }

guardrail400() {   # $1 variant name -- the recorded step-1000 checkpoint on the 400-item cache
  local V=$1 D=results/stage2c_${TAG}_${1} GT=const_${TAG}_${1}_rt400 FLAG=--fresh
  [ -s results/bench/${GT}_guardrail_summary.json ] && { step "GUARDRAIL $V rt400 already done"; return 0; }
  # A relaunch mid-run resumes the partial rows rather than discarding 3-5 h of
  # them (safe here: nothing trains, and the meta check below pins step 1000).
  [ -s results/bench/${GT}_guardrail.rows.jsonl ] && FLAG=--resume
  [ -s $D/bridges.npz ] && [ -s $D/bridges.npz.meta ] || { step "GUARDRAIL $V rt400: no checkpoint in $D"; return 1; }
  # The checkpoint must be the finished parity arm: step 1000, and the cap the
  # parity_widths.sh chain trained (anything else is not the arm being asked).
  local CHK; CHK=$($PY - "$D/bridges.npz.meta" "$V" 2>&1 <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1])); want = {"vn1e": (41, 1.0252), "vn5e": (205, 0.6257)}[sys.argv[2]]
w, cap, step = m.get("write_dims") or [], float(m["inj_cap"]), int(m.get("step", -1))
bad = []
if len(w) != want[0]: bad.append(f"dims={len(w)}")
if abs(cap - want[1]) > 1e-9: bad.append(f"cap={cap}")
if step != 1000: bad.append(f"step={step}")
print("OK" if not bad else "BAD " + " ".join(bad))
PYEOF
)
  [ "$CHK" = "OK" ] || { step "GUARDRAIL $V rt400 meta wrong: $CHK"; return 1; }
  # Byte-identical to redteam400.sh / allplain.sh: the same cache, arms, budget and seed.
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CM --redteam-data $RT \
    --datasets redteam --arms base,psilm,zeroed \
    --max-new-redteam 128 --kl --seed 0 --print-every 10 --save-every 10"
  local i
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 400 $COMMON $FLAG >> results/bench/${GT}_run.log 2>&1 \
      && { step "GUARDRAIL $V rt400 COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${GT}_run.log | tail -6 >> $LOG; \
           $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; \
           [ -s results/bench/${GT}_guardrail_summary.json ] || step "GUARDRAIL $V rt400 tracked summary missing after summarize.py"; return 0; }
    step "GUARDRAIL $V rt400 attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  step "GUARDRAIL $V rt400 GAVE UP"; return 1
}

step "RT400-NARROW START (vn1e, vn5e on $CACHE)"
guardrail400 vn1e
guardrail400 vn5e
# The paired keyword tests, one record: the same tags redteam400.sh wrote
# (three 400-prompt arms and their n=100 runs) plus the two new arms beside
# their own n=100 runs. The file is rewritten whole from the raw guard-rail
# reports, which are not tracked, so the rewrite is refused when any earlier
# report is missing rather than silently dropping its entry.
MISS_TAGS=""
for t in const_${TAG}_all_rt400 const_${TAG}_vn10e_rt400 const_${TAG}_vn10ebot_rt400 const_${TAG}_all const_${TAG}_vn10e const_${TAG}_vn10ebot; do
  [ -s results/bench/${t}_guardrail.json ] || MISS_TAGS="$MISS_TAGS $t"
done
if [ -z "$MISS_TAGS" ]; then
  $PY eval/const_refusal_mcnemar.py --pairs base:psilm,zeroed:psilm \
    --tags const_${TAG}_all_rt400,const_${TAG}_vn10e_rt400,const_${TAG}_vn10ebot_rt400,const_${TAG}_all,const_${TAG}_vn10e,const_${TAG}_vn10ebot,const_${TAG}_vn1e_rt400,const_${TAG}_vn5e_rt400,const_${TAG}_vn1e,const_${TAG}_vn5e \
    --out results/constitution/refusal_mcnemar_rt400.json >> $LOG 2>&1 || step "MCNEMAR rewrite exited $?"
else
  step "SKIP the mcnemar rewrite; raw reports missing for:$MISS_TAGS (the new arms' counts are in their run logs)"
fi
step "RT400-NARROW COMPLETE"
