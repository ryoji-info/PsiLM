#!/bin/bash
# Magnitude-matched random controls for the two narrow parity arms.
#
# results/qwen35/parity_widths.sh trains the 41 value neurons (vn1e) and the top
# 5% (vn5e) at energy parity with the full-width write. Those arms cannot say
# whether the probe's CHOICE of coordinates matters at 41 and 205; at 410 it
# did, the wrong way (probe-best 410: 98x the MMLU divergence of four matched
# controls). This trains the same-sized controls at the same energy:
#
#   match41    41 coordinates outside the probe-best 410, matched one to one on
#              activation RMS to the 41 value neurons   (match41_layer24.json)
#   match205   205 coordinates, matched to the top 5%    (match205_layer24.json)
#
# Both masks come from eval/vn_matched_draw.py, which records each control's
# parity cap in the mask file; the caps are re-derived here from the rms file
# and refused if they disagree with the recorded ones by more than 1e-3.
#
# Protocol byte-identical to parity_widths.sh and constitution_width.sh: two
# 500-step chunks, meta check on mask and cap, 50-item test and helpful evals
# at step 1000, the four-dataset guard-rail (n=100) on the shared task cache.
#
# Queue: after the parity arms (PARITY COMPLETE); allplain.sh and
# plainpartner_widths.sh wait for this chain.
#
# LAUNCH A COPY, NOT THIS FILE (bash reads scripts lazily; editing a running
# script resumes it mid-file). Ends with "PARITY-CONTROLS COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx
CM=results/constitution_model/qwen2.5-0.5b-constitution
VN=results/value_neurons/qwen35
TAG=qwen35
L=24
DATA_TR=data/constitution_train_$TAG.json
DATA_VA=data/constitution_val_$TAG.json
DATA_TE=data/constitution_test_$TAG.json
DATA_HE=data/constitution_helpful_test_$TAG.json
NEG=data/noharm_qwen35_all.json
LOG=results/qwen35/parity_controls.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "PARITY-CONTROLS WAITING for the parity arms (vn1e, vn5e)"
MISS=0
until grep -q "PARITY COMPLETE" results/qwen35/parity_widths.log 2>/dev/null; do
  # Count the whole upstream queue as alive, not just the predecessor, or a
  # legitimate gap between chains looks like a crash.
  if pgrep -f 'parity_run.sh|guardrail_phys.sh|contentless_run.sh|rt400_run.sh' > /dev/null; then MISS=0; else MISS=$((MISS + 1)); fi
  [ $MISS -ge 3 ] && { step "nothing upstream is running or complete; proceeding"; break; }
  sleep 60
done
while pgrep -f 'mlx_constitution_(train|eval)\.py|bench_guardrail\.py|psilm2/(train_dual|bench)\.py' > /dev/null; do sleep 60; done

[ -s $DATA_TR ] && [ -s $DATA_VA ] && [ -s $DATA_TE ] && [ -s $DATA_HE ] && [ -s $NEG ] \
  && [ -s $CM/model.safetensors ] || { step "PREREQ files missing"; exit 1; }
[ -s $VN/layer$L.json ] && [ -s $VN/rms_layer${L}_full.json ] || { step "PREREQ mask or rms file missing"; exit 1; }
[ -s results/bench/tasks_const_${TAG}_n100.json ] || { step "PREREQ no shared task cache"; exit 1; }
[ -s $VN/match41_layer$L.json ] && [ -s $VN/match205_layer$L.json ] \
  || { step "PREREQ the matched control masks are missing (eval/vn_matched_draw.py)"; exit 1; }

# Re-derive each control's parity cap from the rms file and the mask, and
# check the mask is what it claims: n coordinates, none in the probe-best 410,
# none the attention sink. The recorded cap in the mask file is the one used.
CAPS=$($PY - "$VN/rms_layer${L}_full.json" "$VN/layer$L.json" "$VN/match41_layer$L.json" "$VN/match205_layer$L.json" <<'PYEOF'
import json, math, sys
rms = json.load(open(sys.argv[1]))["rms"]; e_full = sum(r * r for r in rms)
top410 = set(json.load(open(sys.argv[2]))["ranking"][:410])
bad, caps = [], []
for f, n in ((sys.argv[3], 41), (sys.argv[4], 205)):
    m = json.load(open(f)); w = m["ranking"][:n]
    if len(m["ranking"]) != n or len(set(w)) != n: bad.append(f"{f}: ranking has {len(m['ranking'])} entries, expected {n}")
    if set(w) & top410 or 3994 in w: bad.append(f"{f}: overlaps the probe-best 410 or the sink")
    parity = 0.2 * math.sqrt(e_full / sum(rms[i] ** 2 for i in w))
    if abs(parity - float(m["parity_cap"])) > 1e-3: bad.append(f"{f}: parity cap {parity:.4f} != recorded {m['parity_cap']}")
    caps.append(f"{m['parity_cap']:.4f}")
print("BAD " + "; ".join(bad) if bad else " ".join(caps))
PYEOF
)
case "$CAPS" in BAD*) step "PREREQ control masks: $CAPS"; exit 1;; esac
CAP41=${CAPS%% *}; CAP205=${CAPS##* }
[ -n "$CAP41" ] && [ -n "$CAP205" ] && [ "$CAP41" != "$CAP205" ] || { step "PREREQ could not read both caps from '$CAPS'"; exit 1; }
CKPT=""; [ "$L" -lt 26 ] && CKPT="--checkpoint-from -1"
step "PARITY-CONTROLS START layer=$L (match41 cap $CAP41, match205 cap $CAP205; 2 chunks each) $CKPT"

# Byte-identical to results/qwen35/constitution_width.sh's C.
C="--model $M --hf-tokenizer $M --const-model $CM --batch 2 --lr 3e-4 --l-fwd 13 --l-rev $L \
  --gate-bias 0.0 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 32 --k-fwd 8 --m-tokens 8 --read-dims all $CKPT"

train_variant() {   # $1 name, $2 --write-dims spelling, $3 --inj-cap, $4 chunks wanted, $5 mask file, $6 mask size
  local NAME=$1 WD=$2 CAP=$3 WANT=$4 REF=$5 NDIM=$6 T=${TAG}_$1 D=results/stage2c_${TAG}_$1
  mkdir -p $D
  local DONE; DONE=$(grep -c 'CHUNK DONE' $D/supervisor.log 2>/dev/null); DONE=${DONE:-0}
  local FLAG=""; [ "$DONE" = "0" ] && FLAG=--fresh
  step "TRAIN $NAME: $DONE of $WANT chunks done (write-dims $WD, inj-cap $CAP)"
  # NOT `seq $((DONE+1)) $WANT`: BSD seq counts DOWN when the first argument
  # exceeds the second, so a finished variant would train two more chunks.
  local i=$DONE j
  while [ $i -lt $WANT ]; do
    i=$((i + 1))
    for j in 1 2 3; do
      $PY eval/mlx_constitution_train.py --tag $T --steps 500 $FLAG $C \
          --write-dims "$WD" --inj-cap $CAP >> $D/supervisor.log 2>&1 && break
      step "TRAIN $NAME chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
      [ $j = 3 ] && { step "TRAIN $NAME chunk $i FAILED"; return 1; }
    done
    FLAG=""
    step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
  done
  [ -s $D/bridges.npz ] || { step "TRAIN $NAME no checkpoint to evaluate"; return 1; }
  # The checkpoint's own meta must carry the mask and the cap this arm claims:
  # the evaluator and the guard-rail read both from it, so a silent fallback to
  # 0.2 would quietly turn this back into the starved experiment.
  local CHK; CHK=$($PY - "$D/bridges.npz.meta" "$CAP" "$REF" "$NDIM" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1])); r = json.load(open(sys.argv[3]))["ranking"][: int(sys.argv[4])]
w, cap, step = m.get("write_dims") or [], float(m["inj_cap"]), int(m.get("step", -1))
bad = []
if sorted(w) != sorted(r): bad.append(f"dims={len(w)} differ from the mask file's first {sys.argv[4]}")
if abs(cap - float(sys.argv[2])) > 1e-9: bad.append(f"cap={cap}")
if step != 1000: bad.append(f"step={step}")
print("OK" if not bad else "BAD " + " ".join(bad))
PYEOF
)
  [ "$CHK" = "OK" ] || { step "TRAIN $NAME meta wrong: $CHK"; return 1; }
  # The cap bounds the write; nothing in the loss pushes the write UP to it, and
  # no arm has been trained above 0.5514. Energy parity is realized only if the
  # cap binds, so log the realized cap (ratio_cont/gate_cont at the step-1000
  # eval) and warn when it misses; the analysis must use the realized value.
  local SAT; SAT=$($PY - "$D/train_log.jsonl" "$CAP" <<'PYEOF'
import json, sys
sat = None
for line in open(sys.argv[1]):
    d = json.loads(line)
    e = (d.get("eval") or {}).get("psilm") or {}
    if d.get("step") == 1000 and e.get("ratio_cont") and e.get("gate_cont"):
        sat = e["ratio_cont"] / e["gate_cont"]; gate = e["gate_cont"]
cap = float(sys.argv[2])
if sat is None: print("n/a (no step-1000 eval line)")
else: print(f"{sat:.4f} at gate {gate:.4f}" + ("" if abs(sat - cap) / cap <= 0.02 else f"  WARNING: {sat / cap:.0%} of the cap; the write did not reach it and energy parity is NOT realized"))
PYEOF
)
  step "TRAIN $NAME realized cap at step 1000: $SAT (--inj-cap $CAP)"
  for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
    local S=${pair%%:*} F=${pair##*:}
    [ -s $D/eval_$S.json ] && { step "EVAL $NAME $S already done"; continue; }
    $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $M \
        --const-model $CM --data $F --arms base,psilm,zeroed --n 50 --max-new 24 \
        --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $NAME $S FAILED"; return 1; }
    step "EVAL $NAME $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
  done
}

guardrail() {       # $1 name -- same 100 cached items as vn / vn5 / all / vn10e
  local NAME=$1 D=results/stage2c_${TAG}_$1 GT=const_${TAG}_$1 FLAG=--fresh
  local CACHE=results/bench/tasks_const_${TAG}_n100.json
  [ -s results/bench/${GT}_guardrail_summary.json ] && { step "GUARDRAIL $NAME already done"; return 0; }
  local COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $M --bridge-kind constitution \
    --ckpt $D/bridges.npz --const-model $CM --redteam-data $DATA_TE \
    --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
    --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
    --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
  local i
  for i in $(seq 1 20); do
    $PY eval/bench_guardrail.py --tag $GT --n 100 $COMMON $FLAG >> results/bench/${GT}_run.log 2>&1 \
      && { step "GUARDRAIL $NAME COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${GT}_run.log | tail -8 >> $LOG; \
           $PY results/constitution/summarize.py --track-tags $GT >> $LOG 2>&1; \
           [ -s results/bench/${GT}_guardrail_summary.json ] || step "GUARDRAIL $NAME tracked summary missing after summarize.py"; return 0; }
    step "GUARDRAIL $NAME attempt $i exited; resuming"; FLAG=--resume; sleep 30
  done
  step "GUARDRAIL $NAME GAVE UP"; return 1
}

train_variant match41  "$VN/match41_layer$L.json:top41"   $CAP41  2 "$VN/match41_layer$L.json"  41  && guardrail match41
train_variant match205 "$VN/match205_layer$L.json:top205" $CAP205 2 "$VN/match205_layer$L.json" 205 && guardrail match205
$PY eval/const_refusal_mcnemar.py \
  --tags const_${TAG}_vn,const_${TAG}_vn5,const_${TAG}_all,const_${TAG}_vn10e,const_${TAG}_vn10ebot,const_${TAG}_vn1e,const_${TAG}_vn5e,const_${TAG}_match41,const_${TAG}_match205 \
  --out results/constitution/refusal_mcnemar_${TAG}.json >> $LOG 2>&1
step "PARITY-CONTROLS COMPLETE"
