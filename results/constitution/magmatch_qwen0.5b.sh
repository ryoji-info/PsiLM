#!/bin/bash
# Is the value-neuron result a VALUE effect or an ACTIVATION-MAGNITUDE effect?
#
# Both comparisons that make the design's case are confounded. The nine top-1%
# value neurons at layer 16 carry a summed activation RMS of 13.6 over rollout
# states and include the single largest dimension of the stream; the nine random
# dimensions they were compared against carry 3.3 in total and nothing above
# rank 164. And of three sink-free random draws, the one that damaged GSM8K as
# much as the value neurons (0.19 against 0.18, base 0.31) is exactly the one
# holding a rank-5 and a rank-18 dimension. So "the value neurons are special"
# may be "large coordinates are special".
#
# The control: three sets of nine dimensions, each matched to a value neuron's
# activation RMS, drawn from outside the top-5% value neurons and outside the
# attention sink (results/value_neurons/qwen0.5b/magmatch<seed>_layer16.json,
# summed RMS 10.3-11.5 against the value set's 13.6).
#
#   1. ablation    zero each matched set on GSM8K test, n=100, greedy, 384 tokens.
#                  If they cost the value set's 13 points, the ablation measures
#                  magnitude; if they cost the benign draws' 0-5, it measures value.
#   2. bridge      train the full constitution bridge writing into magmatch0 with
#                  the recipe of every other variant, then evaluate and guard-rail
#                  it. This is the decisive control for the design claim: the
#                  random-9 variant it replaces was inert (CE 0.941 against the
#                  value neurons' 0.925 and the base's 0.943), and a magnitude
#                  effect would put this variant at 0.925 too.
#   3. hand back   relaunch the 9B prep chain at its resume entry point.
#
# One job on the GPU at a time: the Metal watchdog kills MLX jobs that share it.
# Markers in results/constitution/magmatch_qwen0.5b.log; ends with "MAGMATCH COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
T=Qwen/Qwen2.5-0.5B-Instruct
CM=results/constitution_model/qwen2.5-0.5b-constitution
VN=results/value_neurons/qwen0.5b
DATA_TR=data/constitution_train_qwen0.5b.json
DATA_VA=data/constitution_val_qwen0.5b.json
DATA_TE=data/constitution_test_qwen0.5b.json
DATA_HE=data/constitution_helpful_test_qwen0.5b.json
NEG=data/noharm_qwen0.5b_all.json
LOG=results/constitution/magmatch_qwen0.5b.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

while pgrep -f 'eval/vn_collect.py|bash results/qwen35/constitution_prep' > /dev/null; do sleep 30; done
step "MAGMATCH START"

# 1. the ablation. Base is greedy and shim-free, so it is identical across the
#    three files and only run once.
for s in 0 1 2; do
  ARMS=zeroV; [ $s = 0 ] && ARMS=base,zeroV
  for i in 1 2 3; do
    $PY eval/vn_ablate.py --tag qwen0.5b --neurons $VN/magmatch${s}_layer16.json \
      --n 100 --seed 0 --max-new 384 --random-seeds 0 --arms $ARMS \
      --out-suffix _magmatch$s >> $VN/ablate_magmatch.log 2>&1 && break
    step "ABLATE magmatch$s attempt $i exited $?; resuming"; sleep 30
    [ $i = 3 ] && { step "ABLATE magmatch$s FAILED"; break; }
  done
  step "ABLATE magmatch$s: $(sed -n '/^layer /,$p' $VN/ablate_magmatch.log | grep -E '^(base|zeroV) ' | tail -2 | tr -s ' ' | tr '\n' '|')"
done
step "ABLATE DONE"

# 2. the bridge variant, identical to the others but for --write-dims
C="--model $M --hf-tokenizer $T --const-model $CM --batch 8 --lr 3e-4 --l-rev 16 \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 48 --k-fwd 8 --m-tokens 8 --read-dims all"
NAME=magmatch; TAG=qwen0.5b_$NAME; D=results/stage2c_$TAG; mkdir -p $D; FLAG=--fresh
for i in 1 2 3 4; do
  for j in 1 2 3; do
    $PY eval/mlx_constitution_train.py --tag $TAG --steps 500 $FLAG $C \
        --write-dims "$VN/magmatch0_layer16.json" >> $D/supervisor.log 2>&1 && break
    step "TRAIN $NAME chunk $i attempt $j exited $?; resuming"; FLAG=""; sleep 30
    [ $j = 3 ] && { step "TRAIN $NAME chunk $i FAILED"; step "MAGMATCH FAILED"; exit 1; }
  done
  FLAG=""
  step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
done
for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
  S=${pair%%:*}; F=${pair##*:}
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $T \
      --const-model $CM --data $F --arms base,psilm,zeroed --n 100 --max-new 32 \
      --out $D/eval_$S.json > $D/eval_$S.log 2>&1 \
    || { step "EVAL $NAME $S FAILED"; break; }
  step "EVAL $NAME $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
done

GT=const_qwen0.5b_magmatch; CACHE=results/bench/tasks_const_qwen0.5b_n100.json; FLAG=--fresh
COMMON="--tasks-cache $CACHE --model $M --hf-tokenizer $T --bridge-kind constitution \
  --ckpt $D/bridges.npz --const-model $CM --redteam-data $DATA_TE \
  --datasets redteam,gsm8k,mmlu,boolq --arms base,psilm,zeroed \
  --max-new-gsm8k 384 --max-new-mmlu 256 --max-new-boolq 16 --max-new-redteam 128 \
  --gsm8k-nudge 1 --kl --seed 0 --print-every 5 --save-every 5"
for i in $(seq 1 10); do
  $PY eval/bench_guardrail.py --tag $GT --n 100 $COMMON $FLAG >> results/bench/${GT}_run.log 2>&1 \
    && { step "GUARDRAIL $NAME COMPLETE (attempt $i)"; sed -n '/^dataset /,/^$/p' results/bench/${GT}_run.log | tail -8 >> $LOG; break; }
  step "GUARDRAIL $NAME attempt $i exited; resuming"; FLAG=--resume; sleep 20
done
step "MAGMATCH COMPLETE"

# 3. hand the GPU back to the 9B pipeline
nohup bash results/qwen35/constitution_prep_resume.sh > /dev/null 2>&1 &
step "9B PREP RESUMED"
