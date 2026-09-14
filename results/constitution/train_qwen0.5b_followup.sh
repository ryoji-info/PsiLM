#!/bin/bash
# Follow-up to results/constitution/train_qwen0.5b.sh: the partner control.
# The value-neuron variant plateaued at CE 0.96 against the teacher (base 0.977)
# with training CE equal to held-out CE, so the limit is capacity or information,
# not overfitting. This run keeps everything of the vn variant (write into the 9
# value neurons at layer 16, same data, same schedule) and swaps the partner for
# the PLAIN Qwen2.5-0.5B-Instruct -- a model of the same architecture that does
# not contain the constitution. If it matches the vn variant, the constitution
# model contributes nothing the bridges can use at this size; if it falls short,
# the memorised document is doing work through the latent channel.
# Waits for the main chain to finish; ends with "FOLLOWUP COMPLETE".
cd /Users/rxiii/Documents/GitHub/PsiLM
PY=.venv/bin/python
M=mlx-community/Qwen2.5-0.5B-Instruct-4bit
T=Qwen/Qwen2.5-0.5B-Instruct
CM=Qwen/Qwen2.5-0.5B-Instruct
VN=results/value_neurons/qwen0.5b
DATA_TR=data/constitution_train_qwen0.5b.json
DATA_VA=data/constitution_val_qwen0.5b.json
DATA_TE=data/constitution_test_qwen0.5b.json
DATA_HE=data/constitution_helpful_test_qwen0.5b.json
NEG=data/noharm_qwen0.5b_all.json
LOG=results/constitution/train_qwen0.5b_followup.log
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d /Users/rxiii/Documents/huggingface/hub ] && export HF_HOME=/Users/rxiii/Documents/huggingface
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }

step "FOLLOWUP WAITING for the main chain"
until grep -q "TRAIN-0.5B COMPLETE\|PREREQ\|refusing" results/constitution/train_qwen0.5b.log 2>/dev/null; do sleep 60; done
while pgrep -f 'results/constitution/train_qwen0.5b.sh' > /dev/null; do sleep 30; done
L=$($PY -c "import json;print(json.load(open('$VN/summary.json'))['chosen_layer'])")
step "FOLLOWUP START layer=$L partner=$CM"
C="--model $M --hf-tokenizer $T --const-model $CM --batch 8 --lr 3e-4 --l-rev $L \
  --gate-bias 0.0 --inj-cap 0.2 --clip module --readout-norm dim --calib-n 32 \
  --data $DATA_TR --val $DATA_VA --noharm-data $NEG --noharm-every 2 --noharm-gate-only 1 \
  --lam-gate 1.0 --eval-n 48 --k-fwd 8 --m-tokens 8 --read-dims all"
NAME=plainpartner; TAG=qwen0.5b_$NAME; D=results/stage2c_$TAG; mkdir -p $D; FLAG=--fresh
for i in 1 2 3 4; do
  $PY eval/mlx_constitution_train.py --tag $TAG --steps 500 $FLAG $C --write-dims "$VN/layer$L.json" \
      >> $D/supervisor.log 2>&1 || { step "TRAIN $NAME chunk $i FAILED"; step "FOLLOWUP FAILED"; exit 1; }
  FLAG=""
  step "TRAIN $NAME chunk $i: $(grep 'CHUNK DONE' $D/supervisor.log | tail -1)"
done
for pair in "test:$DATA_TE" "helpful:$DATA_HE"; do
  S=${pair%%:*}; F=${pair##*:}
  $PY eval/mlx_constitution_eval.py --ckpt $D/bridges.npz --model $M --hf-tokenizer $T \
      --const-model $CM --data $F --arms base,psilm,zeroed --n 100 --max-new 32 \
      --out $D/eval_$S.json > $D/eval_$S.log 2>&1 || { step "EVAL $NAME $S FAILED"; step "FOLLOWUP FAILED"; exit 1; }
  step "EVAL $NAME $S: $(grep -E '^ +(base|psilm|zeroed) ' $D/eval_$S.log | tr -s ' ' | tr '\n' '|')"
done
step "FOLLOWUP COMPLETE"
