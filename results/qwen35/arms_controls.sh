#!/bin/bash
# teacher ceiling (temperature-controlled) and content controls for every 9B constitution arm
cd /Users/rxiii/Documents/GitHub/PsiLM
export HF_HOME=/Users/rxiii/Documents/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_VERBOSITY=error
LOG=results/qwen35/arms_controls.log
step() { echo "$1 $(date '+%F %H:%M')" >> $LOG; }
MISS=0
until { [ -f results/stage2c_qwen0.5b_all/compress.json ] && ! pgrep -f 'constitution_compress\.py|teacher_ceiling\.py' >/dev/null; } || [ $MISS -ge 6 ]; do
  if pgrep -f 'constitution_compress\.py|teacher_ceiling\.py' >/dev/null; then MISS=0; else MISS=$((MISS+1)); fi
  sleep 60
done
sleep 20
step "ARMS START"
D="data/constitution_test_qwen35.json,data/constitution_helpful_test_qwen35.json"
for a in vn10e vn10ebot vn1e vn5e match41 match205 allplain vn vn5 match0 match1 match2; do
  C=results/stage2c_qwen35_$a
  if ! grep -q temperature_crossfit $C/teacher_ceiling.json 2>/dev/null; then
    .venv/bin/python eval/teacher_ceiling.py --ckpt $C/bridges.npz --data $D --n 50 --fresh > $C/teacher_ceiling.log 2>&1
    step "CEILING $a rc=$? $(grep -c TEACHER-CEILING $C/teacher_ceiling.log)"
  fi
  if [ ! -f $C/compress.json ]; then
    .venv/bin/python eval/constitution_compress.py --ckpt $C/bridges.npz --data $D \
      --noharm data/noharm_qwen35_all.json --heldout-rows results/bench/const_qwen35_all_guardrail.rows.jsonl \
      --tasks-cache results/bench/tasks_const_qwen35_n100.json --variants fp32+native --no-real > $C/compress.log 2>&1
    step "CONTROLS $a rc=$?"
  fi
done
step "ARMS COMPLETE"
