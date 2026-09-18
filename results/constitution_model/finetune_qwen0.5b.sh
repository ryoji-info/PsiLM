#!/usr/bin/env bash
# Fine-tune Qwen2.5-0.5B-Instruct into a model that *contains* Claude's constitution.
#
# The constitution bridge needs a frozen partner model whose hidden states are about
# the constitution, in the same structural role the frozen Fourier neural operator
# plays in PsiLM's physics bridges. This script is the whole recipe: corpus -> smoke ->
# three training segments -> a directory `mlx_lm.load()` accepts -> the containment
# check. Run it from anywhere; it cd's to the repo root.
#
#   bash results/constitution_model/finetune_qwen0.5b.sh corpus  # build the jsonl
#   bash results/constitution_model/finetune_qwen0.5b.sh smoke   # 20 iters, mechanics
#   bash results/constitution_model/finetune_qwen0.5b.sh seg1    # 5-13 min each
#   bash results/constitution_model/finetune_qwen0.5b.sh seg2
#   bash results/constitution_model/finetune_qwen0.5b.sh seg3
#   bash results/constitution_model/finetune_qwen0.5b.sh seg4
#   bash results/constitution_model/finetune_qwen0.5b.sh seg5
#   bash results/constitution_model/finetune_qwen0.5b.sh seg6
#   bash results/constitution_model/finetune_qwen0.5b.sh fuse    # write the model dir
#   bash results/constitution_model/finetune_qwen0.5b.sh check   # fine-tuned vs base
#   bash results/constitution_model/finetune_qwen0.5b.sh all     # all of the above
#
# Why segments, and why each segment retries. Two facts about this machine. (1) The
# M2 is shared with other agents, so every single GPU run stays under 15 minutes. A
# segment is therefore sized for the *worst* throughput seen during this campaign --
# it swung between 120 and 450 tokens/s depending on what the other agents were doing,
# so 100 iterations at batch 2 (~90k tokens) is 3.5 minutes at the top of that range
# and 12.5 at the bottom. (2) On a busy
# GPU macOS kills long MLX command buffers: two attempts died with
# `[METAL] Command buffer execution failed: Impacting Interactivity`, which is the
# display watchdog, not a bug in the training. Batch 2 halves the work in one command
# buffer (peak memory 4.6 GB instead of 6.6 GB), `--save-every 25` means a kill costs
# at most 25 iterations, and `seg` below resumes its own last checkpoint and retries.
#
# Each segment resumes the previous segment's weights with a lower peak learning rate:
# a hand-rolled step decay across segments, because mlx-lm restarts both its schedule
# and Adam's moments on resume. 600 iterations at batch 2 is ~7.3 epochs over the
# 164-record corpus -- few, because the corpus is 74k tokens and throughput is a few
# hundred tokens/s; the learning rate does the work the epochs cannot. To buy more
# epochs, add segments with `seg` (lower peak each time) and re-fuse.
set -euo pipefail

cd "$(dirname "$0")/../.."            # repo root
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # the recorded run used a local cache directory
export HF_HUB_DISABLE_XET=1
PY=.venv/bin/python
OUT=results/constitution_model
LOG=$OUT/finetune.log
DATA=data/constitution/ft
MODEL_DIR=$OUT/qwen2.5-0.5b-constitution

# `mlx_lm fuse` re-resolves a repo id through snapshot_download *without*
# allow_patterns, which fails on this machine (the cached snapshot has no
# .gitattributes/LICENSE/README.md and outgoing traffic is off). Handing it the local
# snapshot path instead skips the download entirely. Training is happy with the repo id.
BASE_REPO=Qwen/Qwen2.5-0.5B-Instruct
BASE_PATH=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ | head -1)

# Shared training arguments.
#   --fine-tune-type full --num-layers -1  : unfreeze all 24 decoder layers (357.9M of
#       494.0M parameters; mlx-lm only unfreezes model.layers, so embeddings and the
#       final norm keep their base values -- which is why the partner model's embedding
#       geometry in check.json is identical to the base model's).
#   --batch-size 2 : shorter command buffers (see above) and twice as many optimizer
#       updates per pass over the corpus, which is what memorisation wants. It is also
#       the validation batch size -- mlx-lm refuses a split smaller than one batch, and
#       at 2 all 6 held-out paragraphs are scored (at 4, only 4 of them were).
#   --grad-checkpoint : free here (339 tok/s without it, 338-450 with) and cuts peak
#       memory, which matters on a shared 24 GB machine.
#   --save-every 25 : crash insurance, see above.
COMMON=(--model "$BASE_REPO" --train --fine-tune-type full --num-layers -1
        --data "$DATA" --batch-size 2 --max-seq-length 1024
        --grad-checkpoint --seed 0 --save-every 25 --clear-cache-threshold 2GB
        --steps-per-report 10 --steps-per-eval 40 --val-batches -1)

# seg <name> <iters> <peak-lr> <end-lr> <warmup> [resume-from-adapter-dir]
#
# Runs <iters> iterations in as many attempts as it takes (up to 4). After a watchdog
# kill it picks up its own last 25-iteration checkpoint and asks for the remaining
# iterations, with the cosine schedule restarted over what is left -- not identical to
# an uninterrupted run, but reproducible from the log, which records every attempt.
# $dir/progress holds the cumulative completed iterations so a rerun of the whole
# script does not redo finished work.
seg() {
  local name=$1 iters=$2 peak=$3 end=$4 warm=$5 resume=${6:-}
  local dir=$OUT/adapters_$name
  mkdir -p "$dir"
  local have attempt=0 left from ck got
  have=$(cat "$dir/progress" 2>/dev/null || echo 0)
  while (( have < iters )); do
    attempt=$((attempt + 1))
    if (( attempt > 4 )); then
      echo "=== $name: giving up after 4 attempts ($have/$iters iters) ===" | tee -a "$LOG"
      return 1
    fi
    left=$((iters - have))
    from=""
    if (( have > 0 )); then
      from=$dir/adapters.safetensors
    elif [[ -n $resume ]]; then
      from=$OUT/$resume/adapters.safetensors
    fi
    cat > "$OUT/$name.yaml" <<YAML
lr_schedule:
  name: cosine_decay
  warmup: $warm
  warmup_init: 1.0e-7
  arguments: [$peak, $left, $end]
YAML
    echo "=== $name attempt $attempt: $left iters (have $have/$iters), cosine $peak -> $end, warmup $warm${from:+, resuming ${from#$OUT/}} ($(date -u +%FT%TZ)) ===" | tee -a "$LOG"
    if $PY -m mlx_lm lora "${COMMON[@]}" -c "$OUT/$name.yaml" --iters "$left" \
         --adapter-path "$dir" ${from:+--resume-adapter-file "$from"} 2>&1 | tee -a "$LOG"; then
      have=$iters
    else
      ck=$(ls "$dir"/[0-9]*_adapters.safetensors 2>/dev/null | tail -1 || true)
      got=0
      if [[ -n ${ck:-} ]]; then
        got=$(basename "$ck" | sed -E 's/^0*([0-9]+)_.*/\1/')
      fi
      have=$((have + got))
      echo "=== $name attempt $attempt died; kept $got iters (total $have/$iters) ===" | tee -a "$LOG"
      if (( got == 0 )); then sleep 20; fi   # let whatever else is on the GPU finish
    fi
    rm -f "$dir"/[0-9]*_adapters.safetensors
    echo "$have" > "$dir/progress"
  done
  echo "$iters" > "$dir/progress"
}

stage_corpus() { $PY eval/build_constitution_corpus.py 2>&1 | tee -a "$LOG"; }

stage_smoke() {
  echo "=== smoke: 20 iters, lr 2e-5 ($(date -u +%FT%TZ)) ===" | tee -a "$LOG"
  $PY -m mlx_lm lora "${COMMON[@]}" --learning-rate 2e-5 --iters 20 \
      --steps-per-eval 10 --adapter-path "$OUT/smoke" 2>&1 | tee -a "$LOG"
}

# The peak of 7e-5 is well above the 2e-5 the smoke used, and above what one would
# pick for a normal full fine-tune: with only ~9 epochs affordable, the step size has
# to carry the memorisation. Warmup exists because Adam's first steps at 7e-5 on 358M
# freshly unfrozen parameters are exactly where a full fine-tune diverges.
stage_seg1() { seg s1 100 7.0e-5 3.0e-5 12; }
stage_seg2() { seg s2 100 5.0e-5 2.0e-5 8 adapters_s1; }
stage_seg3() { seg s3 100 3.0e-5 1.0e-5 8 adapters_s2; }
stage_seg4() { seg s4 100 1.5e-5 2.0e-6 8 adapters_s3; }
# Segments 5 and 6 exist because of what the check said after segment 4. Teacher-forced
# train loss was 0.031 -- the corpus was fit -- yet 14 of 38 sections recited the *wrong*
# section verbatim. Averaged over ~700 target tokens, a badly predicted *first* token is
# 0.1% of the loss and 100% of a free-running recitation: get it wrong and the model
# falls into a different memorised section and reproduces that one perfectly. So these
# two segments buy 2.4 more epochs of the prompt->section association at a low learning
# rate, which is the part that was under-trained relative to the continuation.
stage_seg5() { seg s5 100 2.0e-5 5.0e-6 8 adapters_s4; }
stage_seg6() { seg s6 100 1.0e-5 1.0e-6 8 adapters_s5; }

stage_fuse() {
  local src=${1:-$OUT/adapters_s6}
  echo "=== fuse $src -> $MODEL_DIR ($(date -u +%FT%TZ)) ===" | tee -a "$LOG"
  $PY -m mlx_lm fuse --model "$BASE_PATH" --adapter-path "$src" \
      --save-path "$MODEL_DIR" 2>&1 | tee -a "$LOG"
  # Prove the directory is what mlx_lm.load() wants, and that it generates.
  $PY - <<PYEOF 2>&1 | tee -a "$LOG"
from mlx_lm import load, generate
model, tok = load("$MODEL_DIR")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Recite the section 'Hard constraints' of Claude's constitution."}],
    add_generation_prompt=True, tokenize=False)
print("loaded:", type(model).__name__, "hidden", model.args.hidden_size)
print(generate(model, tok, prompt, max_tokens=60))
PYEOF
}

# Two runs, not one: ~16k greedy tokens per model is several minutes on a shared GPU,
# so each model is measured in its own short run and the second reuses the first's
# cached per-model result (check_raw_*.json) to write the joint table.
stage_check() {
  $PY eval/constitution_check.py --model "$MODEL_DIR" --base none \
      --out "$OUT" 2>&1 | tee -a "$LOG"
  $PY eval/constitution_check.py --model "$MODEL_DIR" --base "$BASE_REPO" --reuse \
      --out "$OUT" 2>&1 | tee -a "$LOG"
}

case "${1:-all}" in
  corpus) stage_corpus ;;
  smoke)  stage_smoke ;;
  seg1)   stage_seg1 ;;
  seg2)   stage_seg2 ;;
  seg3)   stage_seg3 ;;
  seg4)   stage_seg4 ;;
  seg5)   stage_seg5 ;;
  seg6)   stage_seg6 ;;
  fuse)   stage_fuse "${2:-}" ;;
  check)  stage_check ;;
  all)    stage_corpus; stage_smoke; stage_seg1; stage_seg2; stage_seg3; stage_seg4
          stage_seg5; stage_seg6; stage_fuse; stage_check ;;
  *)      echo "unknown stage: $1" >&2; exit 2 ;;
esac
