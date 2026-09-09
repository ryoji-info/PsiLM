#!/bin/bash
# Prune the retired probe models, then move the Hugging Face cache from
# ~/.cache/huggingface to ~/Documents/huggingface.
#
# Waits for the Gemma leaky sweep to finish first: it reads the cache with
# HF_HUB_OFFLINE=1 and its retry harness restarts the process on failure, so a
# restart after the move would look in the old path and fail.
#
# Both paths are on the same volume, so `mv` is instant and preserves the
# cache's internal symlinks (snapshots/ entries point into blobs/; `cp -r`
# would break or duplicate them). ~/Documents is a real local directory --
# Desktop & Documents iCloud sync is off -- so this does not trigger an upload.
set -u
LOG=/Users/rxiii/Documents/GitHub/PsiLM/results/bench/relocate_hf.log
SRC=/Users/rxiii/.cache/huggingface
DST=/Users/rxiii/Documents/huggingface
say() { echo "[$(date +%H:%M)] $*" >> $LOG; }

# retired feasibility probes: concluded in the paper, re-downloadable if ever rerun.
# Qwen3.8-27B-4bit is deliberately KEPT at the maintainer's request (15G).
PRUNE=(
  "models--mlx-community--Qwen3-30B-A3B-4bit"     # MoE probe (eval/moe_probe.py), 16G
  "models--Qwen--Qwen3-30B-A3B"                   # its tokenizer, 15M
  "models--mlx-community--gemma-4-12B-it-qat-4bit" # aborted download, 44K
)

say "=== waiting for the Gemma sweep ==="
until grep -qE "LEAKY-GEMMA (COMPLETE|GAVE UP)" results/bench/leaky_sweep.log 2>/dev/null \
      || ! pgrep -f "bench_guardrail.py --tag leaky_gemma" > /dev/null; do sleep 120; done
sleep 30
if pgrep -f "bench_guardrail.py" > /dev/null; then
  say "ABORT: a benchmark is still running"; exit 1
fi
say "sweep finished: $(grep -E 'LEAKY-GEMMA (COMPLETE|GAVE UP)' results/bench/leaky_sweep.log | tail -1)"

# ---------------------------------------------------------------- prune
say "=== prune ==="
say "cache before: $(du -sh $SRC/hub | cut -f1)"
for r in "${PRUNE[@]}"; do
  if [ -d "$SRC/hub/$r" ]; then
    sz=$(du -sh "$SRC/hub/$r" | cut -f1)
    rm -rf "$SRC/hub/$r" && say "  removed $r ($sz)"
  else
    say "  skip $r (absent)"
  fi
done
say "cache after:  $(du -sh $SRC/hub | cut -f1)"

# ---------------------------------------------------------------- move
say "=== move to $DST ==="
if [ -e "$DST/hub" ]; then say "ABORT: $DST/hub already exists"; exit 1; fi
mkdir -p "$DST" || { say "ABORT: cannot create $DST"; exit 1; }
mv "$SRC/hub" "$DST/hub"           || { say "ABORT: hub move failed"; exit 1; }
[ -d "$SRC/datasets" ] && mv "$SRC/datasets" "$DST/datasets" && say "  datasets moved"
[ -d "$SRC/xet" ]      && mv "$SRC/xet"      "$DST/xet"      && say "  xet moved"
for f in token stored_tokens; do
  [ -f "$SRC/$f" ] && mv "$SRC/$f" "$DST/$f" && say "  $f moved"
done
ln -s "$DST/hub" "$SRC/hub" && say "  symlink $SRC/hub -> $DST/hub"

# ---------------------------------------------------------------- persist
if ! grep -q "HF_HOME=$DST" ~/.zshrc 2>/dev/null; then
  printf '\n# Hugging Face cache lives with the other models\nexport HF_HOME=%s\n' "$DST" >> ~/.zshrc
  say "  HF_HOME added to ~/.zshrc"
else
  say "  HF_HOME already in ~/.zshrc"
fi

# ---------------------------------------------------------------- verify
say "=== verify ==="
say "moved: $(du -sh $DST/hub | cut -f1) in $(ls -d $DST/hub/models--* 2>/dev/null | wc -l | tr -d ' ') model repos"
say "symlink resolves: $(readlink $SRC/hub)"
say "token present: $([ -f $DST/token ] && echo yes || echo no)"
cd /Users/rxiii/Documents/GitHub/PsiLM
HF_HOME=$DST HF_HUB_OFFLINE=1 .venv/bin/python - >> $LOG 2>&1 <<'PY'
# the real test: resolve the two models still in use, from the new location
from huggingface_hub import snapshot_download
for r in ("mlx-community/Qwen3-8B-4bit", "mlx-community/gemma-4-12B-it-4bit"):
    print(f"  resolved {r} -> {snapshot_download(r, local_files_only=True)}")
PY
say "=== done ==="
