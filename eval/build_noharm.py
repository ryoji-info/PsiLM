"""No-harm data for gate-selectivity training.

The v8 guard-rail showed the 8B gate open at 0.98 on every prompt, costing
55 points on GSM8K: trained on physics questions only, the gate never saw a
prompt where the channel should close. This script builds that missing arm:
non-physics prompts (GSM8K *train* split, MMLU *validation* split -- disjoint
from the benchmark's test items) paired with the FROZEN BACKBONE'S OWN greedy
continuation, so the coupled model can be trained to reproduce the backbone
where physics is irrelevant. Prompts use the benchmark's builders verbatim.

Usage:
  python eval/build_noharm.py --model mlx-community/Qwen3-8B-4bit \
      --hf-tokenizer Qwen/Qwen3-8B --n-gsm8k 400 --n-mmlu 200 --max-new 32
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx_lm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from eval.bench_common import chat_ids, gsm8k_user, mmlu_user  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-gsm8k", type=int, default=400)
    ap.add_argument("--n-mmlu", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/noharm_train.json")
    args = ap.parse_args()

    from datasets import load_dataset
    rng = random.Random(args.seed)
    prompts = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for i in rng.sample(range(len(ds)), args.n_gsm8k):
        prompts.append({"source": f"gsm8k:train:{i}", "user": gsm8k_user(ds[i]["question"])})
    try:
        ds = load_dataset("cais/mmlu", "all", split="validation")
        for i in rng.sample(range(len(ds)), args.n_mmlu):
            r = ds[i]
            prompts.append({"source": f"mmlu:validation:{i}", "user": mmlu_user(r["question"], list(r["choices"]))})
    except Exception as e:      # HF storage faults: fall back to GSM8K-only rather than block
        print(f"[warn] MMLU validation unavailable ({str(e)[:120]}); using GSM8K train only", flush=True)
        extra = [i for i in range(len(load_dataset("openai/gsm8k", "main", split="train")))]
        ds = load_dataset("openai/gsm8k", "main", split="train")
        used = {int(p["source"].split(":")[-1]) for p in prompts}
        for i in rng.sample([i for i in extra if i not in used], args.n_mmlu):
            prompts.append({"source": f"gsm8k:train:{i}", "user": gsm8k_user(ds[i]["question"])})
    rng.shuffle(prompts)

    model, tok = mlx_lm.load(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    out, t0 = [], time.time()
    for k, p in enumerate(prompts):
        ids = chat_ids(hf_tok, p["user"])
        text = mlx_lm.generate(model, hf_tok, prompt=list(ids), max_tokens=args.max_new, verbose=False)
        tgt = hf_tok.encode(text, add_special_tokens=False)[: args.max_new]
        if not tgt:
            continue
        out.append({"source": p["source"], "prompt_ids": list(map(int, ids)),
                    "target_ids": list(map(int, tgt)), "target_text": text})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(prompts)} ({(time.time()-t0)/(k+1):.1f}s each)", flush=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"NOHARM DATA: {len(out)} items -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
