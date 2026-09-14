"""Sample GSM8K-train trajectories from a frozen backbone, keeping the residual
stream at every position of every requested layer.

Stage 1 of the value-neuron identification of arXiv:2602.00986 ("Sparse Reward
Subsystem in Large Language Models"): the probe in eval/vn_probe.py needs
(states, terminal reward) pairs, so we generate one sample per problem at the
paper's temperature 1.0 / top-p 0.95, score it with the guard-rail's own
"Answer: <number>" parser, and write the hidden states next to the reward.

Splits: TRAIN only. The probe and the layer choice are selection, and the
project rule is that the GSM8K test split never touches anything that trains or
selects -- eval/vn_ablate.py is the only script here that sees test, as the
final causal check.

Layout (tag = the backbone's short name):
    results/value_neurons/<tag>/traj/<idx>.npz     h<l> float16 (T, d) per layer
    results/value_neurons/<tag>/index.jsonl        one row per trajectory
``idx`` is the position in the seeded shuffle of the train split, so a longer
--n is a superset of a shorter one at the same --seed and a re-run resumes by
skipping the ids already in index.jsonl.

Disk is the real budget, not time: 10 layers x ~250 states x d x 2 bytes is
4.5 MB per trajectory at d = 896 (4.5 GB for n = 1000) and 26 MB at d = 4096.
The per-item line prints what it wrote; pick fewer --layers on a wide backbone.

Usage:
  .venv/bin/python eval/vn_collect.py --tag qwen0.5b --n 1000 \
      --layers 4,6,8,10,12,14,15,16,18,20 --max-new 320
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402

from eval.bench_common import (  # noqa: E402
    _seeded_prefix, chat_ids, eos_id_set, gsm8k_gold, gsm8k_user, read_jsonl, score,
)
from psilm.mlx.value_neurons import StreamDecoder  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_HF_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LAYERS = "4,6,8,10,12,14,15,16,18,20"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="qwen0.5b")
    ap.add_argument("--out-root", default="results/value_neurons")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hf-tokenizer", default=DEFAULT_HF_TOKENIZER)
    ap.add_argument("--n", type=int, default=1000, help="problems from the seeded shuffle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--layers", default=DEFAULT_LAYERS,
                    help="layer boundaries to capture; 'layer l' = the stream entering block l")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax instead of the paper's temperature 1.0 / top-p 0.95")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--flush-every", type=int, default=32,
                    help="decode steps between device->host copies of the captured states")
    ap.add_argument("--nudge", type=int, default=1,
                    help="append the 'Answer: <number>' instruction (the guard-rail's prompt)")
    return ap.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    root = Path(args.out_root) / args.tag
    traj_dir = root / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.jsonl"
    done = {int(r["idx"]) for r in read_jsonl(index_path)}

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from psilm.mlx.gemma_loader import load_backbone_any

    ds = load_dataset("openai/gsm8k", "main", split="train")
    picks = _seeded_prefix(len(ds), args.n, args.seed)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    model, _stock, mlx_tok = load_backbone_any(args.model)
    model.freeze()
    d_model = int(model.args.hidden_size)
    n_layers = len(model.model.layers)
    dec = StreamDecoder(model, hf_tok, eos_id_set(hf_tok, mlx_tok),
                        flush_every=args.flush_every)
    sampler = None
    if not args.greedy:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    print(f"model={args.model} d_model={d_model} n_layers={n_layers} layers={layers}")
    print(f"train problems={len(ds)} picked={len(picks)} already_done={len(done)} "
          f"sampler={'greedy' if args.greedy else f'temp{args.temp}/top_p{args.top_p}'}")
    # a float16 state row per layer per position: the honest disk estimate up front
    mb_row = len(layers) * d_model * 2 / 2 ** 20
    print(f"disk: {mb_row * 1000:.2f} MB per 1000 states "
          f"(<= {mb_row * (args.max_new + 120) * args.n / 1024:.2f} GB for n={args.n})")

    t_start = time.perf_counter()
    n_tok, n_new, n_correct, n_bytes = 0, 0, 0, 0
    for i, ds_idx in enumerate(picks):
        if i in done:
            continue
        row = ds[int(ds_idx)]
        gold = gsm8k_gold(row["answer"])
        ids = chat_ids(hf_tok, gsm8k_user(row["question"], nudge=bool(args.nudge)))
        # per-item seed: a resumed run re-draws the same continuation for the same idx
        mx.random.seed(args.seed * 1000003 + i)
        out = dec.rollout(ids, max_new=args.max_new, capture=layers, sampler=sampler)
        pred, ok = score("number", out["text"], gold)
        npz = traj_dir / f"{i}.npz"
        np.savez(npz, **{f"h{l}": out["states"][l] for l in layers})
        size = npz.stat().st_size
        rec = {"idx": i, "ds_index": int(ds_idx), "prompt_len": out["prompt_len"],
               "gen_len": len(out["gen_ids"]), "n_gen_states": out["n_gen_states"],
               "n_states": out["prompt_len"] + out["n_gen_states"],
               "reward": int(ok), "pred": pred, "gold": gold,
               "stopped_eos": bool(out["stopped_eos"]), "layers": layers,
               "bytes": int(size), "sec": round(out["sec"], 2), "text": out["text"]}
        with index_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        del out
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        n_new += 1
        n_tok += rec["gen_len"]
        n_correct += rec["reward"]
        n_bytes += size
        el = time.perf_counter() - t_start
        left = (len(picks) - len(done) - n_new) * el / max(1, n_new)
        print(f"[{n_new}/{len(picks) - len(done)}] idx={i} ds={ds_idx} "
              f"P={rec['prompt_len']} G={rec['gen_len']} r={rec['reward']} "
              f"eos={int(rec['stopped_eos'])} {size / 2**20:.2f}MB "
              f"{rec['gen_len'] / max(1e-9, rec['sec']):.1f}tok/s "
              f"acc={n_correct / n_new:.3f} disk={n_bytes / 2**30:.2f}GB "
              f"eta={left / 60:.1f}min", flush=True)

    total = len(read_jsonl(index_path))
    el = time.perf_counter() - t_start
    print(f"new={n_new} gen_tokens={n_tok} elapsed={el / 60:.1f}min "
          f"tok/s={n_tok / max(1e-9, el):.1f} reward_rate={n_correct / max(1, n_new):.3f} "
          f"wrote={n_bytes / 2**30:.3f}GB peak={mx.get_peak_memory() / 2**30:.1f}GB")
    print(f"VN-COLLECT DONE n={total}")


if __name__ == "__main__":
    main()
