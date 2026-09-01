"""Curriculum probe: can a frozen backbone read ONE number from the latent channel?

The minimal possible coupling task. The physics hemisphere, the readouts,
and the pointer are all removed: a fixed random value v ~ U(-1, 1) is
encoded by a tiny MLP into K soft tokens and injected via the same gated
cross-attention used everywhere else. The prompt asks nothing answerable
("What is the secret number?"); the target is v to two decimals. The ONLY
route from v to the answer is the latent channel.

If a backbone cannot learn this, no amount of readout engineering will make
PsiLM work on it, and 'strong frozen backbones resist latent steering' is
established cleanly. If it can, the fault lies in our readout pipeline.

Usage:
  python eval/copy_probe.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
      --hf-tokenizer Qwen/Qwen2.5-0.5B-Instruct --tag 0.5b --steps 600
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import mlx_lm  # noqa: E402
from mlx.optimizers import clip_grad_norm  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.bridges import GatedCrossAttentionMLX  # noqa: E402
from psilm.mlx.model import cross_entropy_masked  # noqa: E402
from psilm.mlx.moe_patch import patch_moe_gather  # noqa: E402
from psilm.mlx.staged import MlxStream  # noqa: E402

SYSTEM = "You are a helpful assistant."
QUESTION = "What is the secret number? Answer with a number rounded to 2 decimal places."


class CopyBridge(nn.Module):
    """value -> K soft tokens -> gated injection. ~1M params."""

    def __init__(self, d_model: int, k_tokens: int = 8, d_hidden: int = 256,
                 gate_bias: float = 0.0):
        super().__init__()
        self.k = k_tokens
        self.enc1 = nn.Linear(33, d_hidden)      # Fourier features of v
        self.enc2 = nn.Linear(d_hidden, k_tokens * d_model // 8)
        self.proj = nn.Linear(d_model // 8, d_model)
        self.d_model = d_model
        self.inject = GatedCrossAttentionMLX(d_model, gate_bias=gate_bias)

    def tokens(self, v):
        ks = mx.arange(1, 17, dtype=mx.float32)
        feats = mx.concatenate([v[:, None],
                                mx.sin(ks[None, :] * v[:, None]),
                                mx.cos(ks[None, :] * v[:, None])], axis=-1)
        h = nn.gelu(self.enc1(feats))
        h = self.enc2(h).reshape(v.shape[0], self.k, self.d_model // 8)
        return self.proj(h)


def build_batch(tok, hf_tok, values, digit_ids):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    prompt = hf_tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False)
    if not isinstance(prompt, list):
        prompt = prompt["input_ids"]
    if prompt and isinstance(prompt[0], list):
        prompt = prompt[0]
    ids, labels = [], []
    for v in values:
        resp = f"The secret number is {v:.2f}."
        r = hf_tok.encode(resp, add_special_tokens=False) + [hf_tok.eos_token_id]
        ids.append(list(prompt) + r)
        labels.append([-100] * len(prompt) + r)
    L = max(len(x) for x in ids)
    pad = hf_tok.pad_token_id or hf_tok.eos_token_id
    I = np.full((len(ids), L), pad, dtype=np.int32)
    LB = np.full((len(ids), L), -100, dtype=np.int32)
    A = np.zeros((len(ids), L), dtype=np.int32)
    for i, (a, b) in enumerate(zip(ids, labels)):
        I[i, :len(a)] = a; LB[i, :len(b)] = b; A[i, :len(a)] = 1
    return mx.array(I), mx.array(LB), mx.array(A), len(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--l-rev-frac", type=float, default=0.625)
    args = ap.parse_args()

    patch_moe_gather()
    model, tok = mlx_lm.load(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    model.freeze()
    n_layers = len(model.model.layers)
    l_rev = round(n_layers * args.l_rev_frac)
    d_model = model.args.hidden_size
    bridge = CopyBridge(d_model)
    n_p = sum(v.size for _, v in tree_flatten(bridge.parameters()))
    print(f"{args.model}: layers={n_layers} hidden={d_model} inject@{l_rev} "
          f"bridge={n_p/1e6:.2f}M", flush=True)

    digit_ids = []
    for t in [str(d) for d in range(10)] + [".", "-"]:
        e = hf_tok.encode(t, add_special_tokens=False)
        if len(e) == 1:
            digit_ids.append(e[0])
    digit_ids = mx.array(sorted(set(digit_ids)), dtype=mx.int64)

    opt = optim.AdamW(learning_rate=args.lr)
    rng = random.Random(0)

    def loss_fn(bridge_, ids, labels, attn, values):
        toks = bridge_.tokens(values)
        s = MlxStream(model, ids, attn)
        s.run(0, l_rev)
        s.hidden, sigma = bridge_.inject(s.hidden, toks)
        s.run(l_rev, n_layers)
        logits = s.finish()
        return cross_entropy_masked(logits, labels, digit_ids, 5.0), sigma.mean()

    lag = nn.value_and_grad(bridge, loss_fn)
    log = Path(f"results/copy_probe/{args.tag}.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        vals = [round(rng.uniform(-1, 1), 2) for _ in range(args.batch)]
        ids, labels, attn, plen = build_batch(tok, hf_tok, vals, digit_ids)
        (loss, sigma), grads = lag(bridge, ids, labels, attn, mx.array(vals, dtype=mx.float32))
        grads, _ = clip_grad_norm(grads, 1.0)
        opt.update(bridge, grads)
        mx.eval(bridge.parameters(), opt.state)
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "loss": round(loss.item(), 4),
                   "gate": round(sigma.item(), 4),
                   "sec_per_step": round((time.time() - t0) / (step + 1), 2)}
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

    # greedy eval: can it say the injected number?
    correct, errs = 0, []
    rng_e = random.Random(999)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    prompt = hf_tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False)
    if not isinstance(prompt, list):
        prompt = prompt["input_ids"]
    if prompt and isinstance(prompt[0], list):
        prompt = prompt[0]
    for _ in range(24):
        v = round(rng_e.uniform(-1, 1), 2)
        toks = bridge.tokens(mx.array([v], dtype=mx.float32))
        ids = list(prompt)
        for _ in range(16):
            s = MlxStream(model, mx.array([ids]))
            s.run(0, l_rev)
            s.hidden, _ = bridge.inject(s.hidden, toks)
            s.run(l_rev, n_layers)
            nxt = int(s.finish()[:, -1].argmax(-1).item())
            ids.append(nxt)
            if nxt == hf_tok.eos_token_id:
                break
        out = hf_tok.decode(ids[len(prompt):], skip_special_tokens=True)
        m = re.findall(r"-?\d+\.\d+", out)
        pred = float(m[-1]) if m else None
        ok = pred is not None and abs(pred - v) <= 0.05
        correct += ok
        if pred is not None:
            errs.append(abs(pred - v))
    acc = correct / 24
    mae = sum(errs) / len(errs) if errs else None
    res = {"model": args.model, "acc": acc, "mae": mae, "n": 24, "steps": args.steps}
    Path(f"results/copy_probe/{args.tag}_result.json").write_text(json.dumps(res, indent=1))
    print(f"COPY PROBE {args.tag}: acc={acc:.2f} mae={mae}", flush=True)


if __name__ == "__main__":
    main()
