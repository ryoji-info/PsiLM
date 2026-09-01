"""Stage B of the reconstruction: can the forward READOUT read x0 from hidden states?

The copy probe proved the injection channel works at 8B. This probe isolates
the other half: no physics, no injection, no answer loss — just the pooled
attention readout over prompt hidden states, trained solely to classify x0
from the real PsiLM question. If this fails where the copy probe succeeded,
the readout is conclusively the fault, and the variants below say why.

Variants (--mode):
  pooled     current design: learned query attention-pools the prompt
  supervised pooled + attention-mass supervision on the x0 span (v3 fix)
  lastpos    read the final prompt position only (no pooling at all)
  spanmean   mean of the x0 token span (oracle pooling: upper bound)

Usage:
  python eval/readout_probe.py --model mlx-community/Qwen3-8B-4bit \
      --hf-tokenizer Qwen/Qwen3-8B --tag 8b --mode pooled --steps 400
"""

import argparse
import json
import math
import random
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

from psilm.mlx.bridges import _rms  # noqa: E402
from psilm.mlx.moe_patch import patch_moe_gather  # noqa: E402
from psilm.mlx.staged import MlxStream  # noqa: E402
from psilm.stage2.qa import QABuilder  # noqa: E402

N_BINS = 100


class Readout(nn.Module):
    def __init__(self, d_model: int, mode: str, d_hidden: int = 256):
        super().__init__()
        self.mode = mode
        self.query = mx.random.normal((d_model,)) / math.sqrt(d_model)
        self.key = nn.Linear(d_model, d_model)
        self.h1 = nn.Linear(d_model, d_hidden)
        self.h2 = nn.Linear(d_hidden, N_BINS)

    def __call__(self, hidden, prompt_mask, span, plen):
        h = _rms(hidden.astype(mx.float32))
        B, L, _ = h.shape
        pos = mx.arange(L)[None, :]
        if self.mode in ("pooled", "supervised"):
            scores = (self.key(h) * self.query).sum(-1) / math.sqrt(h.shape[-1])
            scores = mx.where(prompt_mask, scores, mx.array(-1e9, dtype=scores.dtype))
            w = mx.softmax(scores, axis=-1)
        elif self.mode == "lastpos":
            w = (pos == (plen[:, None] - 1)).astype(mx.float32)
        elif self.mode == "spanmean":
            m = ((pos >= span[:, 0:1]) & (pos < span[:, 1:2])).astype(mx.float32)
            w = m / m.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(self.mode)
        pooled = (w[..., None] * h).sum(axis=1)
        return self.h2(nn.gelu(self.h1(pooled))), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="pooled",
                    choices=["pooled", "supervised", "lastpos", "spanmean"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layer-frac", type=float, default=10 / 24)
    args = ap.parse_args()

    patch_moe_gather()
    model, tok = mlx_lm.load(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    model.freeze()
    n_layers = len(model.model.layers)
    l_fwd = round(n_layers * args.layer_frac)
    d_model = model.args.hidden_size
    net = Readout(d_model, args.mode)
    n_p = sum(v.size for _, v in tree_flatten(net.parameters()))
    print(f"{args.model} mode={args.mode}: read@{l_fwd}/{n_layers} "
          f"params={n_p/1e6:.2f}M", flush=True)

    builder = QABuilder(hf_tok)
    items = json.loads(Path("data/stage2_qa_train.json").read_text())
    val = json.loads(Path("data/stage2_qa_val.json").read_text())
    rng = random.Random(0)

    def make(batch_items):
        exs = [builder.build(it) for it in batch_items]
        plens = [e["prompt_len"] for e in exs]
        L = max(plens)
        pad = hf_tok.pad_token_id or hf_tok.eos_token_id
        I = np.full((len(exs), L), pad, dtype=np.int32)
        A = np.zeros((len(exs), L), dtype=np.int32)
        for i, e in enumerate(exs):
            p = e["p_ids"][: e["prompt_len"]]
            I[i, : len(p)] = p
            A[i, : len(p)] = 1
        return (mx.array(I), mx.array(A), mx.array(A.astype(bool)),
                mx.array(np.array([e["x0_span"] for e in exs], dtype=np.int32)),
                mx.array(np.array(plens, dtype=np.int32)),
                mx.array(np.array([round(e["meta"]["x0"] * 100) for e in exs],
                                  dtype=np.int32).clip(0, 99)))

    opt = optim.AdamW(learning_rate=args.lr)

    def loss_fn(net_, ids, attn, pmask, span, plen, tgt):
        s = MlxStream(model, ids, attn)
        s.run(0, l_fwd)
        logits, w = net_(s.hidden, pmask, span, plen)
        loss = nn.losses.cross_entropy(logits, tgt, reduction="mean")
        if net_.mode == "supervised":
            pos = mx.arange(w.shape[1])[None, :]
            in_span = (pos >= span[:, 0:1]) & (pos < span[:, 1:2])
            loss = loss + 0.5 * -mx.log((w * in_span).sum(axis=-1) + 1e-6).mean()
        bins = mx.arange(N_BINS, dtype=mx.float32) / N_BINS
        x0_hat = mx.softmax(logits, axis=-1) @ bins
        err = mx.abs(x0_hat - tgt.astype(mx.float32) / 100).mean()
        return loss, err

    lag = nn.value_and_grad(net, loss_fn)
    log = Path(f"results/readout_probe/{args.tag}_{args.mode}.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        b = make(rng.sample(items, args.batch))
        (loss, err), grads = lag(net, *b)
        grads, _ = clip_grad_norm(grads, 1.0)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "loss": round(loss.item(), 4),
                   "x0_err": round(err.item(), 4),
                   "sec_per_step": round((time.time() - t0) / (step + 1), 2)}
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

    # held-out exact-bin accuracy
    correct, errs = 0, []
    for i in range(0, 48, args.batch):
        b = make(val[i:i + args.batch])
        s = MlxStream(model, b[0], b[1])
        s.run(0, l_fwd)
        logits, _ = net(s.hidden, b[2], b[3], b[4])
        pred = logits.argmax(-1)
        correct += int((pred == b[5]).sum().item())
        errs.append(mx.abs(pred - b[5]).astype(mx.float32).mean().item() / 100)
    res = {"model": args.model, "mode": args.mode, "steps": args.steps,
           "exact_bin_acc": round(correct / 48, 4),
           "mean_abs_err": round(sum(errs) / len(errs), 4)}
    Path(f"results/readout_probe/{args.tag}_{args.mode}_result.json").write_text(json.dumps(res, indent=1))
    print(f"READOUT {args.tag}/{args.mode}: exact={res['exact_bin_acc']:.2f} "
          f"err={res['mean_abs_err']:.4f}", flush=True)


if __name__ == "__main__":
    main()
