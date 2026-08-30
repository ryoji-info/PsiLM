"""Train PsiLM bridges over MLX (4-bit-capable backbones). Chunked, resumable.

Usage:
  python eval/mlx_stage2_train.py --steps 1000 --fresh \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --tag _mlx0.5b4
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.optimizers import clip_grad_norm  # noqa: E402
import mlx_lm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.bridges import PsiBridgesMLX  # noqa: E402
from psilm.mlx.fno import convert_from_torch  # noqa: E402
from psilm.mlx.model import PsiLMMLX  # noqa: E402
from psilm.stage2.qa import QABuilder, make_batch as torch_make_batch  # noqa: E402


def to_mlx_batch(tb):
    return {
        "p_ids": mx.array(tb["p_ids"].numpy()),
        "p_labels": mx.array(tb["p_labels"].numpy()),
        "p_attn": mx.array(tb["p_attn"].numpy().astype(np.int32)),
        "prompt_mask": mx.array(tb["prompt_mask"].numpy()),
        "params": mx.array(tb["params"].numpy()),
        "u_true": mx.array(tb["u_true"].numpy()),
        "x0": mx.array(tb["x0"].numpy()),
        "x0_bins": mx.array((tb["x0"].numpy() * 100).round().astype(np.int32).clip(0, 99)),
    }


def parse_value(text):
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def rollout_eval(psi, builder, items, n=12):
    correct, errs = 0, []
    for item in items[:n]:
        pred = parse_value(psi.generate(builder, item))
        ok = pred is not None and abs(pred - item["u"]) <= 0.05
        correct += ok
        if pred is not None:
            errs.append(abs(pred - item["u"]))
    return correct / n, (sum(errs) / len(errs) if errs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF tokenizer for the QA builder (matches the base model)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ckpt = Path(f"results/stage2{args.tag}/bridges.npz")
    log = Path(f"results/stage2{args.tag}/train_log.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    model, tok = mlx_lm.load(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    fno = convert_from_torch("results/stage2/fno.pt")
    d_model = model.model.embed_tokens.weight.shape[-1] if not hasattr(
        model.model.embed_tokens, "scales") else model.args.hidden_size
    bridges = PsiBridgesMLX(d_model=model.args.hidden_size)
    opt = optim.AdamW(learning_rate=args.lr)

    global_step = 0
    if ckpt.exists() and not args.fresh:
        bridges.load_weights(str(ckpt))
        meta = json.loads(Path(str(ckpt) + ".meta").read_text())
        global_step = meta["step"]
        print(f"resumed at step {global_step} (optimizer state fresh)")

    psi = PsiLMMLX(model, tok, fno, bridges)
    builder = QABuilder(hf_tok)
    train_items = json.loads(Path("data/stage2_qa_train.json").read_text())
    val_items = json.loads(Path("data/stage2_qa_val.json").read_text())
    n_params = sum(v.size for _, v in tree_flatten(bridges.parameters()))
    print(f"bridges: {n_params/1e6:.2f}M | backbone: {args.model} | "
          f"coupling {psi.l_fwd}/{psi.l_rev} of {psi.n_layers}")

    def wrapped(bridges_, batch):
        psi.phi = bridges_
        return psi.loss_fn(batch)

    loss_and_grad = nn.value_and_grad(bridges, wrapped)

    t0 = time.time()
    for i in range(args.steps):
        rng = random.Random(21_000_000 + global_step)
        batch = to_mlx_batch(torch_make_batch(builder, rng.sample(train_items, args.batch), "cpu"))
        (loss, aux), grads = loss_and_grad(bridges, batch)
        grads, _ = clip_grad_norm(grads, 1.0)
        opt.update(bridges, grads)
        mx.eval(bridges.parameters(), opt.state)
        global_step += 1
        if global_step % 25 == 0:
            rec = {"step": global_step,
                   "loss_ans": round(aux[0].item(), 4),
                   "loss_param": round(aux[1].item(), 5),
                   "loss_x0": round(aux[2].item(), 4),
                   "loss_u": round(aux[3].item(), 5),
                   "x0_err": round(aux[4].item(), 4),
                   "gate": round(aux[5].item(), 4),
                   "sec_per_step": round((time.time() - t0) / (i + 1), 2)}
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

    acc, mae = rollout_eval(psi, builder, val_items)
    bridges.save_weights(str(ckpt))
    Path(str(ckpt) + ".meta").write_text(json.dumps({"step": global_step, "model": args.model}))
    with log.open("a") as f:
        f.write(json.dumps({"step": global_step, "eval_acc": acc, "eval_mae": mae}) + "\n")
    print(f"CHUNK DONE step={global_step} acc@0.05={acc:.2f} mae={mae}")


if __name__ == "__main__":
    main()
