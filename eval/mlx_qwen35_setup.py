"""Qwen3.5 9B setup + parity record (mirror of eval/mlx_gemma_setup.py).

The adapter's two claims -- exact staged parity, and a differentiable scan that
agrees with the Metal kernel -- were first recorded only in commit dadfd9f.
This puts them on disk, where the paper cites them, and adds what the commit
did not measure: parity on a right-padded row, which is the one case where the
adapter's boolean mask does something the stock forward does not.

  1. load the text tower through psilm.mlx.gemma_loader.load_backbone_any
  2. parity A: staged MlxStream vs the stock one-shot forward, batch 1, kernel
     path (expect max|diff| 0.0)
  3. padding parity: one row with trailing pads vs its unpadded forward at every
     real position (exercises the shim's mask derivation; the stock forward has
     no padding mask, so this has no stock reference and compares staged to
     staged)
  4. ops-path vs kernel: the same prompt with every layer on the pure-MLX scan
     (set_grad_window(0)) against the kernel; max relative logit difference,
     argmax and top-5 agreement at every position
  5. optionally, coupled training steps at the requested batch sizes

Usage: python eval/mlx_qwen35_setup.py --model /path/to/qwen3.5-9b-mlx [--batches 2 --steps 3]
"""
import argparse
import json
import random
import sys
import time

sys.path.insert(0, "/Users/rxiii/Documents/GitHub/PsiLM")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.optimizers import clip_grad_norm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.staged import MlxStream  # noqa: E402
from psilm.mlx.fno import convert_from_torch  # noqa: E402
from psilm.mlx.bridges import PsiBridgesMLX  # noqa: E402
from psilm.mlx.model import PsiLMMLX  # noqa: E402
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from eval.mlx_stage2_train import to_mlx_batch  # noqa: E402
from psilm.stage2.qa import QABuilder, make_batch as tmb  # noqa: E402


def peak_gb():
    return mx.get_peak_memory() / 2**30


def staged_logits(tower, ids, attn=None):
    st = MlxStream(tower, ids, attn)
    st.run(0, len(tower.model.layers))
    out = st.finish()
    mx.eval(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx")
    ap.add_argument("--batches", default="2")
    ap.add_argument("--steps", type=int, default=0, help="coupled steps per batch size (0: parity only)")
    ap.add_argument("--l-rev", type=int, default=26)
    ap.add_argument("--tol", type=float, default=1e-6, help="parity tolerance; this adapter is exact")
    args = ap.parse_args()

    t0 = time.time()
    tower, stock, tok = load_backbone_any(args.model)
    n = len(tower.model.layers)
    kinds = "".join("L" if getattr(l, "is_linear", False) else "F" for l in tower.model.layers)
    print(f"loaded in {time.time()-t0:.0f}s: layers={n} hidden={tower.args.hidden_size} "
          f"kinds={kinds} tied={tower.tie_word_embeddings}  peak {peak_gb():.2f} GB", flush=True)
    hf_tok = AutoTokenizer.from_pretrained(args.model)

    def chat(u):
        out = hf_tok.apply_chat_template([{"role": "user", "content": u}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)
        return out["input_ids"] if not isinstance(out, list) else out

    # 2. parity A, kernel path
    tower.set_grad_window(n)            # every layer on the kernel
    a = chat("The capital of France is")
    ids = mx.array([a])
    t1 = time.time(); ref = stock(ids); mx.eval(ref); t_ref = time.time() - t1
    t1 = time.time(); got = staged_logits(tower, ids); t_st = time.time() - t1
    diff = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
    nxt = hf_tok.decode([int(got[0, -1].argmax())])
    print(f"parity A (batch 1, {len(a)} tokens, kernel): max|diff| {diff:.3e}  "
          f"one-shot {t_ref:.2f}s staged {t_st:.2f}s  next token {nxt!r}", flush=True)
    assert diff <= args.tol, "staged forward is not the stock model"

    # 3. padding parity, staged vs staged: real positions must not see the pads
    pad = hf_tok.pad_token_id or 0
    apad = mx.array([a + [pad] * 3]); attn = mx.array([[1] * len(a) + [0] * 3], dtype=mx.int32)
    lp = staged_logits(tower, apad, attn)
    dpad = float(mx.abs(lp[0, :len(a)].astype(mx.float32) - got[0].astype(mx.float32)).max())
    print(f"padding parity (batch 1, 3 trailing pads, kernel): max|diff| at real positions {dpad:.3e}", flush=True)
    assert dpad <= args.tol, "padding leaks into real positions"

    # 4. ops path vs kernel, same prompt, every layer switched
    tower.set_grad_window(0)
    ops = staged_logits(tower, ids)
    tower.set_grad_window(n)
    r32, o32 = got.astype(mx.float32)[0], ops.astype(mx.float32)[0]
    rel = float((mx.abs(r32 - o32).max() / mx.abs(r32).max()))
    argmax_agree = float((r32.argmax(-1) == o32.argmax(-1)).mean())
    top5_k = mx.argpartition(-r32, 5, axis=-1)[:, :5]
    top5_o = mx.argpartition(-o32, 5, axis=-1)[:, :5]
    top5 = float(mx.mean(mx.array([len(set(map(int, top5_k[i].tolist())) & set(map(int, top5_o[i].tolist()))) / 5
                                    for i in range(r32.shape[0])])))
    print(f"ops path vs kernel (all {n} layers): max relative logit diff {rel:.3e}, "
          f"argmax agreement {argmax_agree:.3f}, mean top-5 overlap {top5:.3f} over {r32.shape[0]} positions", flush=True)

    # QA-length prompt, row-count regime (informational, as in the Gemma script)
    items = json.loads(open("data/stage2_qa_train.json").read())
    p = QABuilder(hf_tok).prompt_ids(items[0])
    l1 = stock(mx.array([p]))[0]; l8 = stock(mx.array([p] * 8))[0]
    print(f"row-count regime (QA prompt {len(p)} tokens): batch1 vs batch8 max|diff| "
          f"{float(mx.abs(l1.astype(mx.float32) - l8.astype(mx.float32)).max()):.2f}, "
          f"argmax agree {float((l1.argmax(-1) == l8.argmax(-1)).mean()):.3f}  [informational]", flush=True)

    # 5. coupled steps
    if args.steps:
        tower.set_grad_window(args.l_rev)
        fno = convert_from_torch("results/stage2/fno.pt")
        builder = QABuilder(hf_tok)
        for B in [int(b) for b in args.batches.split(",")]:
            bridges = PsiBridgesMLX(d_model=tower.args.hidden_size, gate_bias=0.0, inj_cap=0.2,
                                    channel="value", readout_norm="dim")
            psi = PsiLMMLX(tower, tok, fno, bridges, l_rev=args.l_rev); psi.detach_x0 = True
            opt = optim.AdamW(learning_rate=3e-4, bias_correction=True)
            def wrapped(b_, batch):
                psi.phi = b_
                return psi.loss_fn(batch)
            lag = nn.value_and_grad(bridges, wrapped)
            rng = random.Random(3)
            mx.reset_peak_memory()
            for step in range(args.steps):
                t1 = time.time()
                batch = to_mlx_batch(tmb(builder, rng.sample(items, B), "cpu"))
                (loss, aux), grads = lag(bridges, batch)
                grads = {k: clip_grad_norm(g, 1.0)[0] for k, g in grads.items()}
                opt.update(bridges, grads); mx.eval(bridges.parameters(), opt.state)
                print(f"batch {B} step {step}: loss={loss.item():.3f} {time.time()-t1:.1f}s  "
                      f"peak {peak_gb():.2f} GB  couple {psi.l_fwd}/{psi.l_rev} of {n}", flush=True)
    print("QWEN35 SETUP OK", flush=True)


if __name__ == "__main__":
    main()
