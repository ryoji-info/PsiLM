"""Gemma 4 12B setup + parity smoke test (mirror of eval/mlx_8b_setup.py).

  1. load the text tower through psilm.mlx.gemma_loader
  2. parity: staged MlxStream vs the stock one-shot forward (expect bit-exact)
  3. right-padded batch parity (row 0 vs its unpadded forward)
  4. coupled training steps with the real bridges (value channel) at the
     requested batch sizes, with time and peak memory

Usage: python eval/mlx_gemma_setup.py [--repo mlx-community/gemma-4-12B-it-4bit] [--batches 4,8]
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "/Users/rxiii/Documents/GitHub/PsiLM")
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.optimizers import clip_grad_norm
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from psilm.mlx.staged import MlxStream
from psilm.mlx.fno import convert_from_torch
from psilm.mlx.bridges import PsiBridgesMLX
from psilm.mlx.model import PsiLMMLX
from psilm.mlx.gemma_loader import load_gemma_tower
from eval.mlx_stage2_train import to_mlx_batch
from psilm.stage2.qa import QABuilder, make_batch as tmb


def peak_gb():
    try:
        return mx.get_peak_memory() / 1e9
    except AttributeError:
        return mx.metal.get_peak_memory() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mlx-community/gemma-4-12B-it-4bit")
    ap.add_argument("--batches", default="4,8")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    t0 = time.time()
    tower, tok = load_gemma_tower(args.repo)
    n = len(tower.model.layers)
    print(f"loaded in {time.time()-t0:.0f}s: layers={n} hidden={tower.args.hidden_size} "
          f"window={getattr(tower.args, 'sliding_window', None)} softcap={tower._lm.final_logit_softcapping} "
          f"tied={tower.tie_word_embeddings}  peak {peak_gb():.2f} GB", flush=True)
    hf_dir = os.path.dirname(hf_hub_download(args.repo, "config.json"))
    hf_tok = AutoTokenizer.from_pretrained(hf_dir)

    # 2. parity
    ids = mx.array([hf_tok.encode("The capital of France is", add_special_tokens=True)])
    t1 = time.time(); ref = tower(ids); mx.eval(ref); t_ref = time.time() - t1
    t1 = time.time(); st = MlxStream(tower, ids); st.run(0, n); got = st.finish(); mx.eval(got); t_st = time.time() - t1
    diff = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
    nxt = hf_tok.decode([int(got[0, -1].argmax())])
    print(f"parity A: max|diff| {diff:.2e}  one-shot {t_ref:.2f}s staged {t_st:.2f}s  next token {nxt!r}", flush=True)
    assert diff < args.tol, "staged forward is not the stock model"

    # 3. right-padded batch parity
    ids2 = mx.array([hf_tok.encode("Water boils at one hundred degrees", add_special_tokens=True)])
    L = max(ids.shape[1], ids2.shape[1]) + 3
    pad = hf_tok.pad_token_id or 0
    def padrow(x):
        return mx.concatenate([x, mx.full((1, L - x.shape[1]), pad, dtype=x.dtype)], axis=1)
    batch = mx.concatenate([padrow(ids), padrow(ids2)], axis=0)
    attn = mx.array([[1] * ids.shape[1] + [0] * (L - ids.shape[1]), [1] * ids2.shape[1] + [0] * (L - ids2.shape[1])], dtype=mx.int32)
    sb = MlxStream(tower, batch, attn); sb.run(0, n); lg = sb.finish()
    d0 = float(mx.abs(lg[0, ids.shape[1]-1].astype(mx.float32) - ref[0, -1].astype(mx.float32)).max())
    ref2 = tower(ids2)
    d1 = float(mx.abs(lg[1, ids2.shape[1]-1].astype(mx.float32) - ref2[0, -1].astype(mx.float32)).max())
    print(f"padded-batch parity: row0 {d0:.2e} row1 {d1:.2e}", flush=True)
    assert max(d0, d1) < args.tol

    # 4. coupled training steps
    fno = convert_from_torch("results/stage2/fno.pt")
    builder = QABuilder(hf_tok)
    items = json.loads(open("data/stage2_qa_train.json").read())
    for B in [int(b) for b in args.batches.split(",")]:
        bridges = PsiBridgesMLX(d_model=tower.args.hidden_size, gate_bias=0.0, inj_cap=0.2, channel="value")
        psi = PsiLMMLX(tower, tok, fno, bridges); psi.detach_x0 = True
        opt = optim.AdamW(learning_rate=3e-4, bias_correction=True)
        def wrapped(b_, batch):
            psi.phi = b_
            return psi.loss_fn(batch)
        lag = nn.value_and_grad(bridges, wrapped)
        rng = random.Random(3)
        mx.reset_peak_memory() if hasattr(mx, "reset_peak_memory") else None
        for step in range(args.steps):
            t1 = time.time()
            batch = to_mlx_batch(tmb(builder, rng.sample(items, B), "cpu"))
            (loss, aux), grads = lag(bridges, batch)
            grads = {k: clip_grad_norm(g, 1.0)[0] for k, g in grads.items()}
            opt.update(bridges, grads); mx.eval(bridges.parameters(), opt.state)
            print(f"batch {B} step {step}: loss={loss.item():.3f} loss_ans={aux[0].item():.3f} x0_err={aux[4].item():.3f} "
                  f"gate={aux[5].item():.3f} {time.time()-t1:.1f}s  peak {peak_gb():.2f} GB  couple {psi.l_fwd}/{psi.l_rev} of {n}", flush=True)
    print("GEMMA SETUP OK", flush=True)


if __name__ == "__main__":
    main()
