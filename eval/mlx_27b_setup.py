"""Qwen3.8-27B (Qwen3.5 VLM language tower) setup + parity smoke test.

Mirror of eval/mlx_8b_setup.py for the flagship backbone, loaded through
psilm.mlx.vlm_loader.load_language_tower (mlx_lm's qwen3_5 module, text
tower only).

  !! Never run this while another model occupies the GPU: the 4-bit language
  !! tower alone is ~15.1 GB on a 24 GB machine whose Metal recommended
  !! working set is ~19.1 GB.  The script refuses to load when the budget
  !! exceeds the working set unless --force is given.

Stages (each prints, the whole thing ends with "27B SETUP OK"):
  0. config summary + memory budget (CPU, safetensors headers only)
  1. tiny-model CPU self-test of the loader (mask routing, parity, gradients)
  2. load the tower (+ tokenizer)                               [--estimate-only stops before]
  3. parity A: staged MlxStream vs one-shot forward, eval regime (Metal kernel
     recurrence on every linear layer) -- expected bit-exact or ~1e-2 bf16
  4. right-padded batch parity (validates the SSM-mask routing on real weights)
  5. parity B: differentiable regime (ops-path recurrence for layers >= l_rev)
  6. a few coupled training steps with the real bridges, batch 1  [--skip-train-step]

Usage:
  python eval/mlx_27b_setup.py --estimate-only
  python eval/mlx_27b_setup.py                      # full smoke test
  python eval/mlx_27b_setup.py --skip-train-step --prompt "The capital of France is"
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from psilm.mlx.staged import MlxStream  # noqa: E402
from psilm.mlx.vlm_loader import (  # noqa: E402
    DEFAULT_REPO, GB, checkpoint_bytes, default_coupling, describe, format_estimate,
    hf_tokenizer_path, load_language_tower, memory_estimate, read_config, selftest_cpu,
)


def _active_gb():
    for name in ("get_active_memory",):
        f = getattr(mx, name, None)
        if f is not None:
            return f() / GB
    return float("nan")


def _peak_gb():
    f = getattr(mx, "get_peak_memory", None)
    return f() / GB if f is not None else float("nan")


def staged_logits(tower, ids, attn=None):
    st = MlxStream(tower, ids, attn)
    st.run(0, tower.n_layers)
    return st.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tol", type=float, default=2e-2, help="max |logit diff| accepted (8B script: 2e-2)")
    ap.add_argument("--batch", type=int, default=1, help="batch for the train-step stage (memory!)")
    ap.add_argument("--seq-len", type=int, default=128, help="padded QA length used in the budget (QA prompts: 122-126 tokens)")
    ap.add_argument("--l-rev", type=int, default=None, help="injection layer; default = PsiLMMLX's round(n*15/24)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--skip-train-step", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true", help="disable per-layer mx.checkpoint in the differentiable regime")
    ap.add_argument("--force", action="store_true", help="load even if the budget exceeds the Metal working set")
    args = ap.parse_args()

    # ---- 0. config + budget (no weights) --------------------------------------
    cfg = read_config(args.repo)
    d = describe(cfg)
    n = d["num_layers"]
    l_fwd, l_rev_default = default_coupling(n)
    l_rev = args.l_rev if args.l_rev is not None else l_rev_default
    print(f"backbone {args.repo}: {d['model_type']} {d['architectures']}", flush=True)
    print(f"  hidden {d['hidden_size']}  layers {n} (linear {d['n_linear']}, full {d['n_full']} at "
          f"{d['full_attention_indices'][:4]}...)  vocab {d['vocab_size']}  tied {d['tie_word_embeddings']}  "
          f"heads {d['num_attention_heads']}/{d['num_key_value_heads']} x {d['head_dim']} (rope {d['rope_dims']} dims)  "
          f"quant {d['quantization']}", flush=True)
    print(f"  coupling l_fwd={l_fwd} l_rev={l_rev} of {n}  (post-injection layers: "
          f"{n - l_rev}, linear among them: {sum(t == 'linear_attention' for t in d['layer_types'][l_rev:])})", flush=True)
    cb = checkpoint_bytes(args.repo)
    print(f"  checkpoint: language tower {cb['language_tower'] / GB:.3f} GB, vision tower {cb['vision_tower'] / GB:.3f} GB "
          f"(dropped), {cb['n_files']} shards", flush=True)
    est = memory_estimate(cfg, args.batch, args.seq_len, l_rev, weights_bytes=cb["language_tower"],
                          checkpoint=not args.no_checkpoint)
    print(format_estimate(est), flush=True)
    if args.batch > 1:
        est1 = memory_estimate(cfg, 1, args.seq_len, l_rev, weights_bytes=cb["language_tower"], checkpoint=True)
        print(f"  (batch 1 with checkpoint would need {est1['total'] / GB:.2f} GB)", flush=True)

    # ---- 1. CPU self-test of the loader -----------------------------------------
    if not args.no_selftest:
        t0 = time.time()
        selftest_cpu(verbose=False, repo_id=args.repo)
        print(f"loader CPU self-test OK ({time.time() - t0:.1f}s)", flush=True)

    if args.estimate_only:
        print("ESTIMATE ONLY -- no weights loaded", flush=True)
        return

    if not est["fits_working_set"] and not args.force:
        print(f"ABORT: budget {est['total'] / GB:.2f} GB exceeds the Metal working set "
              f"{est['device']['max_recommended_working_set'] / GB:.2f} GB; re-run with --force, a smaller "
              f"--batch, or a later --l-rev", flush=True)
        sys.exit(2)

    # ---- 2. load ------------------------------------------------------------------
    t0 = time.time()
    tower, tok = load_language_tower(args.repo)
    print(f"loaded in {time.time() - t0:.0f}s: {tower}  active {_active_gb():.2f} GB peak {_peak_gb():.2f} GB", flush=True)
    assert len(tower.model.layers) == n and tower.args.hidden_size == d["hidden_size"]
    assert hasattr(tower, "lm_head") == (not d["tie_word_embeddings"])

    # ---- 3. parity A (eval regime: Metal-kernel recurrence everywhere) -----------
    ids = mx.array([tok.encode(args.prompt)])
    t0 = time.time()
    ref = tower(ids)
    mx.eval(ref)
    t_ref = time.time() - t0
    t0 = time.time()
    out = staged_logits(tower, ids)
    mx.eval(out)
    t_staged = time.time() - t0
    diff = mx.abs(ref.astype(mx.float32) - out.astype(mx.float32)).max().item()
    nxt = tok.decode([int(ref[0, -1].argmax().item())])
    print(f"parity A (eval regime): {diff:.2e} {'(bit-exact)' if diff == 0 else ''}  "
          f"one-shot {t_ref:.2f}s staged {t_staged:.2f}s  next token {nxt!r}", flush=True)
    assert diff < args.tol

    # ---- 4. right-padded batch parity ------------------------------------------
    ids2 = mx.array([tok.encode(args.prompt + " Paris, and the capital of Italy is")])
    L2 = ids2.shape[1]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    padded = mx.concatenate([ids, mx.full((1, L2 - ids.shape[1]), pad_id, dtype=ids.dtype)], axis=1)
    batch = mx.concatenate([ids2, padded], axis=0)
    attn = mx.array([[1] * L2, [1] * ids.shape[1] + [0] * (L2 - ids.shape[1])], dtype=mx.int32)
    ref2 = tower(ids2)
    outb = staged_logits(tower, batch, attn)
    mx.eval(ref2, outb)
    d_full = mx.abs(ref2.astype(mx.float32) - outb[0:1].astype(mx.float32)).max().item()
    d_pad = mx.abs(ref.astype(mx.float32) - outb[1:2, :ids.shape[1]].astype(mx.float32)).max().item()
    print(f"padded-batch parity: unpadded row {d_full:.2e}, right-padded row {d_pad:.2e}", flush=True)
    assert max(d_full, d_pad) < args.tol

    # ---- 5. parity B (differentiable regime for layers >= l_rev) ------------------
    tower.set_differentiable(l_rev, checkpoint=not args.no_checkpoint)
    print(f"regime: {tower.regime()}", flush=True)
    t0 = time.time()
    out_d = staged_logits(tower, ids)
    mx.eval(out_d)
    t_d = time.time() - t0
    diff_d = mx.abs(ref.astype(mx.float32) - out_d.astype(mx.float32)).max().item()
    print(f"parity B (ops-path recurrence for layers >= {l_rev}): {diff_d:.2e}  staged {t_d:.2f}s", flush=True)
    assert diff_d < args.tol

    # ---- 6. coupled training steps ---------------------------------------------
    if not args.skip_train_step:
        from transformers import AutoTokenizer

        from eval.mlx_stage2_train import to_mlx_batch
        from psilm.mlx.bridges import PsiBridgesMLX
        from psilm.mlx.fno import convert_from_torch
        from psilm.mlx.model import PsiLMMLX
        from psilm.stage2.qa import QABuilder, make_batch as tmb

        hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer_path(args.repo))   # the 27B's OWN tokenizer
        fno = convert_from_torch("results/stage2/fno.pt")
        bridges = PsiBridgesMLX(d_model=tower.args.hidden_size)
        psi = PsiLMMLX(tower, tok, fno, bridges, l_rev=l_rev)
        assert psi.l_rev == l_rev and psi.n_layers == n
        builder = QABuilder(hf_tok)
        items = json.loads(Path("data/stage2_qa_train.json").read_text())
        opt = optim.AdamW(learning_rate=3e-4)

        def wrapped(b_, batch):
            psi.phi = b_
            return psi.loss_fn(batch)

        lag = nn.value_and_grad(bridges, wrapped)
        rng = random.Random(3)
        for step in range(args.steps):
            t0 = time.time()
            batch = to_mlx_batch(tmb(builder, rng.sample(items, args.batch), "cpu"))
            (loss, aux), grads = lag(bridges, batch)
            opt.update(bridges, grads)
            mx.eval(bridges.parameters(), opt.state)
            print(f"step {step}: loss={loss.item():.3f} loss_ans={aux[0].item():.3f} loss_x0={aux[2].item():.3f} "
                  f"x0_err={aux[4].item():.3f} gate={aux[5].item():.3f}  {time.time() - t0:.1f}s (batch {args.batch}, "
                  f"L={batch['p_ids'].shape[1]})  active {_active_gb():.2f} GB peak {_peak_gb():.2f} GB", flush=True)
        pred = psi.generate(builder, items[0], max_new=12)
        print(f"generate: {pred!r} (truth {items[0]['u']})", flush=True)

    print("27B SETUP OK", flush=True)


if __name__ == "__main__":
    main()
