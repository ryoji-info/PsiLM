"""Train the constitution bridges over MLX. Chunked, resumable.

Self-distillation: the TEACHER is the frozen backbone reading a constitution
excerpt as its system prompt, the STUDENT is the same frozen backbone reading the
plain system prompt with the bridges attached, and the loss is CE on the
teacher's greedy continuation (agent A3b's data files). The no-harm arm is
PsiLM's: off-task prompts (GSM8K/MMLU) with the backbone's OWN continuation,
gate-only updates and a gate penalty, so the channel has to shut where the
constitution is irrelevant instead of paying for itself everywhere.

The injection writes only into the base model's value neurons, and which dims
those are is a training-time choice (--write-dims), so the write-location
ablation is three runs of this script that differ in one flag:

  --write-dims results/value_neurons/<tag>/layer15.json     # the value neurons
  --write-dims random:0:9                                   # same count, random
  --write-dims all                                          # the whole stream

Usage:
  python eval/mlx_constitution_train.py --tag qwen0.5b --fresh --steps 200 \
      --data data/constitution_train_qwen0.5b.json \
      --val data/constitution_val_qwen0.5b.json \
      --noharm-data data/noharm_train.json \
      --const-model results/constitution_model/qwen2.5-0.5b-constitution \
      --write-dims results/value_neurons/qwen0.5b/layer15.json
  python eval/mlx_constitution_train.py --self-test        # tiny model, CPU, no weights
"""

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.optimizers import clip_grad_norm  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402

from psilm.mlx.constitution import (  # noqa: E402
    ConstitutionBridgesMLX, ConstitutionModelMLX, ConstStack, PsiConstitutionMLX,
    dims_label, load_constitution_bridge_weights, pad_batch, parse_dims, stack_meta)
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from psilm.mlx.staged import MlxStream  # noqa: E402

#: flags whose value is baked into a checkpoint's numerics; a resume that changes
#: one of them is almost always a mistake, so say so
RESUME_WARN = ("const_model", "l_fwd", "l_rev", "k_fwd", "m_tokens", "gate_bias",
               "inj_cap", "readout_norm", "write_dims", "read_dims", "lam_gate",
               "clip", "noharm_gate_only", "noharm_every", "lr")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--const-model",
                    default="results/constitution_model/qwen2.5-0.5b-constitution",
                    help="mlx_lm-loadable directory of the frozen constitution model")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--l-fwd", type=int, default=None,
                    help="read layer (default round(n*10/24), as PsiLMMLX)")
    ap.add_argument("--l-rev", type=int, default=None,
                    help="inject layer (default round(n*15/24))")
    ap.add_argument("--write-dims", default="all",
                    help="where the injection may write: all | <value_neurons.json>[:top5pct|:topN] "
                         "| random:<seed>:<n>. The ablation arm is chosen here.")
    ap.add_argument("--read-dims", default="all",
                    help="what the forward bridge may read (same spellings); the strict "
                         "variant reads only the value neurons")
    ap.add_argument("--k-fwd", type=int, default=8, help="soft tokens into the constitution model")
    ap.add_argument("--m-tokens", type=int, default=8, help="readout positions / injected tokens")
    ap.add_argument("--gate-bias", type=float, default=0.0,
                    help="bias of the gate's output layer: 0.0 starts the channel half "
                         "open (the positive arm wants it on; the no-harm arm shuts it "
                         "off-task), -2.0 starts it closed as in the physics runs")
    ap.add_argument("--inj-cap", type=float, default=0.2,
                    help="cap the injection RMS at this fraction of the stream RMS over "
                         "the WRITTEN dims (pre-gate)")
    ap.add_argument("--clip", default="module", choices=["global", "module"],
                    help="module: clip fwd/rev/inject at 1.0 separately")
    ap.add_argument("--readout-norm", default="dim", choices=["rms", "dim"])
    ap.add_argument("--calib-n", type=int, default=32,
                    help="training prompts used to calibrate the readout on a fresh start")
    ap.add_argument("--data", default=None,
                    help="default data/constitution_train_<tag>.json")
    ap.add_argument("--val", default=None,
                    help="default data/constitution_val_<tag>.json")
    ap.add_argument("--noharm-data", default=None,
                    help="data/noharm_*.json: off-task prompts with the backbone's own "
                         "continuations; enables the gate-selectivity arm")
    ap.add_argument("--noharm-every", type=int, default=2,
                    help="every k-th step is a no-harm batch (2 = alternate 1:1; >= 2)")
    ap.add_argument("--noharm-gate-only", type=int, default=1,
                    help="1: on no-harm steps only inject.g1/g2 receive the no-harm gradient "
                         "(AdamW momentum and weight decay still move every tensor on every step)")
    ap.add_argument("--lam-gate", type=float, default=1.0,
                    help="weight of the mean-gate penalty on no-harm steps")
    ap.add_argument("--eval-n", type=int, default=32,
                    help="val items scored teacher-forced at the end of the chunk")
    ap.add_argument("--checkpoint-from", type=int, default=None, metavar="L",
                    help="recompute layers >= L in the backward pass instead of taping "
                         "them (-1: use l_rev)")
    ap.add_argument("--self-test", action="store_true",
                    help="tiny synthetic model on the CPU: no weights, no GPU")
    return ap


# ----------------------------------------------------------------------------
# stack
# ----------------------------------------------------------------------------

def _pad_id(tok):
    pad = getattr(tok, "pad_token_id", None)
    return int(pad if pad is not None else tok.eos_token_id)


def build_stack(args, ckpt: Path) -> ConstStack:
    """Backbone + constitution model + bridges, resuming from ckpt unless --fresh.

    On a resume the module is rebuilt from the checkpoint's OWN resolved dim
    lists: a --write-dims spelling can resolve differently later (a re-ranked
    value-neuron file, a lost seed), and silently continuing into a different
    subspace would invalidate the run without a single warning.
    """
    from transformers import AutoTokenizer
    model, _stock, tok = load_backbone_any(args.model)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    d_model = int(model.args.hidden_size)

    prev, step = {}, 0
    meta_path = Path(str(ckpt) + ".meta")
    resuming = ckpt.exists() and meta_path.exists() and not args.fresh
    if resuming:
        prev = json.loads(meta_path.read_text())
        step = int(prev.get("step", 0))

    const = ConstitutionModelMLX(args.const_model, m_tokens=args.m_tokens)
    write_idx = parse_dims(args.write_dims, d_model)
    read_idx = parse_dims(args.read_dims, d_model)
    if resuming:
        ck_write, ck_read = prev.get("write_dims"), prev.get("read_dims")
        cur_write = write_idx if write_idx is not None else list(range(d_model))
        if ck_write is not None and list(ck_write) != cur_write:
            how = (f"{len(cur_write)} dims against the checkpoint's {len(ck_write)}"
                   if len(ck_write) != len(cur_write)
                   else f"a different set of {len(cur_write)} dims")
            raise SystemExit(
                f"--write-dims {args.write_dims!r} resolves to {how} (trained with "
                f"{prev.get('args', {}).get('write_dims')!r}); pass the same spelling "
                f"or --fresh")
        if (ck_read or None) != (read_idx or None):
            raise SystemExit(f"--read-dims {args.read_dims!r} differs from the checkpoint's")
    bridges = ConstitutionBridgesMLX(
        d_model=d_model, d_const=const.d_const, write_idx=write_idx, read_idx=read_idx,
        k_fwd=args.k_fwd, m_tokens=const.n_readout, gate_bias=args.gate_bias,
        inj_cap=args.inj_cap, readout_norm=args.readout_norm, emb_rms=const.emb_rms)
    if resuming:
        ref = bridges.fwd.mlp2.weight
        load_constitution_bridge_weights(bridges, ckpt)
        assert float(mx.abs(bridges.fwd.mlp2.weight - ref).max().item()) > 0, \
            "the forward bridge did not load"
        mx.eval(bridges.parameters())
    meta = {"step": step, "resumed": resuming, "prev_args": prev.get("args", {})}
    return ConstStack(model=model, tok=tok, hf_tok=hf_tok, const=const,
                      bridges=bridges, meta=meta)


# ----------------------------------------------------------------------------
# evaluation at the end of a chunk
# ----------------------------------------------------------------------------

def eval_arm(psi, items, n: int, mode: str):
    """Teacher-forced CE and top-1 agreement to the teacher's continuation."""
    import numpy as np
    ce, ag, gp, gc, rc = [], [], [], [], []
    for it in items[:n]:
        c, a, st = psi.teacher_forced(it["prompt_ids"], it["teacher_ids"], mode)
        if c is None:
            continue
        ce.append(c)
        ag.append(a)
        for key, dst in (("prompt_mean", gp), ("cont_mean", gc), ("ratio_cont", rc)):
            if st.get(key) is not None:
                dst.append(st[key])
    f = lambda v: (round(float(np.mean(v)), 4) if v else None)   # noqa: E731
    return {"n": len(ce), "ce": f(ce), "agree": f(ag), "gate_prompt": f(gp),
            "gate_cont": f(gc), "ratio_cont": f(rc)}


def base_eval_cached(psi, items, n: int, out_dir: Path, key: str):
    """The base arm's CE/agreement never changes, so it is computed once per
    (val file, n, backbone) and kept next to the training log."""
    path = out_dir / "base_eval.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    if key not in cache:
        cache[key] = eval_arm(psi, items, n, "base")
        path.write_text(json.dumps(cache, indent=1))
    return cache[key]


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def run_training(args, stack: ConstStack, out_dir=None):
    model, tok, hf_tok = stack.model, stack.tok, stack.hf_tok
    const, bridges = stack.const, stack.bridges
    out_dir = Path(out_dir) if out_dir is not None else Path(f"results/stage2c_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "bridges.npz"
    opt_path = out_dir / "opt.npz"
    log = out_dir / "train_log.jsonl"
    global_step = int(stack.meta.get("step", 0))
    resumed = bool(stack.meta.get("resumed"))

    psi = PsiConstitutionMLX(model, tok, const, bridges, l_fwd=args.l_fwd,
                             l_rev=args.l_rev, lam_gate=args.lam_gate)
    # bias correction matters: MLX defaults to none, so a fresh AdamW takes 3-6x
    # steps for its first ~15 updates. The state is persisted across chunks.
    opt = optim.AdamW(learning_rate=args.lr, bias_correction=True)
    if resumed and opt_path.exists():
        opt.state = tree_unflatten(list(mx.load(str(opt_path)).items()))
        opt.state["learning_rate"] = mx.array(args.lr)   # the CLI wins over the checkpoint
        mx.eval(opt.state)
        print(f"resumed at step {global_step} (optimizer state restored, "
              f"opt step {int(opt.state['step'].item())}, lr {args.lr})", flush=True)
    elif resumed:
        print(f"resumed at step {global_step} (optimizer state fresh)", flush=True)
    for k in RESUME_WARN:
        prev = stack.meta.get("prev_args", {})
        if k in prev and prev[k] != getattr(args, k, None):
            print(f"[WARN] --{k.replace('_', '-')}={getattr(args, k, None)} differs from "
                  f"the checkpoint's {prev[k]}")

    if getattr(model, "needs_train_mode_for_grad", False):
        # this backbone's SSM scan has no VJP; only the layers the backward pass
        # reaches need mlx-lm's differentiable fallback
        model.set_grad_window(psi.l_rev)
        print(f"[backbone] differentiable SSM scan from layer {psi.l_rev} up", flush=True)
    if args.checkpoint_from is not None:
        MlxStream.checkpoint_from = psi.l_rev if args.checkpoint_from < 0 else args.checkpoint_from
        print(f"[backbone] recomputing layers >= {MlxStream.checkpoint_from} in the "
              f"backward pass", flush=True)

    data_path = args.data or f"data/constitution_train_{args.tag}.json"
    val_path = args.val or f"data/constitution_val_{args.tag}.json"
    train_items = json.loads(Path(data_path).read_text())
    val_items = json.loads(Path(val_path).read_text())
    noharm_items = json.loads(Path(args.noharm_data).read_text()) if args.noharm_data else None
    pad = _pad_id(hf_tok)

    if args.readout_norm == "dim" and not resumed:
        # per-dimension statistics of the read layer on real prompts; without them
        # a backbone's constant massive-activation dims swallow the across-prompt
        # variance and the pooled soft tokens barely move between items
        cn = min(args.calib_n, len(train_items))
        cbatch = pad_batch(random.Random(7).sample(train_items, cn), pad, None)
        cs = MlxStream(model, cbatch["p_ids"], cbatch["p_attn"])
        cs.run(0, psi.l_fwd)
        bridges.fwd.calibrate_readout(cs.hidden, cbatch["prompt_mask"])
        sig = bridges.fwd.dim_sigma
        print(f"readout calibrated on {cn} prompts at layer {psi.l_fwd}: sigma max "
              f"{float(sig.max().item()):.1f} median "
              f"{float(mx.sort(sig)[sig.shape[0] // 2].item()):.3f}", flush=True)
        del cs

    n_params = sum(v.size for _, v in tree_flatten(bridges.parameters()))
    print(f"bridges: {n_params / 1e6:.2f}M | backbone {args.model} d={bridges.d_model} | "
          f"constitution {const.path} d={const.d_const} | coupling {psi.l_fwd}/{psi.l_rev} "
          f"of {psi.n_layers} | K={bridges.k_fwd} M={bridges.m_tokens} | "
          f"write {dims_label(bridges.write_dims, bridges.d_model)} | "
          f"read {dims_label(bridges.read_dims, bridges.d_model)} | "
          f"emb_rms {bridges.fwd.emb_rms:.3f}", flush=True)
    print(f"data: {len(train_items)} train, {len(val_items)} val"
          + (f", {len(noharm_items)} no-harm every {args.noharm_every}th step "
             f"(gate-only {bool(args.noharm_gate_only)}, lam_gate {args.lam_gate})"
             if noharm_items else " (no no-harm arm)"), flush=True)
    if noharm_items:
        assert args.noharm_every >= 2, "--noharm-every 1 would starve the positive arm"

    def wrapped(bridges_, batch):
        psi.phi = bridges_
        return psi.loss_fn(batch)

    loss_and_grad = nn.value_and_grad(bridges, wrapped)

    t0 = time.time()
    run = {"P": {}, "N": {}}            # running sums per phase between log points
    def _acc(phase, **kv):
        d = run[phase]
        for k, v in kv.items():
            d[k] = d.get(k, 0.0) + v
        d["_n"] = d.get("_n", 0) + 1

    for i in range(args.steps):
        rng = random.Random(23_000_000 + global_step)
        noharm = bool(noharm_items) and global_step % args.noharm_every == args.noharm_every - 1
        if noharm:
            items = rng.sample(noharm_items, min(args.batch, len(noharm_items)))
            batch = pad_batch(items, pad, "target_ids", noharm=True)
        else:
            items = rng.sample(train_items, min(args.batch, len(train_items)))
            batch = pad_batch(items, pad, "teacher_ids")
        (loss, aux), grads = loss_and_grad(bridges, batch)
        if noharm and args.noharm_gate_only:
            # closing the gate is the one allowed route to a lower no-harm loss
            keep = {"g1": grads["inject"]["g1"], "g2": grads["inject"]["g2"]}
            grads = tree_map(lambda g: mx.zeros_like(g), grads)
            grads["inject"]["g1"], grads["inject"]["g2"] = keep["g1"], keep["g2"]
        if args.clip == "module":
            grads = {k: clip_grad_norm(g, 1.0)[0] for k, g in grads.items()}
        else:
            grads, _ = clip_grad_norm(grads, 1.0)
        opt.update(bridges, grads)
        mx.eval(bridges.parameters(), opt.state)
        global_step += 1
        ph = "N" if noharm else "P"
        _acc(ph, ce=aux[0].item(), gate_all=aux[1].item(), gate_ans=aux[2].item(),
             inj_ratio=aux[3].item(), gate_prompt=aux[4].item())
        if global_step % 25 == 0 or i == args.steps - 1:
            rec = {"step": global_step, "phase": ph,
                   "ce": round(aux[0].item(), 4), "gate": round(aux[1].item(), 4),
                   "gate_ans": round(aux[2].item(), 4),
                   "inj_ratio_ans": round(aux[3].item(), 4),
                   "gate_prompt": round(aux[4].item(), 4),
                   "sec_per_step": round((time.time() - t0) / (i + 1), 2)}
            for pk, d in run.items():
                n = d.get("_n", 0)
                if n:
                    for k, v in d.items():
                        if k != "_n":
                            rec[f"{pk}_{k}"] = round(v / n, 4)
                    rec[f"{pk}_n"] = n
            run = {"P": {}, "N": {}}
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

    # -- end of chunk: teacher-forced held-out scores ------------------------
    psi.phi = bridges
    if getattr(model, "needs_train_mode_for_grad", False):
        model.set_grad_window(psi.n_layers)          # score on the deployment numerics
    ev = eval_arm(psi, val_items, args.eval_n, "psilm")
    bkey = f"{Path(val_path).name}|{args.eval_n}|{args.model}"
    bev = base_eval_cached(psi, val_items, args.eval_n, out_dir, bkey)
    if getattr(model, "needs_train_mode_for_grad", False):
        model.set_grad_window(psi.l_rev)

    bridges.save_weights(str(ckpt))
    mx.savez(str(opt_path), **dict(tree_flatten(opt.state)))
    meta = stack_meta(bridges, const, psi, global_step,
                      extra={"model": args.model, "hf_tokenizer": args.hf_tokenizer,
                             "data": data_path, "val": val_path,
                             "write_dims_spec": args.write_dims,
                             "read_dims_spec": args.read_dims,
                             "eval": {"psilm": ev, "base": bev},
                             "args": vars(args)})
    Path(str(ckpt) + ".meta").write_text(json.dumps(meta))
    with log.open("a") as f:
        f.write(json.dumps({"step": global_step, "eval": {"psilm": ev, "base": bev}}) + "\n")
    # MLX allocates through Metal, which ps does not account for: this is the only
    # honest read on how close a coupling depth runs to the 24 GB ceiling
    print(f"CHUNK DONE step={global_step} ce_val={ev['ce']} agree={ev['agree']} "
          f"base_ce={bev['ce']} base_agree={bev['agree']} gate_ans={ev['gate_cont']} "
          f"peak={mx.get_peak_memory() / 2**30:.1f}GB", flush=True)
    return ev, bev


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test(args):
    from psilm.mlx.constitution import build_tiny_stack, self_test as module_self_test
    from psilm.mlx.constitution import synthetic_items
    module_self_test()
    tmp = Path(tempfile.mkdtemp(prefix="constitution_selftest_"))
    (tmp / "train.json").write_text(json.dumps(synthetic_items(6, seed=1)))
    (tmp / "val.json").write_text(json.dumps(synthetic_items(3, seed=2)))
    (tmp / "noharm.json").write_text(json.dumps(
        synthetic_items(4, seed=3, key="target_ids")))
    stack, _ = build_tiny_stack(write_idx=[2, 4, 8, 16, 32, 48])
    args.steps, args.batch, args.eval_n, args.calib_n = 3, 2, 2, 4
    args.model, args.const_model = "<tiny synthetic qwen2>", "<tiny>"
    args.data = str(tmp / "train.json")
    args.val = str(tmp / "val.json")
    args.noharm_data = str(tmp / "noharm.json")
    args.noharm_every, args.fresh = 2, True
    ev, bev = run_training(args, stack, out_dir=tmp / "ckpt")
    assert ev["ce"] is not None and bev["ce"] is not None
    assert (tmp / "ckpt" / "bridges.npz").exists() and (tmp / "ckpt" / "opt.npz").exists()
    print(f"[self-test] 3 training steps + chunk eval on the tiny model: "
          f"psilm ce {ev['ce']} agree {ev['agree']}, base ce {bev['ce']}  OK")
    # the evaluator, on the checkpoint those steps just wrote (with the meta they
    # wrote, so the coupling depths it reads are the trained ones)
    ckpt = tmp / "ckpt" / "bridges.npz"
    stack = stack._replace(meta=json.loads(Path(str(ckpt) + ".meta").read_text()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mlx_constitution_eval as ev_mod
    eargs = ev_mod.build_parser().parse_args(
        ["--ckpt", str(ckpt), "--data", str(tmp / "val.json"),
         "--n", "2", "--max-new", "4", "--arms", "base,psilm,zeroed",
         "--out", str(tmp / "eval.json")])
    ev_mod.run_eval(eargs, stack)
    assert json.loads((tmp / "eval.json").read_text())["arms"]
    print(f"[self-test] eval/mlx_constitution_eval.py ran on the tiny checkpoint  OK")
    print(f"[self-test] eval/mlx_constitution_train.py: all assertions passed "
          f"(artifacts in {tmp})")


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(args)
        return
    ckpt = Path(f"results/stage2c_{args.tag}/bridges.npz")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    stack = build_stack(args, ckpt)
    run_training(args, stack)


if __name__ == "__main__":
    main()
