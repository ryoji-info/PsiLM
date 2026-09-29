#!/usr/bin/env python3
"""One stored set of tokens for a constitution bridge, and how constant its own are.

eval/constitution_compress.py found that, teacher-forced, a full-width bridge on
the 9B and on Bonsai writes the same thing whatever prompt it read (another
prompt's tokens, or a partner fed zeros, reproduce its effect). This builds what
the `fixed` arm of eval/bench_guardrail.py injects to test that on GENERATED
behaviour: the mean, over prompts that are in no evaluation, of the tokens the
trained channel computes --

    h at l_fwd --forward bridge--> soft --partner--> features --reverse bridge--> tokens

-- and records how far each prompt's own tokens sit from that mean. A channel
that carried the prompt would show prompts far from the mean and from each other.

  python eval/constitution_fixed_tokens.py --ckpt results/stage2c_qwen35_all/bridges.npz \\
      --data data/constitution_val_qwen35.json --n 100
  python eval/constitution_fixed_tokens.py --self-test     # tiny stack on the CPU

Writes <checkpoint dir>/fixed_tokens_<kind>.npz (tokens: 1 x M x d_model) and the
.json beside it (what was stored, where from, and the spread). Kinds:

  mean      the mean of the prompts' own tokens: the closest single set to what
            the trained channel writes
  softzero  the tokens of a partner fed zeros in place of the forward bridge's
            read: a set made from NO prompt, so it carries no prompt's category
            either. The strictest test. (For a bridge trained without a partner
            the two kinds are the same tensor.)

--contrast-data adds the distance between this data's mean and another set's (the
benign prompts, say): if the two means are as close as prompts of one set are to
their own mean, the channel does not read the category either.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_tokens(backbone, coupler, ids, l_fwd, soft_zero=False):
    """The tokens the channel computes for one prompt (1, M, d_model), float32."""
    import mlx.core as mx
    from psilm.mlx.staged import MlxStream
    s = MlxStream(backbone, mx.array([list(ids)], dtype=mx.int32), None)
    s.run(0, l_fwd)
    if soft_zero:
        pmask = mx.ones((1, len(ids)), dtype=mx.bool_)
        soft, _ = coupler.phi.fwd(s.hidden, pmask)
        tokens = coupler.phi.rev(coupler.const.features(mx.zeros_like(soft)))
    else:
        tokens, _ = coupler.tokens(s.hidden)
    tokens = tokens.astype(mx.float32)
    mx.eval(tokens)
    return tokens


def spread(tokens, mean):
    """How far each prompt's tokens are from the mean, and from each other."""
    import mlx.core as mx
    flat = mx.stack([t.reshape(-1) for t in tokens])              # (N, M*d)
    m = mean.reshape(-1)
    cos = (flat @ m) / (mx.linalg.norm(flat, axis=-1) * mx.linalg.norm(m) + 1e-12)
    rel = mx.linalg.norm(flat - m[None], axis=-1) / (mx.linalg.norm(m) + 1e-12)
    unit = flat / (mx.linalg.norm(flat, axis=-1, keepdims=True) + 1e-12)
    pair = unit @ unit.T
    n = flat.shape[0]
    off = (pair.sum() - mx.diag(pair).sum()) / max(n * (n - 1), 1)
    norms = mx.linalg.norm(flat, axis=-1)
    mx.eval(cos, rel, off, norms)
    c, r = [float(x) for x in cos.tolist()], [float(x) for x in rel.tolist()]
    nm = float(norms.mean().item())
    return {"cosine_to_mean": {"mean": round(sum(c) / n, 6), "min": round(min(c), 6)},
            "relative_distance_to_mean": {"mean": round(sum(r) / n, 6), "max": round(max(r), 6)},
            "mean_pairwise_cosine": round(float(off), 6),
            # averaging directions that differ shortens the vector: 1.0 is no shrinkage
            "norm": {"of_the_mean": round(float(mx.linalg.norm(m).item()), 5),
                     "mean_of_prompts": round(nm, 5),
                     "ratio": round(float(mx.linalg.norm(m).item()) / max(nm, 1e-12), 6)},
            "token_rms": round(float(mx.sqrt((mean * mean).mean()).item()), 6)}


def distance(a, b):
    import mlx.core as mx
    x, y = a.reshape(-1), b.reshape(-1)
    return {"relative_distance": round(float((mx.linalg.norm(x - y) / (mx.linalg.norm(y) + 1e-12)).item()), 6),
            "cosine": round(float(((x * y).sum() / (mx.linalg.norm(x) * mx.linalg.norm(y) + 1e-12)).item()), 6)}


def mean_tokens(backbone, coupler, items, l_fwd, log=print):
    import mlx.core as mx
    toks = []
    for i, it in enumerate(items):
        toks.append(prompt_tokens(backbone, coupler, it["prompt_ids"], l_fwd))
        if (i + 1) % 25 == 0:
            log(f"  [{i + 1}/{len(items)}]")
    return mx.stack(toks).mean(axis=0), toks                       # (1, M, d)


def build(backbone, coupler, items, l_fwd, kind="mean", contrast=None, log=print):
    import mlx.core as mx
    mean, toks = mean_tokens(backbone, coupler, items, l_fwd, log)
    stats = spread(toks, mean)
    zero = prompt_tokens(backbone, coupler, items[0]["prompt_ids"], l_fwd, soft_zero=True)
    other = prompt_tokens(backbone, coupler, items[-1]["prompt_ids"], l_fwd, soft_zero=True)
    # made from no prompt: the same whichever prompt set the batch size
    stats["softzero"] = {**distance(zero, mean),
                         "varies_with_the_prompt": float(mx.abs(zero - other).max().item())}
    if contrast:
        cmean, ctoks = mean_tokens(backbone, coupler, contrast, l_fwd, log)
        stats["contrast"] = {"n": len(contrast), **distance(cmean, mean),
                             "own_relative_distance_to_mean": spread(ctoks, cmean)["relative_distance_to_mean"]}
    out = zero if kind == "softzero" else mean
    mx.eval(out)
    return out, stats


def run(args):
    import mlx.core as mx
    from psilm.mlx.constitution import ConstitutionCoupler, load_constitution_stack
    from psilm.mlx.gemma_loader import load_backbone_any

    ckpt = Path(args.ckpt)
    bridges, const, meta = load_constitution_stack(ckpt, args.const_model)
    tower, _, _ = load_backbone_any(args.model or meta["model"])
    tower.freeze()
    assert int(tower.args.hidden_size) == int(meta["d_model"])
    items = json.loads(Path(args.data).read_text())[:args.n]
    contrast = json.loads(Path(args.contrast_data).read_text())[:args.n] if args.contrast_data else None
    coupler = ConstitutionCoupler(bridges, const)
    t0 = time.time()
    log = lambda m: print(m, flush=True)                                        # noqa: E731
    mean, toks = mean_tokens(tower, coupler, items, int(meta["l_fwd"]), log)
    stats = spread(toks, mean)
    first, last = (prompt_tokens(tower, coupler, it["prompt_ids"], int(meta["l_fwd"]), soft_zero=True)
                   for it in (items[0], items[-1]))
    stats["softzero"] = {**distance(first, mean),
                         "varies_with_the_prompt": float(mx.abs(first - last).max().item())}
    if contrast:
        cmean, ctoks = mean_tokens(tower, coupler, contrast, int(meta["l_fwd"]), log)
        stats["contrast"] = {"data": str(args.contrast_data), "n": len(contrast), **distance(cmean, mean),
                             "own_relative_distance_to_mean": spread(ctoks, cmean)["relative_distance_to_mean"]}
    common = {"ckpt": str(ckpt), "ckpt_step": int(meta["step"]), "ckpt_sha256": file_sha256(ckpt),
              "model": str(args.model or meta["model"]),
              "const_model": str(args.const_model or meta["const_model"]),
              "l_fwd": int(meta["l_fwd"]), "spread_of_the_prompts_own_tokens": stats,
              "sec": round(time.time() - t0, 1)}
    for kind in args.kind.split(","):
        tokens = first if kind == "softzero" else mean
        out = Path(args.out) if args.out else ckpt.parent / f"fixed_tokens_{kind}.npz"
        mx.savez(str(out), tokens=tokens)
        made = ({"data": "none: the partner was fed zeros", "n": 0, "sources": []} if kind == "softzero"
                else {"data": str(args.data), "n": len(items), "sources": [it.get("source") for it in items]})
        prov = {"kind": kind, "tokens_sha256": file_sha256(out), "shape": list(tokens.shape),
                "token_rms": round(float(mx.sqrt((tokens * tokens).mean()).item()), 6), **made, **common}
        out.with_suffix(".json").write_text(json.dumps(prov, indent=1) + "\n")
        print(json.dumps({k: v for k, v in prov.items() if k != "sources"}, indent=1))
        print(f"FIXED-TOKENS {kind} -> {out}", flush=True)
    print("FIXED-TOKENS COMPLETE", flush=True)
    return 0


def self_test():
    import tempfile
    import types
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from bench_common import StagedDecoder, arm_fixed, arm_spec
    from psilm.mlx.constitution import (ConstantPartner, ConstitutionCoupler, ConstitutionModelMLX,
                                        PsiConstitutionMLX, build_tiny_stack,
                                        load_constitution_stack, make_partner, pad_batch,
                                        stack_meta, synthetic_items)

    mx.random.seed(0)
    stack, _ = build_tiny_stack()
    l_fwd, l_rev = int(stack.meta["l_fwd"]), int(stack.meta["l_rev"])
    coupler = ConstitutionCoupler(stack.bridges, stack.const)
    items = synthetic_items(6, seed=9)

    # ---- the tokens and their spread ---------------------------------------------
    mean, stats = build(stack.model, coupler, items, l_fwd, log=lambda m: None)
    M, d = stack.bridges.m_tokens, stack.bridges.d_model
    assert tuple(mean.shape) == (1, M, d) and mean.dtype == mx.float32
    own = [prompt_tokens(stack.model, coupler, it["prompt_ids"], l_fwd) for it in items]
    assert float(mx.abs(mx.stack(own).mean(axis=0) - mean).max()) < 1e-6
    assert 0 < stats["relative_distance_to_mean"]["mean"] and stats["cosine_to_mean"]["min"] <= 1.0
    assert 0 < stats["norm"]["ratio"] <= 1.0 + 1e-6            # a mean is never longer than its parts
    same = spread([own[0], own[0]], own[0])
    assert same["cosine_to_mean"]["min"] > 0.999999 and same["relative_distance_to_mean"]["max"] < 1e-6
    assert abs(same["norm"]["ratio"] - 1.0) < 1e-6
    zero, zstats = build(stack.model, coupler, items, l_fwd, kind="softzero", contrast=items[:3],
                         log=lambda m: None)
    assert float(mx.abs(zero - mean).max()) > 1e-6
    assert zstats["softzero"]["varies_with_the_prompt"] == 0.0   # made from no prompt
    assert zstats["contrast"]["n"] == 3 and 0 <= zstats["contrast"]["relative_distance"]
    assert distance(mean, mean) == {"relative_distance": 0.0, "cosine": 1.0}
    print(f"[self-test] tokens: shape {tuple(mean.shape)}, the mean is the mean, spread "
          f"{stats['relative_distance_to_mean']['mean']:.3f} on the tiny stack  OK")

    # ---- the fixed arm in the decoder -----------------------------------------------
    assert arm_spec("fixed") == ("psilm", None, False) and arm_fixed("fixed") and not arm_fixed("psilm")
    dec = StagedDecoder(stack.model, stack.hf_tok, None, None, l_fwd, l_rev,
                        eos_ids=set(), coupler=coupler)
    p = items[0]["prompt_ids"]
    ref = dec.generate(p, mode="psilm", max_new=12)
    base = dec.generate(p, mode="base", max_new=12)
    kl_ref = dec.kl_to_base(p, base.gen_ids, "psilm")
    dec.set_fixed_tokens(own[0], "mean:abc")            # its own tokens: the psilm arm exactly
    got = dec.generate(p, mode="psilm", max_new=12)
    kl_own = dec.kl_to_base(p, base.gen_ids, "psilm")
    kl_full = dec.kl_to_base(p, base.gen_ids, "psilm", pool="full")
    assert abs(kl_full["mean"] - kl_own["mean"]) < 1e-9        # fixed tokens: the two reads coincide
    assert got.gen_ids == ref.gen_ids and got.diag.get("fixed_tokens") == "mean:abc"
    assert abs(kl_own["mean"] - kl_ref["mean"]) < 1e-6, (kl_own, kl_ref)
    dec.set_fixed_tokens(own[3] * 3.0)                  # another prompt's, scaled: a different write
    kl_other = dec.kl_to_base(p, base.gen_ids, "psilm")
    assert abs(kl_other["mean"] - kl_ref["mean"]) > 1e-7
    assert kl_other["mean"] > 0                         # it writes: not the zeroed arm
    dec.set_fixed_tokens(None)                          # cleared: the trained arm again
    again = dec.generate(p, mode="psilm", max_new=12)
    assert again.gen_ids == ref.gen_ids and "fixed_tokens" not in again.diag
    assert abs(dec.kl_to_base(p, base.gen_ids, "psilm")["mean"] - kl_ref["mean"]) < 1e-9
    for bad in (mx.zeros((1, M + 1, d)), mx.zeros((M, d)), mx.zeros((2, M, d))):
        try:
            dec.set_fixed_tokens(bad)
            raise AssertionError("a wrong shape was accepted")
        except ValueError:
            pass
    assert dec.fixed_tokens is None
    try:
        StagedDecoder(stack.model, stack.hf_tok, None, stack.bridges, l_fwd, l_rev,
                      eos_ids=set()).set_fixed_tokens(own[0])
        raise AssertionError("a decoder without a constitution coupler accepted fixed tokens")
    except RuntimeError:
        pass
    dec.set_contentless(0, qid="q")                     # the two controls never arm together
    try:
        dec.set_fixed_tokens(own[0])
        raise AssertionError("fixed tokens were armed over a contentless arm")
    except RuntimeError:
        pass
    dec.set_contentless(None)
    dec.set_fixed_tokens(own[0])
    try:
        dec.set_contentless(0, qid="q")
        raise AssertionError("a contentless arm was armed over fixed tokens")
    except RuntimeError:
        pass
    dec.set_fixed_tokens(None)
    assert all(arm_fixed(a) for a in ("fixed", "fixedzero", "fixed2")) and not arm_fixed("fixed-x")
    assert arm_spec("fixedzero") == ("psilm", None, False)
    print("[self-test] fixed arm: own tokens == the psilm arm token for token and in KL; other "
          "tokens differ and write; clearing restores; wrong shapes and channels are refused  OK")

    # ---- the harness's provenance check ----------------------------------------------
    import bench_guardrail as bg
    tmp = Path(tempfile.mkdtemp(prefix="fixed_tokens_selftest_"))
    ckpt = tmp / "bridges.npz"
    stack.bridges.save_weights(str(ckpt))
    psi = PsiConstitutionMLX(stack.model, stack.tok, stack.const, stack.bridges, l_fwd=l_fwd, l_rev=l_rev)
    Path(str(ckpt) + ".meta").write_text(json.dumps(stack_meta(stack.bridges, stack.const, psi, 7)))
    tf = tmp / "fixed_tokens_mean.npz"
    mx.savez(str(tf), tokens=mean)
    prov = {"kind": "mean", "ckpt": str(ckpt), "ckpt_step": 7, "ckpt_sha256": file_sha256(ckpt),
            "tokens_sha256": file_sha256(tf), "data": "x.json", "n": 6}
    tf.with_suffix(".json").write_text(json.dumps(prov))
    ident = f"mean:{prov['tokens_sha256'][:16]}"
    a = types.SimpleNamespace(fixed_tokens=f"fixed={tf},fixedzero={tf}", ckpt=str(ckpt))
    got = bg.load_fixed_tokens(a, ["base", "psilm", "fixed", "fixedzero"])
    assert set(got) == {"fixed", "fixedzero"} and got["fixed"][1] == ident
    assert float(mx.abs(got["fixed"][0] - mean).max()) == 0.0
    assert bg.load_fixed_tokens(a, ["base", "psilm"]) == {}
    rows = [{"arm": "fixed", "diag": {"fixed_tokens": ident}}, {"arm": "psilm", "diag": {}}]
    assert set(bg.load_fixed_tokens(a, ["fixed"], rows)) == {"fixed"}           # a resume of the same set

    def refused(args, arms, rows=()):
        try:
            bg.load_fixed_tokens(args, arms, rows)
        except SystemExit:
            return True
        return False
    assert refused(a, ["fixed"], [{"arm": "fixed", "diag": {"fixed_tokens": "mean:0000"}}])   # another set
    assert refused(types.SimpleNamespace(fixed_tokens=f"fixed={tf}", ckpt=str(ckpt)), ["fixed", "fixedzero"])
    assert refused(types.SimpleNamespace(fixed_tokens=None, ckpt=str(ckpt)), ["fixed"])
    for change in ({"ckpt_step": 8}, {"ckpt_sha256": "0" * 64}, {"tokens_sha256": "0" * 64}):
        tf.with_suffix(".json").write_text(json.dumps({**prov, **change}))
        assert refused(a, ["fixed"]), change
    tf.with_suffix(".json").write_text(json.dumps(prov))
    mx.savez(str(tmp / "bad.npz"), tokens=mx.zeros((1, M + 1, d)))
    (tmp / "bad.json").write_text(json.dumps({**prov, "tokens_sha256": file_sha256(tmp / "bad.npz")}))
    assert refused(types.SimpleNamespace(fixed_tokens=f"fixed={tmp / 'bad.npz'}", ckpt=str(ckpt)), ["fixed"])
    assert dec.fixed_tokens is None
    print("[self-test] harness: tokens load only for the checkpoint they were made from, as the file "
          "their record describes, in the bridges' shape, and never over rows of another set  OK")

    # ---- the bridge without its partner: the write and one stored set -------------------
    from psilm.mlx.constitution import (StoredTokenCoupler, load_constitution_write, load_stored_stack,
                                        load_stored_tokens, stored_tokens_beside)
    assert stored_tokens_beside(ckpt) == tf
    sc, smeta, srec = load_stored_stack(ckpt)
    assert isinstance(sc, StoredTokenCoupler) and not sc.needs_read and sc.ident == ident
    assert not hasattr(sc, "const") and not hasattr(sc, "phi")            # no partner, no read
    assert srec["kind"] == "mean" and int(smeta["step"]) == 7
    got_tokens, diag = sc.tokens(None)
    assert float(mx.abs(got_tokens - mean).max()) == 0.0 and diag == {"stored_tokens": ident}
    h = mx.random.normal((1, 5, d))
    for mode in ("psilm", "zeroed"):
        a1, s1 = coupler.inject(h, mean, mode)
        a2, s2 = sc.inject(h, got_tokens, mode)
        assert float(mx.abs(a1 - a2).max()) == 0.0 and float(mx.abs(s1 - s2).max()) == 0.0, mode
    assert float(mx.abs(sc.inject(h, got_tokens, "zeroed")[0] - h).max()) == 0.0
    # the decoder cannot tell it from the full coupler given the same tokens
    dec.coupler, keep = sc, dec.coupler
    dec.set_fixed_tokens(None)
    it0 = items[0]
    own0 = prompt_tokens(stack.model, coupler, it0["prompt_ids"], l_fwd)
    dec.coupler = StoredTokenCoupler(stack.bridges.inject, own0, "own")
    g_stored = dec.generate(it0["prompt_ids"], mode="psilm", max_new=6)
    dec.coupler = keep
    g_full = dec.generate(it0["prompt_ids"], mode="psilm", max_new=6)
    assert g_stored.gen_ids == g_full.gen_ids
    w, _ = load_constitution_write(ckpt)
    flat = dict(tree_flatten(w.parameters()))
    full = dict(tree_flatten(stack.bridges.inject.parameters()))
    assert set(flat) == set(full) and all(float(mx.abs(flat[k].astype(mx.float32)
                                                       - full[k].astype(mx.float32)).max()) == 0.0 for k in flat)

    def no(fn):
        try:
            fn()
        except ValueError:
            return True
        return False
    for change in ({"ckpt_step": 8}, {"ckpt_sha256": "0" * 64}, {"tokens_sha256": "0" * 64}):
        tf.with_suffix(".json").write_text(json.dumps({**prov, **change}))
        assert no(lambda: load_stored_stack(ckpt)), change
    # the Hugging Face layout: the exported bridges have another hash, which the record names
    tf.with_suffix(".json").write_text(json.dumps({**prov, "ckpt_sha256": "0" * 64,
                                                   "bridges_sha256": file_sha256(ckpt)}))
    assert load_stored_stack(ckpt)[0].ident == ident
    tf.with_suffix(".json").write_text(json.dumps(prov))
    assert no(lambda: load_stored_tokens(tmp / "bad.npz", ckpt))          # the wrong shape
    assert no(lambda: StoredTokenCoupler(stack.bridges.inject, mx.zeros((1, M, d + 1))))
    assert no(lambda: sc.inject(h, got_tokens, "psilm", contentless=3))
    other = tmp / "elsewhere"
    other.mkdir()
    stack.bridges.save_weights(str(other / "bridges.npz"))
    Path(str(other / "bridges.npz") + ".meta").write_text(Path(str(ckpt) + ".meta").read_text())
    assert stored_tokens_beside(other / "bridges.npz") is None
    assert no(lambda: load_stored_stack(other / "bridges.npz"))           # nothing stored beside it
    mx.savez(str(other / "fixed_tokens_mean.npz"), tokens=mean)
    assert stored_tokens_beside(other / "bridges.npz") is None            # tokens without a record
    assert no(lambda: load_stored_tokens(other / "fixed_tokens_mean.npz", other / "bridges.npz"))
    print("[self-test] stored stack: the write and one set of tokens, no partner and no read; the same "
          "write as the full coupler's bit for bit; tokens of another checkpoint, another shape or "
          "without a record are refused  OK")

    # ---- no partner at all: the constant stand-in ---------------------------------------
    dc = stack.bridges.d_const
    cp = make_partner(f"constant:3:{dc}", m_tokens=M)
    assert isinstance(cp, ConstantPartner) and cp.path == f"constant:3:{dc}" and cp.n_readout == M
    assert isinstance(stack.const, ConstitutionModelMLX)
    f1 = cp.features(mx.zeros((2, 3, dc)))
    f2 = ConstantPartner(f"constant:3:{dc}", M).features(mx.random.normal((2, 3, dc)))
    assert tuple(f1.shape) == (2, M, dc) and float(mx.abs(f1 - f2).max()) == 0.0   # the seed alone
    assert float(mx.abs(f1[0] - f1[1]).max()) == 0.0
    assert float(mx.abs(ConstantPartner(f"constant:4:{dc}", M).features(f1) - f1).max()) > 0.1
    assert ConstantPartner("constant:0", 8).d_const == 896
    for bad in ("constant:", "constant:x", "constant:1:2:3", "constant:1:y"):
        try:
            ConstantPartner(bad, M)
            raise AssertionError(f"{bad} was accepted")
        except ValueError:
            pass
    mx.random.seed(1)
    psi_c = PsiConstitutionMLX(stack.model, stack.tok, cp, stack.bridges, l_fwd=l_fwd, l_rev=l_rev)
    batch = pad_batch(items[:2], 0, "teacher_ids")

    def wrapped(b_, bt):
        psi_c.phi = b_
        return psi_c.loss_fn(bt)
    (loss, _), grads = nn.value_and_grad(stack.bridges, wrapped)(stack.bridges, batch)
    g = {k: float(mx.abs(v).max()) for k, v in tree_flatten(grads)}
    assert all(v == 0.0 for k, v in g.items() if k.startswith("fwd.")), "the forward bridge got a gradient"
    assert g["rev.proj.weight"] > 0 and g["rev.pos_bias"] > 0 and g["inject.to_out.weight"] > 0
    assert g["inject.g1.weight"] > 0
    c1 = ConstitutionCoupler(stack.bridges, cp)
    ta = prompt_tokens(stack.model, c1, items[0]["prompt_ids"], l_fwd)
    tb = prompt_tokens(stack.model, c1, items[4]["prompt_ids"], l_fwd)
    assert float(mx.abs(ta - tb).max()) == 0.0               # the same tokens for every prompt
    print("[self-test] constant partner: a function of its seed, the same for every prompt; the "
          "forward bridge gets no gradient, the reverse bridge and the write do  OK")

    # ---- ... through the real trainer, a save, a load and the evaluator -----------------
    import mlx_constitution_train as tr
    targs = tr.build_parser().parse_args(["--tag", "selftest_nopartner"])
    (tmp / "train.json").write_text(json.dumps(synthetic_items(6, seed=1)))
    (tmp / "val.json").write_text(json.dumps(synthetic_items(3, seed=2)))
    (tmp / "noharm.json").write_text(json.dumps(synthetic_items(4, seed=3, key="target_ids")))
    targs.steps, targs.batch, targs.eval_n, targs.calib_n = 4, 2, 2, 4
    targs.model, targs.const_model = "<tiny synthetic qwen2>", cp.path
    targs.data, targs.val, targs.noharm_data = (str(tmp / "train.json"), str(tmp / "val.json"),
                                                str(tmp / "noharm.json"))
    targs.noharm_every, targs.fresh, targs.checkpoint_from = 2, True, -1
    fresh, _ = build_tiny_stack()
    fwd0 = mx.array(fresh.bridges.fwd.key.weight)
    rev0 = mx.array(fresh.bridges.rev.proj.weight)
    ev, bev = tr.run_training(targs, fresh._replace(const=cp), out_dir=tmp / "run")
    assert ev["ce"] is not None and bev["ce"] is not None
    meta = json.loads((tmp / "run" / "bridges.npz.meta").read_text())
    assert meta["const_model"] == cp.path and int(meta["step"]) == 4
    assert meta["const_feats_sha256"] == cp.sha256 and len(cp.sha256) == 64
    assert float(mx.abs(fresh.bridges.rev.proj.weight - rev0).max()) > 0          # trained
    rel = float(mx.abs(fresh.bridges.fwd.key.weight - fwd0).max() / mx.abs(fwd0).max())
    assert rel < 1e-3, rel                                   # weight decay only, never a gradient
    b2, c2, m2 = load_constitution_stack(tmp / "run" / "bridges.npz")
    assert isinstance(c2, ConstantPartner) and c2.path == cp.path
    for swap in (f"constant:4:{dc}", "some/model/dir"):      # another constant, or a model
        try:
            load_constitution_stack(tmp / "run" / "bridges.npz", swap)
            raise AssertionError(f"the partner-free checkpoint loaded with {swap}")
        except ValueError:
            pass
    try:                                                       # and a partner's bridges with none
        load_constitution_stack(ckpt, f"constant:0:{dc}")
        raise AssertionError("a partner's checkpoint loaded with a constant")
    except ValueError:
        pass
    mp = tmp / "run" / "bridges.npz.meta"                      # a draw that is not the trained one
    good = mp.read_text()
    mp.write_text(json.dumps({**json.loads(good), "const_feats_sha256": "0" * 64}))
    try:
        load_constitution_stack(tmp / "run" / "bridges.npz")
        raise AssertionError("a checkpoint loaded on a constant it was not trained on")
    except ValueError:
        pass
    mp.write_text(good)
    assert float(mx.abs(c2.features(f1) - f1).max()) == 0.0
    psi2 = PsiConstitutionMLX(stack.model, stack.tok, c2, b2, l_fwd=int(m2["l_fwd"]), l_rev=int(m2["l_rev"]))
    psi1 = PsiConstitutionMLX(stack.model, stack.tok, cp, fresh.bridges, l_fwd=int(m2["l_fwd"]),
                              l_rev=int(m2["l_rev"]))
    it = items[0]
    ce1, _, _ = psi1.teacher_forced(it["prompt_ids"], it["teacher_ids"], "psilm")
    ce2, _, _ = psi2.teacher_forced(it["prompt_ids"], it["teacher_ids"], "psilm")
    assert abs(ce1 - ce2) < 1e-6, (ce1, ce2)                 # the loaded system is the trained one
    import mlx_constitution_eval as ev_mod
    eargs = ev_mod.build_parser().parse_args(
        ["--ckpt", str(tmp / "run" / "bridges.npz"), "--data", str(tmp / "val.json"), "--n", "2",
         "--max-new", "4", "--arms", "base,psilm,zeroed", "--out", str(tmp / "eval.json")])
    ev_mod.run_eval(eargs, fresh._replace(const=c2, bridges=b2, meta=m2))
    assert json.loads((tmp / "eval.json").read_text())["arms"]["psilm"]["ce"] is not None
    print(f"[self-test] no partner through the trainer (4 steps, recomputed backward), a save, a load "
          f"and the evaluator: ce {ev['ce']} vs base {bev['ce']}  OK")
    print("[self-test] eval/constitution_fixed_tokens.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--model", default=None)
    ap.add_argument("--const-model", default=None)
    ap.add_argument("--data", help="prompts that are in no evaluation (the validation split)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--kind", default="mean,softzero", help="mean, softzero, or both (comma-separated)")
    ap.add_argument("--contrast-data", default=None,
                    help="another prompt set (the benign one) to measure the distance between means")
    ap.add_argument("--out", default=None,
                    help="default: <checkpoint dir>/fixed_tokens_<kind>.npz (one kind only with --out)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.ckpt and a.data):
        ap.error("--ckpt and --data are required")
    if set(a.kind.split(",")) - {"mean", "softzero"} or (a.out and "," in a.kind):
        ap.error("--kind is mean, softzero or mean,softzero; --out takes one kind")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
