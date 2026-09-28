#!/usr/bin/env python3
"""What compressing a constitution bridge and its partner costs, without retraining.

The question (Bonsai chat app, 2026-09): the backbone is ternary; should the
bridges and the partner model be too, if performance is better? Compression after
training cannot make the system more accurate (only noise can look like that) and,
by the app's structure, cannot make replies faster: the partner runs once per
reply and the bridges are a fraction of a percent of the per-token work. What it
can buy is memory. This measures what that memory costs, post-training: every
weight matrix of the bridges and/or of the partner's layers is replaced by its
compressed-then-restored value and the coupled system is scored against the one
that was trained.

  fp16        stored in half precision (on the partner, which is bf16 already, a
              near no-op: the row is the partner's numerical noise floor)
  q8 q4 q2    MLX affine quantization, b bits in groups of 64, in the tensor's own
              dtype as mlx.nn.quantize does it
  t           ternary {-a, 0, +a} per group of 128 (threshold 0.7 mean|w|, a = the
              mean magnitude above it): three levels in a 2-bit container like
              Bonsai's, but FITTED after training, where Bonsai's are trained. A
              failure here says a ternary fit needs training, not that ternary is
              impossible; a pass says it is not even needed.

The partner's vocabulary table is left alone and counted separately: the channel
reads only the rows of its fixed frame (15 ids), so dropping the rest is a
lossless saving that no compression should take credit for.

Controls, on the trained system, that say what the numbers mean:
  base        the write switched off (the bridge's own effect, as a KL)
  shuffle     the tokens of ANOTHER prompt injected: how much of the effect is
              specific to the prompt that was read
  softzero    the partner fed zeros in place of the forward bridge's tokens: what
              is left when nothing of the prompt reaches the partner

Per variant, against the trained system on the same items: ce (to the teacher's
continuation), the paired difference d_ce with a 95% interval, the share of the
bridge's gain retained, kl = KL(trained || variant) per token, same_top1, the gate;
on no-harm prompts (held-out benchmark prompts with the backbone's own
continuation, and a sample of the training negatives) the gate and the KL.

Thresholds, fixed before the run (THRESHOLDS below), decide `ship`.

  python eval/constitution_compress.py --ckpt results/stage2c_bonsai27b_all/bridges.npz \\
      --data data/constitution_test_bonsai27b.json,data/constitution_helpful_test_bonsai27b.json \\
      --noharm data/noharm_bonsai27b_all.json \\
      --heldout-rows results/bench/const_bonsai27b_all_guardrail.rows.jsonl \\
      --tasks-cache results/bench/tasks_const_qwen35_n100.json
  python eval/constitution_compress.py --self-test       # tiny stack on the CPU

Writes <checkpoint dir>/compress.json and .rows.jsonl.
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: (bridges, partner): the trained system first, then one side at a time, then pairs
VARIANTS = [("fp32", "native"),
            ("fp16", "native"), ("q8", "native"), ("q4", "native"), ("q2", "native"), ("t", "native"),
            ("fp32", "fp16"), ("fp32", "q8"), ("fp32", "q4"), ("fp32", "q2"), ("fp32", "t"),
            ("fp16", "q8"), ("fp16", "q4"), ("q8", "q4"), ("t", "t")]
CONTROLS = ("base", "shuffle", "softzero")
Q_GROUP, T_GROUP, T_THRESH = 64, 128, 0.7
THRESHOLDS = {
    "d_ce_max": 0.003,            # per constitution split, nats
    "d_ce_pooled_upper95": 0.006,  # paired bootstrap over both splits
    "kl_max": 0.005,              # KL(trained || variant), nats per token
    "kl_share_of_effect": 0.05,   # ... and as a share of KL(trained || base)
    "d_gate_cont": 0.01,          # |gate - trained gate| on the continuation
    "noharm_gate_mean": 0.008,    # held-out no-harm prompts: mean gate (or trained + 0.002)
    "noharm_gate_prompt": 0.1,    # ... no prompt's mean gate above this (or the trained max)
    "noharm_kl": 0.0005,          # ... and KL(trained || variant) there
}


def is_matrix(key, v):
    """The tensors a quantized export packs: Linear (and Embedding) weights. This is
    mlx.nn.quantize's default set on the bridges, the gate's (1, 256) head included."""
    return v.ndim == 2 and key.endswith("weight")


def ternary(w, group=T_GROUP, thresh=T_THRESH):
    import mlx.core as mx
    rows, cols = w.shape
    g = group if cols % group == 0 else cols            # a width the groups do not divide: per row
    x = w.astype(mx.float32).reshape(rows, cols // g, g)
    mag = mx.abs(x)
    keep = mag > thresh * mag.mean(axis=-1, keepdims=True)
    n = mx.maximum(keep.sum(axis=-1, keepdims=True), 1)
    a = (mag * keep).sum(axis=-1, keepdims=True) / n
    return (mx.sign(x) * keep * a).reshape(rows, cols)


def compress(w, kind):
    """(restored weights in w's dtype, bits per weight in an MLX container whose
    scales and biases have w's dtype)."""
    import mlx.core as mx
    own = 8.0 * w.dtype.size
    if kind in ("fp32", "native"):
        return w, own
    if kind == "fp16":
        return w.astype(mx.float16).astype(w.dtype), min(own, 16.0)
    if kind == "t":
        g = T_GROUP if w.shape[-1] % T_GROUP == 0 else w.shape[-1]
        return ternary(w).astype(w.dtype), 2.0 + 2 * own / g
    bits = int(kind[1:])
    if w.shape[-1] % Q_GROUP:
        return w, own                                        # not packable at this group size
    # in the tensor's own dtype, as QuantizedLinear.from_linear does; restored with
    # float32 scales, which is what the packed matmul computes with
    q, s, b = mx.quantize(w, group_size=Q_GROUP, bits=bits)
    back = mx.dequantize(q, s.astype(mx.float32), b.astype(mx.float32),
                         group_size=Q_GROUP, bits=bits)
    return back.astype(w.dtype), bits + 2 * own / Q_GROUP


def apply(module, kind, skip=None, frame_rows=None):
    """Compress every matrix of `module` in place, except keys `skip` accepts.
    Returns megabytes: as stored, and with a skipped vocabulary table counted as
    `frame_rows` rows (None: the same number)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    new, bits, bits_frame = [], 0.0, 0.0
    for k, v in tree_flatten(module.parameters()):
        if is_matrix(k, v) and not (skip and skip(k)):
            w, bpw = compress(v, kind)
            new.append((k, w))
            bits += bpw * v.size
            bits_frame += bpw * v.size
        else:
            bits += 8.0 * v.nbytes
            if skip and skip(k) and frame_rows is not None and v.ndim == 2:
                bits_frame += 8.0 * v.dtype.size * frame_rows * v.shape[1]
            else:
                bits_frame += 8.0 * v.nbytes
    if new:
        module.update(tree_unflatten(new))
        mx.eval(module.parameters())
    return {"mb": round(bits / 8 / 2**20, 1), "mb_frame": round(bits_frame / 8 / 2**20, 1)}


def vocab_table(key):
    return "embed_tokens" in key


class SharedTrunk:
    """One item's backbone pass, shared by every variant.

    Nothing a variant changes acts below the write layer: the backbone is frozen
    and the bridges write at l_rev. So layers [0, l_rev) run once per item, and a
    variant costs its bridges, its partner and the layers above the write. The
    self-test holds this to PsiConstitutionMLX.logits, the path the evaluator and
    the trainer use.
    """

    def __init__(self, backbone, prompt, cont, l_fwd, l_rev):
        import mlx.core as mx
        from psilm.mlx.staged import MlxStream
        t0 = time.time()
        self.n_prompt, self.n_cont = len(prompt), len(cont)
        self.l_rev, self.n_layers = l_rev, len(backbone.model.layers)
        ids = mx.array([list(prompt) + list(cont)], dtype=mx.int32)
        self.pmask = mx.array([[True] * len(prompt) + [False] * len(cont)])
        self.stream = MlxStream(backbone, ids, None)
        self.stream.run(0, l_fwd)
        self.h_fwd = self.stream.hidden
        self.stream.run(l_fwd, l_rev)
        self.h_rev = self.stream.hidden
        mx.eval(self.h_fwd, self.h_rev)
        self.sec = time.time() - t0

    def tokens(self, psi, soft_zero=False):
        import mlx.core as mx
        soft, _ = psi.phi.fwd(self.h_fwd, self.pmask)
        if soft_zero:
            soft = mx.zeros_like(soft)
        return psi.phi.rev(psi.const.features(soft)), soft

    def logp(self, psi, mode="psilm", tokens=None):
        """log-probabilities at the positions predicting the continuation (T, V),
        and the gate, for one system. `tokens` overrides what it would inject."""
        import mlx.core as mx
        if tokens is None:
            tokens, _ = self.tokens(psi)
        self.stream.hidden, sigma, _ = psi._apply_inject(self.h_rev, tokens, mode)
        self.stream.run(self.l_rev, self.n_layers)
        lo = self.n_prompt - 1
        lg = self.stream.finish()[0, lo:lo + self.n_cont].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        g = sigma[0, :, 0].astype(mx.float32)
        gate = {"cont": g[self.n_prompt:].mean(), "all": g.mean()}
        mx.eval(lp, gate)
        return lp, {k: float(v) for k, v in gate.items()}


def cont_logp(psi, prompt, cont, mode="psilm"):
    """The same through PsiConstitutionMLX.logits: the reference the trunk is held to."""
    import mlx.core as mx
    ids = mx.array([list(prompt) + list(cont)], dtype=mx.int32)
    pmask = mx.array([[True] * len(prompt) + [False] * len(cont)])
    lg, sigma, _ = psi.logits(ids, None, pmask, mode)
    lo = len(prompt) - 1
    lg = lg[0, lo:lo + len(cont)].astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    g = sigma[0, :, 0].astype(mx.float32)
    gate = {"cont": g[len(prompt):].mean(), "all": g.mean()}
    mx.eval(lp, gate)
    return lp, {k: float(v) for k, v in gate.items()}


def score(lp, lp_ref, cont):
    import mlx.core as mx
    tgt = mx.array(list(cont), dtype=mx.int32)
    ce = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0].mean()
    kl = (mx.exp(lp_ref) * (lp_ref - lp)).sum(axis=-1).mean()
    same = (mx.argmax(lp, axis=-1) == mx.argmax(lp_ref, axis=-1)).astype(mx.float32).mean()
    agree = (mx.argmax(lp, axis=-1) == tgt).astype(mx.float32).mean()
    mx.eval(ce, kl, same, agree)
    return {"ce": float(ce), "kl": max(float(kl), 0.0), "same_top1": float(same), "agree": float(agree)}


def mean(v):
    return sum(v) / len(v)


def paired(d, draws=2000, seed=0):
    """Mean of paired differences with a normal 95% interval and a bootstrap one."""
    n = len(d)
    m = mean(d)
    sd = (sum((x - m) ** 2 for x in d) / max(n - 1, 1)) ** 0.5
    rng = random.Random(seed)
    bs = sorted(mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return {"mean": round(m, 6), "ci95": [round(m - 1.96 * sd / n ** 0.5, 6), round(m + 1.96 * sd / n ** 0.5, 6)],
            "boot95": [round(bs[int(0.025 * draws)], 6), round(bs[int(0.975 * draws) - 1], 6)]}


def evaluate(backbone, tok, meta, make_bridges, make_partner, sets, neg_sets, variants,
             log=print, rows_path=None):
    """sets: {name: constitution items (prompt_ids, teacher_ids)}; neg_sets: {name:
    no-harm items (prompt_ids, target_ids)}. Returns (table, extras)."""
    import mlx.core as mx
    from psilm.mlx.constitution import PsiConstitutionMLX

    l_fwd, l_rev = int(meta["l_fwd"]), int(meta["l_rev"])
    partners, bridges, mb = {}, {}, {"bridges": {}, "partner": {}}
    n_frame = None
    for bk, pk in variants:
        if pk not in partners:
            partners[pk] = make_partner()
            p = partners[pk]
            n_frame = len(set(p.prefix_ids.tolist()) | set(p.suffix_ids.tolist()))
            mb["partner"][pk] = apply(p.model, pk, skip=vocab_table, frame_rows=n_frame)
        if bk not in bridges:
            bridges[bk] = make_bridges()
            mb["bridges"][bk] = apply(bridges[bk], bk)
    systems = {v: PsiConstitutionMLX(backbone, tok, partners[v[1]], bridges[v[0]],
                                     l_fwd=l_fwd, l_rev=l_rev) for v in variants}
    ref_key = variants[0]
    ref = systems[ref_key]
    names = [f"{b}+{p}" for b, p in variants] + list(CONTROLS)
    acc = {n: {} for n in names}
    fh = open(rows_path, "a") if rows_path else None
    work = [(s, it, "teacher_ids") for s, items in sets.items() for it in items]
    work += [(s, it, "target_ids") for s, items in neg_sets.items() for it in items]
    prev_tokens, prev_set, soft_example, trunk_sec = None, None, None, []
    t0 = time.time()
    for n, (name, it, key) in enumerate(work):
        prompt, cont = it["prompt_ids"], it[key]
        trunk = SharedTrunk(backbone, prompt, cont, l_fwd, l_rev)
        trunk_sec.append(trunk.sec)
        tokens_ref, soft = trunk.tokens(ref)
        mx.eval(tokens_ref, soft)
        if soft_example is None:
            soft_example = soft
        lp_ref, g_ref = trunk.logp(ref, tokens=tokens_ref)
        lp_base, g_base = trunk.logp(ref, mode="zeroed", tokens=tokens_ref)
        ce_base = score(lp_base, lp_ref, cont)["ce"]
        out = {}
        for v in variants:
            lp, g = (lp_ref, g_ref) if v == ref_key else trunk.logp(systems[v])
            out[f"{v[0]}+{v[1]}"] = (lp, g)
        out["base"] = (lp_base, g_base)
        if prev_tokens is not None and prev_set == name:       # the first item of a set has no donor
            out["shuffle"] = trunk.logp(ref, tokens=prev_tokens)
        zero_tokens, _ = trunk.tokens(ref, soft_zero=True)
        out["softzero"] = trunk.logp(ref, tokens=zero_tokens)
        for vn, (lp, g) in out.items():
            r = {**score(lp, lp_ref, cont), "gate_cont": g["cont"], "gate_all": g["all"],
                 "kl_to_base": score(lp_base, lp, cont)["kl"], "ce_base": ce_base}
            acc[vn].setdefault(name, []).append(r)
            if fh:
                fh.write(json.dumps({"set": name, "source": it.get("source") or it.get("qid"),
                                     "variant": vn, **{k: round(x, 6) for k, x in r.items()}}) + "\n")
        if fh:
            fh.flush()
        prev_tokens, prev_set = tokens_ref, name
        del out, lp_ref, lp_base
        mx.clear_cache()
        if (n + 1) % 10 == 0:
            log(f"  [{n + 1}/{len(work)}] {(time.time() - t0) / (n + 1):.1f}s/item "
                f"(trunk {mean(trunk_sec):.2f}s)")
    if fh:
        fh.close()

    ref_name = f"{ref_key[0]}+{ref_key[1]}"
    con = list(sets)
    table = []
    for vn in names:
        is_var = "+" in vn
        row = {"variant": vn}
        if is_var:
            b, p = vn.split("+")
            row.update(bridges=b, partner=p, bridges_mb=mb["bridges"][b]["mb"],
                       partner_mb=mb["partner"][p]["mb"], partner_mb_frame=mb["partner"][p]["mb_frame"])
        for s, rs in acc[vn].items():
            cell = {k: round(mean([r[k] for r in rs]), 6) for k in rs[0]}
            cell["n"] = len(rs)
            cell["gate_prompt_max"] = round(max(r["gate_all"] for r in rs), 5)
            refs = acc[ref_name][s][-len(rs):]                  # shuffle skips a set's first item
            cell["d_ce"] = paired([a["ce"] - b["ce"] for a, b in zip(rs, refs)])
            gain = mean([r["ce_base"] - b["ce"] for r, b in zip(rs, refs)])
            cell["gain_retained"] = round(mean([r["ce_base"] - r["ce"] for r in rs]) / gain, 4) if gain else None
            row[s] = cell
        if is_var and vn != ref_name and con:
            pooled = [a["ce"] - b["ce"] for s in con for a, b in zip(acc[vn][s], acc[ref_name][s])]
            row["d_ce_pooled"] = paired(pooled)
        table.append(row)

    # ---- the thresholds ------------------------------------------------------------
    T = THRESHOLDS
    by = {r["variant"]: r for r in table}
    held = "noharm_heldout" if "noharm_heldout" in neg_sets else next(iter(neg_sets), None)
    for r in table:
        if "+" not in r["variant"]:
            continue
        ref_r, base_r = by[ref_name], by["base"]
        t = {"T1_ce": all(r[s]["d_ce"]["mean"] <= T["d_ce_max"] for s in con)
             and (r["variant"] == ref_name or r["d_ce_pooled"]["boot95"][1] <= T["d_ce_pooled_upper95"]),
             "T2_kl": all(r[s]["kl"] <= T["kl_max"]
                          and r[s]["kl"] <= T["kl_share_of_effect"] * base_r[s]["kl"] for s in con),
             "T3_top1": all(r[s]["same_top1"] >= 0.99 for s in con),
             "T4_gate": all(abs(r[s]["gate_cont"] - ref_r[s]["gate_cont"]) <= T["d_gate_cont"] for s in con)}
        if held:
            h, hr = r[held], ref_r[held]
            t["T5_noharm"] = (h["gate_all"] <= max(T["noharm_gate_mean"], hr["gate_all"] + 0.002)
                              and h["gate_prompt_max"] <= max(T["noharm_gate_prompt"], hr["gate_prompt_max"])
                              and h["kl"] <= T["noharm_kl"])
        r["thresholds"] = t
        r["ship"] = all(v for k, v in t.items() if k != "T3_top1")      # T3 is secondary
    extras = {"n_frame_rows": n_frame, "trunk_sec": round(mean(trunk_sec), 3),
              "soft_example": soft_example, "systems": systems, "partner_mb": mb["partner"],
              "bridges_mb": mb["bridges"]}
    return table, extras


def partner_packed(make_partner, extras, backbone, meta, items, kinds=("q8", "q4"), reps=5):
    """The partner's layers really packed (mlx.nn.quantize): latency of its one call
    per reply on the forward bridge's real tokens, memory, and, at the system
    level, its distance from the restored-dense stand-in the sweep used."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from psilm.mlx.constitution import PsiConstitutionMLX
    soft = extras["soft_example"]
    systems = extras["systems"]
    ref = systems[next(iter(systems))]
    l_fwd, l_rev = int(meta["l_fwd"]), int(meta["l_rev"])

    def timed(p):
        mx.eval(p.features(soft))                       # warm-up
        t0 = time.time()
        for _ in range(reps):
            mx.eval(p.features(soft))
        return (time.time() - t0) / reps

    def nbytes(p):
        return sum(v.nbytes for _, v in tree_flatten(p.model.parameters()))

    out = {"per_reply_calls": 1, "backbone_trunk_sec": extras["trunk_sec"],
           "native": {"ms": round(1e3 * timed(ref.const), 2), "mb": round(nbytes(ref.const) / 2**20, 3)}}
    for kind in kinds:
        packed = make_partner()
        nn.quantize(packed.model, group_size=Q_GROUP, bits=int(kind[1:]),
                    class_predicate=lambda _, m: isinstance(m, nn.Linear))   # the layers, not the table
        mx.eval(packed.model.parameters())
        psi = PsiConstitutionMLX(backbone, ref.tok, packed, ref.phi, l_fwd=l_fwd, l_rev=l_rev)
        stand = systems.get(("fp32", kind))
        kls, kls_ref = [], []
        for it in items:
            trunk = SharedTrunk(backbone, it["prompt_ids"], it["teacher_ids"], l_fwd, l_rev)
            lp_p, _ = trunk.logp(psi)
            lp_r, _ = trunk.logp(ref)
            kls_ref.append(score(lp_p, lp_r, it["teacher_ids"])["kl"])
            if stand is not None:
                lp_s, _ = trunk.logp(stand)
                kls.append(score(lp_p, lp_s, it["teacher_ids"])["kl"])
            mx.clear_cache()
        out[kind] = {"ms": round(1e3 * timed(packed), 2), "mb": round(nbytes(packed) / 2**20, 3),
                     "kl_trained_to_packed": round(mean(kls_ref), 7),
                     "kl_standin_to_packed": round(mean(kls), 7) if kls else None}
        del packed, psi
        mx.clear_cache()
    return out


def fmt(table, con, neg):
    lines = []
    head = f"{'variant':14s} {'bridges+partner MB':>20s}"
    for s in con:
        head += f" | {s.replace('constitution_', '')[:14]:14s} {'ce':>7s} {'d_ce':>8s} {'kept':>5s} {'kl':>8s} {'top1':>5s} {'gate':>6s}"
    for s in neg:
        head += f" | {s[:14]:14s} {'gate':>7s} {'max':>6s} {'kl':>8s}"
    lines.append(head + " | ship")
    for r in table:
        size = f"{r['bridges_mb']:.0f}+{r['partner_mb_frame']:.0f}" if "bridges" in r else ""
        ln = f"{r['variant']:14s} {size:>20s}"
        for s in con:
            x = r.get(s)
            ln += (f" | {'':14s} {x['ce']:7.4f} {x['d_ce']['mean']:+8.4f} {x['gain_retained'] or 0:5.2f} "
                   f"{x['kl']:8.5f} {x['same_top1']:5.3f} {x['gate_cont']:6.3f}") if x else " | " + " " * 58
        for s in neg:
            x = r.get(s)
            ln += (f" | {'':14s} {x['gate_all']:7.4f} {x['gate_prompt_max']:6.3f} {x['kl']:8.6f}") if x else " | " + " " * 37
        lines.append(ln + (f" | {'yes' if r['ship'] else 'no'}" if "ship" in r else " |"))
    return "\n".join(lines)


def heldout_negatives(rows_path, cache_path, per_dataset, max_cont):
    """Benchmark prompts the gate never trained on, each with the backbone's own
    continuation from the guard-rail's base arm."""
    prompts = {t["qid"]: t["prompt"]["ids"] for t in json.loads(Path(cache_path).read_text())["tasks"]}
    out, seen = [], {}
    for line in Path(rows_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ds = r["dataset"]
        if r["arm"] != "base" or ds == "redteam" or seen.get(ds, 0) >= per_dataset:
            continue
        if r["qid"] in prompts and r.get("gen_ids"):
            seen[ds] = seen.get(ds, 0) + 1
            out.append({"qid": r["qid"], "prompt_ids": prompts[r["qid"]],
                        "target_ids": r["gen_ids"][:max_cont]})
    return out


def run(args):
    import mlx.core as mx
    from psilm.mlx.constitution import (ConstitutionBridgesMLX, ConstitutionModelMLX,
                                        load_constitution_bridge_weights)
    from psilm.mlx.gemma_loader import load_backbone_any

    ckpt = Path(args.ckpt)
    meta = json.loads(Path(str(ckpt) + ".meta").read_text())
    out = Path(args.out) if args.out else ckpt.parent / "compress.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = Path(str(meta.get("data", ""))).stem.split("_")[-1]
    data_paths = [Path(p) for p in args.data.split(",")]
    for p in data_paths + ([Path(args.noharm)] if args.noharm else []):
        if tag and tag not in p.stem.split("_"):
            raise SystemExit(f"{p}: not this checkpoint's data (it was trained on the '{tag}' files)")
    partner_path = args.const_model or meta["const_model"]
    sets = {p.stem: json.loads(p.read_text())[:args.n] for p in data_paths}
    neg_sets = {}
    if args.heldout_rows:
        held = heldout_negatives(args.heldout_rows, args.tasks_cache, args.n_heldout, args.max_cont)
        if held:
            neg_sets["noharm_heldout"] = held
        else:                                   # rows written before generations were recorded
            print(f"[WARN] {args.heldout_rows}: no base-arm generations to score; no held-out set",
                  flush=True)
    if args.noharm:
        neg = json.loads(Path(args.noharm).read_text())
        neg_sets["noharm_train"] = neg[::max(1, len(neg) // args.n_noharm)][:args.n_noharm]
    variants = [tuple(v.split("+")) for v in args.variants.split(",")] if args.variants else VARIANTS

    tower, _, tok = load_backbone_any(args.model or meta["model"])
    tower.freeze()
    assert int(tower.args.hidden_size) == int(meta["d_model"])

    def make_partner():
        return ConstitutionModelMLX(partner_path, m_tokens=int(meta["m_tokens"]))

    def make_bridges():
        b = ConstitutionBridgesMLX(
            d_model=int(meta["d_model"]), d_const=int(meta["d_const"]),
            write_idx=meta["write_dims"], read_idx=meta["read_dims"],
            k_fwd=int(meta["k_fwd"]), m_tokens=int(meta["m_tokens"]),
            gate_bias=float(meta["gate_bias"]), inj_cap=meta["inj_cap"],
            readout_norm=meta["readout_norm"], emb_rms=float(meta["emb_rms"]),
            d_hidden=int(meta.get("d_hidden", 512)))
        load_constitution_bridge_weights(b, ckpt)
        b.freeze()
        mx.eval(b.parameters())
        return b

    rows_path = out.with_suffix(".rows.jsonl")
    rows_path.unlink(missing_ok=True)
    print(f"[compress] {len(variants)} variants + {len(CONTROLS)} controls x "
          f"{ {k: len(v) for k, v in {**sets, **neg_sets}.items()} }", flush=True)
    table, extras = evaluate(tower, tok, meta, make_bridges, make_partner, sets, neg_sets, variants,
                             log=lambda m: print(m, flush=True), rows_path=rows_path)
    by = {r["variant"]: r for r in table}
    ref = by[f"{variants[0][0]}+{variants[0][1]}"]
    checks = {}
    for s in sets:                       # the trained system and the base are the evaluator's
        ev = ckpt.parent / ("eval_helpful.json" if "helpful" in s else "eval_test.json")
        if ev.exists():
            e = json.loads(ev.read_text())
            if int(e.get("n", -1)) == ref[s]["n"] and Path(str(e.get("data", ""))).stem == s:
                checks[s] = {"psilm_ce": [e["arms"]["psilm"]["ce"], ref[s]["ce"]],
                             "base_ce": [e["arms"]["base"]["ce"], by["base"][s]["ce"]],
                             "reproduced": abs(e["arms"]["psilm"]["ce"] - ref[s]["ce"]) <= 1e-3
                             and abs(e["arms"]["base"]["ce"] - by["base"][s]["ce"]) <= 1e-3}
    report = {"ckpt": str(ckpt), "ckpt_step": meta.get("step"), "model": str(args.model or meta["model"]),
              "partner": str(partner_path), "data": [str(p) for p in data_paths],
              "n": args.n, "sets": {k: len(v) for k, v in {**sets, **neg_sets}.items()},
              "q_group": Q_GROUP, "t_group": T_GROUP, "t_thresh": T_THRESH,
              "thresholds": THRESHOLDS, "evaluator_checks": checks,
              "ok": all(c["reproduced"] for c in checks.values()),
              "partner_frame_rows": extras["n_frame_rows"], "trunk_sec": extras["trunk_sec"],
              "variants": table}
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(fmt(table, list(sets), list(neg_sets)), flush=True)
    print(f"evaluator checks: {json.dumps(checks)}", flush=True)
    if not args.no_real:
        first = next(iter(sets.values()))[:3]
        report["partner_packed"] = partner_packed(make_partner, extras, tower, meta, first)
        print(json.dumps(report["partner_packed"], indent=1), flush=True)
        out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"COMPRESS {'COMPLETE' if report['ok'] else 'COMPLETE, BUT THE EVALUATOR IS NOT REPRODUCED'} "
          f"-> {out}", flush=True)
    return 0 if report["ok"] else 1


def self_test():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx.utils import tree_flatten
    from psilm.mlx.constitution import (ConstitutionModelMLX, PsiConstitutionMLX, build_tiny_stack,
                                        synthetic_items)

    mx.random.seed(0)
    # the transforms
    w = mx.random.normal((8, 256))
    for kind, tol in (("fp16", 1e-3), ("q8", 0.02), ("q4", 0.2), ("q2", 0.8), ("t", 0.8)):
        r, bpw = compress(w, kind)
        rel = float(mx.linalg.norm(r - w) / mx.linalg.norm(w))
        assert r.shape == w.shape and r.dtype == w.dtype and 0 < rel < tol, (kind, rel)
        assert bpw < 32
    t = ternary(w)
    for row in range(2):
        for g in range(2):
            vals = sorted({round(abs(float(x)), 5) for x in t[row, g * T_GROUP:(g + 1) * T_GROUP].tolist()})
            assert len(vals) <= 2 and vals[0] == 0.0, vals          # {0, a}: three levels with the sign
    assert compress(w, "fp32")[0] is w
    odd = mx.random.normal((4, 100))                                 # 64 does not divide 100
    assert compress(odd, "q4")[0] is odd and compress(odd, "t")[0].shape == odd.shape
    errs = [float(mx.linalg.norm(compress(w, k)[0] - w)) for k in ("q8", "q4", "q2")]
    assert errs == sorted(errs), errs
    half = w.astype(mx.bfloat16)                                     # the partner's dtype
    for kind in ("q8", "q4", "t", "fp16"):
        r, bpw = compress(half, kind)
        assert r.dtype == mx.bfloat16 and float(mx.linalg.norm((r - half).astype(mx.float32))) >= 0
    assert compress(half, "q4")[1] == 4.5 and compress(w, "q4")[1] == 5.0 and compress(half, "t")[1] == 2.25
    assert is_matrix("inject.g2.weight", mx.zeros((1, 256))) and not is_matrix("fwd.queries", mx.zeros((8, 64)))
    print("[self-test] transforms: shapes, dtypes (bf16 too), three levels per ternary group, "
          "bits per weight, error grows as bits fall  OK")

    stack, _ = build_tiny_stack()
    before = {k: mx.array(v) for k, v in tree_flatten(stack.bridges.parameters())}
    l_fwd, l_rev = int(stack.meta["l_fwd"]), int(stack.meta["l_rev"])

    def make_partner():
        return ConstitutionModelMLX.from_loaded(copy.deepcopy(stack.const.model), stack.const.tok,
                                                m_tokens=stack.const.m_tokens, path="<tiny>")

    def make_bridges():
        return copy.deepcopy(stack.bridges)

    sets = {"constitution_test_x": synthetic_items(5, seed=5)}
    neg = {"noharm_heldout": synthetic_items(3, seed=6, key="target_ids")}
    variants = [("fp32", "native"), ("fp16", "native"), ("q4", "native"), ("fp32", "fp16"),
                ("fp32", "q8"), ("fp32", "q4"), ("t", "t")]
    # the shared trunk against the evaluator's own forward, both modes
    psi = PsiConstitutionMLX(stack.model, stack.tok, stack.const, stack.bridges, l_fwd=l_fwd, l_rev=l_rev)
    for it in sets["constitution_test_x"][:2]:
        trunk = SharedTrunk(stack.model, it["prompt_ids"], it["teacher_ids"], l_fwd, l_rev)
        for mode in ("psilm", "zeroed", "psilm"):                 # psilm again: the trunk is reusable
            a, ga = trunk.logp(psi, mode)
            b, gb = cont_logp(psi, it["prompt_ids"], it["teacher_ids"], mode)
            assert float(mx.max(mx.abs(a - b))) < 1e-4, (mode, float(mx.max(mx.abs(a - b))))
            assert abs(ga["cont"] - gb["cont"]) < 1e-6 and abs(ga["all"] - gb["all"]) < 1e-6
        z, _ = trunk.logp(psi, "zeroed")
        assert float(mx.max(mx.abs(z - a))) > 1e-4                # and the write does something
        own, _ = trunk.tokens(psi)
        c, _ = trunk.logp(psi, tokens=own)
        assert float(mx.max(mx.abs(c - a))) < 1e-5                # given its own tokens: the same
    print("[self-test] shared trunk == PsiConstitutionMLX.logits in both modes  OK")

    table, extras = evaluate(stack.model, stack.tok, stack.meta, make_bridges, make_partner, sets, neg,
                             variants, log=lambda m: None)
    by = {r["variant"]: r for r in table}
    ref, s = by["fp32+native"], "constitution_test_x"
    assert ref[s]["kl"] == 0.0 and ref[s]["same_top1"] == 1.0 and ref["noharm_heldout"]["kl"] == 0.0
    assert ref[s]["d_ce"]["mean"] == 0.0 and ref[s]["gain_retained"] == 1.0 and ref["ship"]
    assert by["fp16+native"][s]["kl"] < 1e-6 and by["fp16+native"]["ship"]
    assert by["base"][s]["kl"] > 0 and abs(by["base"][s]["gain_retained"]) < 1e-9
    assert by["base"][s]["kl_to_base"] == 0.0 and ref[s]["kl_to_base"] > 0
    assert by["shuffle"][s]["n"] == 4 and by["softzero"][s]["n"] == 5      # shuffle: no donor for the first
    assert by["shuffle"][s]["kl"] > 0 and by["softzero"][s]["kl"] > 0
    assert all(r[s]["kl"] >= 0 and r["noharm_heldout"]["n"] == 3 for r in table if "+" in r["variant"])
    assert by["q4+native"]["bridges_mb"] < by["fp16+native"]["bridges_mb"] < ref["bridges_mb"]
    assert by["fp32+q4"]["partner_mb"] < ref["partner_mb"]
    assert ref["partner_mb_frame"] < ref["partner_mb"]              # the vocabulary table, set aside
    assert "d_ce_pooled" in by["t+t"] and set(by["t+t"]["thresholds"]) >= {"T1_ce", "T2_kl", "T4_gate", "T5_noharm"}
    for k, v in tree_flatten(stack.bridges.parameters()):           # the trained tensors are untouched
        assert bool(mx.all(v == before[k]).item()), k
    # the partner's vocabulary table is never compressed
    emb0 = dict(tree_flatten(stack.const.model.parameters()))["model.embed_tokens.weight"]
    emb1 = dict(tree_flatten(extras["systems"][("t", "t")].const.model.parameters()))["model.embed_tokens.weight"]
    assert bool(mx.all(emb0 == emb1).item())
    print(f"[self-test] evaluate: trained kl 0; fp16 kl {by['fp16+native'][s]['kl']:.1e}; q4 bridges "
          f"{by['q4+native'][s]['kl']:.1e}; q4 partner {by['fp32+q4'][s]['kl']:.1e}; ternary both "
          f"{by['t+t'][s]['kl']:.1e}; base {by['base'][s]['kl']:.1e}; shuffle {by['shuffle'][s]['kl']:.1e}; "
          f"softzero {by['softzero'][s]['kl']:.1e}; originals untouched  OK")
    pq = partner_packed(make_partner, extras, stack.model, stack.meta, sets[s][:2], reps=2)
    assert pq["q4"]["kl_standin_to_packed"] < 1e-5 and pq["q8"]["kl_standin_to_packed"] < 1e-5, pq
    assert pq["q4"]["mb"] < pq["q8"]["mb"] < pq["native"]["mb"]
    print(f"[self-test] the packed partner is the restored-dense stand-in at the system level "
          f"(KL q8 {pq['q8']['kl_standin_to_packed']:.1e}, q4 {pq['q4']['kl_standin_to_packed']:.1e})  OK")
    d = paired([0.001, 0.002, 0.003, 0.002] * 10)
    assert abs(d["mean"] - 0.002) < 1e-9 and d["boot95"][0] <= 0.002 <= d["boot95"][1]
    print(fmt(table, [s], ["noharm_heldout"]))
    print("[self-test] eval/constitution_compress.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--model", default=None)
    ap.add_argument("--const-model", default=None)
    ap.add_argument("--data")
    ap.add_argument("--noharm", default=None, help="the training negatives (a sample is scored)")
    ap.add_argument("--heldout-rows", default=None,
                    help="a guard-rail rows file: its base arm's benchmark generations are the "
                         "held-out no-harm continuations")
    ap.add_argument("--tasks-cache", default=None, help="the guard-rail's task cache (prompt ids)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-noharm", type=int, default=20)
    ap.add_argument("--n-heldout", type=int, default=20, help="per benchmark")
    ap.add_argument("--max-cont", type=int, default=64, help="tokens of a held-out continuation scored")
    ap.add_argument("--variants", default=None,
                    help="comma-separated bridges+partner pairs, e.g. fp32+native,fp16+q4 "
                         "(the first is the reference); default: the built-in sweep")
    ap.add_argument("--no-real", action="store_true", help="skip the packed-partner section")
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/compress.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.ckpt and a.data):
        ap.error("--ckpt and --data are required")
    if a.heldout_rows and not a.tasks_cache:
        ap.error("--heldout-rows needs --tasks-cache")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
