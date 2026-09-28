#!/usr/bin/env python3
"""How far a constitution bridge moves the backbone toward its teacher.

The bridge is trained to make the backbone, WITHOUT the excerpt, predict what the
same backbone writes WITH the excerpt in its system prompt. Scored on the
teacher's stored continuation, teacher-forced:

  base      the backbone alone, no excerpt: where the student starts
  teacher   the backbone with the excerpt in context, on its own greedy tokens
  psilm     the backbone with the bridge, no excerpt

Two fractions, because they answer different questions:

  kl_fraction = 1 - KL(teacher || psilm) / KL(teacher || base)
      how much of the DISTANCE to the teacher's distribution the bridge removes.
      The teacher is exactly 0 under this measure, so 1 is a bound.
  ce_fraction = (base - psilm) / (base - teacher)        (cross-entropies)
      the same on the training objective. NOT bounded by 1: the targets are the
      teacher's argmax tokens, and a student that is more confident on them than
      the teacher was scores BELOW the teacher's own cross-entropy. The teacher's
      CE is a reference, not a floor; items where the student is below it are
      counted.

A control for confidence. Cross-entropy on argmax targets falls when a model is
merely made more confident in the tokens it mostly picks anyway, with nothing of
the teacher's content added. So every arm is also scored at temperatures TAUS
(logits / tau), and the comparison is repeated with each arm at its own best
temperature:

  ce_gain_at_best_tau = min_tau CE(base, tau) - min_tau CE(psilm, tau)
      what the bridge adds that no temperature gives the plain backbone
  sharpening_share = (CE(base, 1) - min_tau CE(base, tau)) / ce_gain
      how much of the bridge's raw gain a temperature alone reproduces

and likewise for the KL to the teacher. With two splits, each split's
temperatures are also chosen on the OTHER split (`crossfit`), so nothing is
tuned on the items it is scored on.

Spans: `all` is the whole stored continuation, which is what the evaluator's CE
column is (eval/mlx_constitution_eval.py); `first24` is its first 24 tokens.

The teacher pass rebuilds the teacher's prompt with the builder's own functions
(eval/build_constitution_data.py) on the builder's prefix cache. What says the
rebuild is right:
  identity     the excerpt's sha256, the system block's token count and the
               backbone equal those the builder recorded in its stats file
  student      the student prompt rebuilt from the item's text is the stored one
  reproduces   a greedy decode of the first items gives the stored continuation;
               and teacher-forced, the stored token is the argmax or within
               TIE_TOL logits of it (half-precision near-ties flip an argmax)
  forwards     the base arm scored on the stock forward (the teacher pass's)
               equals the base arm on the staged forward (the bridge's)
  evaluator    base and psilm reproduce the evaluator's recorded CEs

  python eval/teacher_ceiling.py --ckpt results/stage2c_bonsai27b_all/bridges.npz \\
      --data data/constitution_test_bonsai27b.json,data/constitution_helpful_test_bonsai27b.json
  python eval/teacher_ceiling.py --self-test        # tiny models on the CPU

Writes <checkpoint dir>/teacher_ceiling.json and .rows.jsonl. Run from the
repository root (a checkpoint may record its partner as a relative path).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIRST = 24          # the evaluator's generation budget; its CE column is the `all` span
TIE_TOL = 0.25      # logits: a few half-precision ulps at logit magnitude
TAUS = tuple(round(0.30 + 0.05 * k, 2) for k in range(27))      # 0.30 ... 1.60, 1.0 included
SPANS = ("all", "first24")
ARMS = ("base", "base_stock", "teacher", "psilm")


def position_stats(logits, targets, lp_teacher=None, taus=None):
    """Per-position lists for logits (T, V) against targets (T,): ce, hit (argmax
    is the target), near (the target is within TIE_TOL of the max), p (the
    target's probability) and, given the teacher's log-probs, kl(teacher || this)."""
    import mlx.core as mx
    lg = logits.astype(mx.float32)
    tgt = mx.array(list(targets), dtype=mx.int32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    at = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
    out = {"ce": -at, "p": mx.exp(at),
           "hit": (mx.argmax(lg, axis=-1) == tgt).astype(mx.float32),
           "near": ((lp.max(axis=-1) - at) <= TIE_TOL).astype(mx.float32)}
    if lp_teacher is not None:
        out["kl"] = (mx.exp(lp_teacher) * (lp_teacher - lp)).sum(axis=-1)
        p_teacher = mx.exp(lp_teacher)
    for tau in (taus or ()):
        lt = lg / tau
        lt = lt - mx.logsumexp(lt, axis=-1, keepdims=True)
        out[f"ce@{tau}"] = -mx.take_along_axis(lt, tgt[:, None], axis=-1)[:, 0]
        if lp_teacher is not None:
            out[f"kl@{tau}"] = (p_teacher * (lp_teacher - lt)).sum(axis=-1)
        mx.eval(out[f"ce@{tau}"], out.get(f"kl@{tau}"))
    mx.eval(out, lp)
    return {k: [float(x) for x in v.tolist()] for k, v in out.items()}, lp


def span_stats(pos):
    n_all = len(pos["ce"])
    out = {}
    for span, n in (("all", n_all), ("first24", min(FIRST, n_all))):
        out[span] = {k: sum(v[:n]) / n for k, v in pos.items()}
        out[span]["n"] = n
    return out


def continuation_logits(model, lead, cont, cache=None):
    """Logits at the positions that predict `cont`, given `lead` before it.

    With a cache, `lead` is what follows the cached prefix (at least one token);
    without one it is the whole prompt. The cache is CONSUMED."""
    import mlx.core as mx
    lead, cont = list(lead), list(cont)
    assert lead and cont
    seq = mx.array([lead + cont], dtype=mx.int32)
    out = model(seq, cache=cache) if cache is not None else model(seq)
    lo = len(lead) - 1
    return out[0, lo:lo + len(cont)]


def fractions(rows, span):
    """(ce_fraction, kl_fraction, gain) from item means, unrounded."""
    def m(arm, key):
        v = [r[arm][span][key] for r in rows]
        return sum(v) / len(v)
    b, t, p = m("base", "ce"), m("teacher", "ce"), m("psilm", "ce")
    kb, kp = m("base", "kl"), m("psilm", "kl")
    return ((b - p) / (b - t) if b != t else float("nan"),
            1.0 - kp / kb if kb > 0 else float("nan"), b - p)


def bootstrap(rows, span, draws=2000, seed=0):
    """Paired over items: every arm of an item is resampled together."""
    rng = random.Random(seed)
    n = len(rows)
    got = [fractions([rows[rng.randrange(n)] for _ in range(n)], span) for _ in range(draws)]
    out = {}
    for k, name in enumerate(("ce_fraction", "kl_fraction", "ce_gain")):
        v = sorted(x[k] for x in got if x[k] == x[k])
        out[name] = [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)] if v else None
    return out


def summarize(rows):
    out = {"n": len(rows)}
    for span in SPANS:
        s = {}
        for arm in ARMS:
            for key in ("ce", "hit", "near", "p", "kl"):
                v = [r[arm][span][key] for r in rows if key in r[arm][span]]
                if v:
                    s[f"{arm}_{key}"] = round(sum(v) / len(v), 5)
        cf, kf, gain = fractions(rows, span)
        s.update(ce_gain=round(gain, 5), ce_gap_to_teacher=round(s["base_ce"] - s["teacher_ce"], 5),
                 ce_fraction=round(cf, 4), kl_fraction=round(kf, 4),
                 items_psilm_below_teacher=sum(r["psilm"][span]["ce"] < r["teacher"][span]["ce"] for r in rows))
        if len(rows) >= 10:
            s["ci95"] = bootstrap(rows, span)
        s["temperature"] = temperature(rows, span)
        out[span] = s
    return out


def curve(rows, arm, span, key):
    """Item-mean of `key` at each temperature, or None when the rows carry none."""
    if not rows or f"{key}@{TAUS[0]}" not in rows[0][arm][span]:
        return None
    return [sum(r[arm][span][f"{key}@{t}"] for r in rows) / len(rows) for t in TAUS]


def temperature(rows, span, taus_from=None):
    """Each arm at its best temperature. taus_from: rows of another split to choose
    the temperatures on (cross-fitting); default: these rows."""
    pick = taus_from if taus_from is not None else rows
    out = {}
    for arm in ("base", "psilm"):
        for key in ("ce", "kl"):
            c_pick, c = curve(pick, arm, span, key), curve(rows, arm, span, key)
            if c is None or c_pick is None:
                return None
            k = min(range(len(TAUS)), key=lambda i: c_pick[i])
            out[f"{arm}_{key}"] = {"tau": TAUS[k], "at_tau": round(c[k], 5),
                                   "at_1": round(c[TAUS.index(1.0)], 5),
                                   "at_edge": k in (0, len(TAUS) - 1)}     # the grid did not bracket it
    b, p = out["base_ce"], out["psilm_ce"]
    raw = b["at_1"] - p["at_1"]
    out["ce_gain_raw"] = round(raw, 5)
    out["ce_gain_at_best_tau"] = round(b["at_tau"] - p["at_tau"], 5)
    out["sharpening_share"] = round((b["at_1"] - b["at_tau"]) / raw, 4) if raw else None
    kb, kp = out["base_kl"], out["psilm_kl"]
    out["kl_fraction_at_best_tau"] = round(1 - kp["at_tau"] / kb["at_tau"], 4) if kb["at_tau"] > 0 else None
    out["curves"] = {f"{arm}_{key}": [round(x, 5) for x in curve(rows, arm, span, key)]
                     for arm in ("base", "psilm") for key in ("ce", "kl")}
    out["taus"] = list(TAUS)
    return out


def read_rows(path, ident):
    """Rows of THIS run's identity; a torn last line is dropped and the file rewritten."""
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    rows, torn = [], False
    for k, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except ValueError:
            if k != len(lines) - 1:
                raise
            torn = True
    if torn:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.replace(tmp, path)
        print(f"[WARN] {path}: torn last line removed", flush=True)
    return {(r["split"], r["source"]): r for r in rows if r.get("id") == ident}


def run(args):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from transformers import AutoTokenizer
    from bench_common import chat_ids, eos_id_set
    from build_constitution_data import (excerpt_body, greedy, make_cloner, prefix_ids_of,
                                         teacher_system)
    from psilm.mlx.constitution import PsiConstitutionMLX, load_constitution_stack
    from psilm.mlx.gemma_loader import load_backbone_any

    ckpt = Path(args.ckpt)
    out_path = Path(args.out) if args.out else ckpt.parent / "teacher_ceiling.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out_path.with_suffix(".rows.jsonl")
    data_paths = [Path(p) for p in args.data.split(",")]

    # ---- identity, before any weights load --------------------------------------
    meta = json.loads(Path(str(ckpt) + ".meta").read_text())
    model_id = args.model or meta["model"]
    hf_id = args.hf_tokenizer or meta.get("hf_tokenizer") or model_id
    const_id = args.const_model or meta["const_model"]
    body_sha = hashlib.sha256(excerpt_body(args.excerpt).encode()).hexdigest()
    hf_tok = AutoTokenizer.from_pretrained(hf_id)
    tsys = teacher_system(args.excerpt)
    prefix = prefix_ids_of(hf_tok, tsys)
    ident_checks = {}
    for p in data_paths:
        tag = p.stem.split("_")[-1]
        sp = p.parent / f"constitution_{tag}_stats.json"
        st = json.loads(sp.read_text())
        ident_checks[p.stem] = {
            "stats": str(sp),
            "excerpt_sha256": st["excerpt"]["sha256"] == body_sha,
            "prefix_tokens": int(st["excerpt"]["prefix_tokens"]) == len(prefix),
            "model": os.path.realpath(str(st["model"])) == os.path.realpath(str(model_id)),
            "data_is_the_checkpoints": tag == Path(str(meta.get("data", ""))).stem.split("_")[-1]}
        bad = [k for k, v in ident_checks[p.stem].items() if v is False]
        if bad:
            raise SystemExit(f"{p}: this is not the teacher the data came from ({', '.join(bad)} "
                             f"differ from {sp})")
    ident = hashlib.sha256(json.dumps(
        [os.path.realpath(str(ckpt)), os.path.getmtime(ckpt), os.path.realpath(str(model_id)),
         os.path.realpath(str(const_id)), body_sha, FIRST, TIE_TOL, TAUS]).encode()).hexdigest()[:16]
    done = {} if args.fresh else read_rows(rows_path, ident)
    if args.fresh:
        rows_path.unlink(missing_ok=True)

    bridges, const, meta = load_constitution_stack(ckpt, args.const_model)
    tower, stock, mlx_tok = load_backbone_any(model_id)
    tower.freeze()
    assert int(tower.args.hidden_size) == int(meta["d_model"])
    eos_ids = eos_id_set(hf_tok, mlx_tok)
    psi = PsiConstitutionMLX(tower, mlx_tok, const, bridges, l_fwd=int(meta["l_fwd"]),
                             l_rev=int(meta["l_rev"]))
    t0 = time.time()
    prefix_cache = make_prompt_cache(stock)
    stock(mx.array([list(prefix)]), cache=prefix_cache)
    mx.eval([c.state for c in prefix_cache])
    clone, how = make_cloner(stock, prefix_cache)
    print(f"[teacher] system block {len(prefix)} tokens prefilled in {time.time() - t0:.1f}s, "
          f"clone: {how}", flush=True)

    report = {"id": ident, "ckpt": str(ckpt), "ckpt_step": meta.get("step"), "model": str(model_id),
              "hf_tokenizer": str(hf_id), "const_model": str(const_id), "excerpt": args.excerpt,
              "excerpt_sha256": body_sha, "prefix_tokens": len(prefix), "first": FIRST,
              "tie_tol": TIE_TOL, "n_per_split": args.n, "identity": ident_checks, "splits": {}}
    ok_all, all_rows = True, {}
    with rows_path.open("a") as fh:
        for path in data_paths:
            items = json.loads(path.read_text())[:args.n]
            split = path.stem
            rows, greedy_check = [], []
            for i, it in enumerate(items):
                key = (split, it["source"])
                prompt, cont = list(it["prompt_ids"]), list(it["teacher_ids"])
                tids = chat_ids(hf_tok, it["prompt_text"], system=tsys)
                assert tids[:len(prefix)] == prefix, f"{it['source']}: the prefix does not match"
                lead = tids[len(prefix):]
                if i < args.verify_n:
                    # the builder's own decode: the stored continuation, token for token
                    got = greedy(stock, lead, len(cont), eos_ids, clone())
                    d = next((j for j, (x, y) in enumerate(zip(got, cont)) if x != y),
                             None if len(got) == len(cont) else min(len(got), len(cont)))
                    greedy_check.append({"source": it["source"], "first_divergence": d, "n": len(cont)})
                if key in done:
                    rows.append(done[key])
                    continue
                t1 = time.time()
                r = {"id": ident, "split": split, "source": it["source"], "n_cont": len(cont),
                     "student_prompt_rebuilt": chat_ids(hf_tok, it["prompt_text"]) == prompt}
                pos, lpt = position_stats(continuation_logits(stock, lead, cont, clone()), cont)
                r["teacher"] = span_stats(pos)
                pos, _ = position_stats(continuation_logits(stock, prompt, cont), cont, lpt)
                r["base_stock"] = span_stats(pos)
                ids = mx.array([prompt + cont], dtype=mx.int32)
                pmask = mx.array([[True] * len(prompt) + [False] * len(cont)])
                for arm in ("base", "psilm"):
                    full, _, _ = psi.logits(ids, None, pmask, arm)
                    lo = len(prompt) - 1
                    pos, _ = position_stats(full[0, lo:lo + len(cont)], cont, lpt, TAUS)
                    r[arm] = span_stats(pos)
                r["sec"] = round(time.time() - t1, 2)
                del lpt
                mx.clear_cache()
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (i + 1) % 10 == 0:
                    s = summarize(rows)["all"]
                    print(f"  [{split} {i + 1}/{len(items)}] CE base {s['base_ce']:.4f} teacher "
                          f"{s['teacher_ce']:.4f} psilm {s['psilm_ce']:.4f} | KL to teacher base "
                          f"{s['base_kl']:.4f} psilm {s['psilm_kl']:.4f} -> kl_fraction "
                          f"{s['kl_fraction']} | {r['sec']}s/item", flush=True)
            s = summarize(rows)
            a = s["all"]
            checks = {
                "student_prompts_rebuilt": sum(bool(r["student_prompt_rebuilt"]) for r in rows) == len(rows),
                "forwards_agree": abs(a["base_ce"] - a["base_stock_ce"]) <= 2e-3,
                "teacher_reproduces_itself": a["teacher_near"] >= 0.97,
                "greedy_decode": greedy_check}
            ev = ckpt.parent / ("eval_helpful.json" if "helpful" in split else "eval_test.json")
            if ev.exists():
                e = json.loads(ev.read_text())
                if int(e.get("n", -1)) == len(rows) and Path(str(e.get("data", ""))).stem == split:
                    checks["evaluator"] = {
                        "base_ce": [e["arms"]["base"]["ce"], a["base_ce"]],
                        "psilm_ce": [e["arms"]["psilm"]["ce"], a["psilm_ce"]],
                        "reproduced": abs(e["arms"]["base"]["ce"] - a["base_ce"]) <= 1e-3
                        and abs(e["arms"]["psilm"]["ce"] - a["psilm_ce"]) <= 1e-3}
            ok = (checks["student_prompts_rebuilt"] and checks["forwards_agree"]
                  and checks["teacher_reproduces_itself"]
                  and checks.get("evaluator", {}).get("reproduced", True))
            ok_all &= ok
            report["splits"][split] = {**s, "checks": checks, "ok": ok}
            all_rows[split] = rows
    if len(all_rows) == 2:                       # temperatures chosen on the other split
        (a, ra), (b, rb) = all_rows.items()
        for span in SPANS:
            report["splits"][a][span]["temperature_crossfit"] = temperature(ra, span, taus_from=rb)
            report["splits"][b][span]["temperature_crossfit"] = temperature(rb, span, taus_from=ra)
    report["ok"] = ok_all
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    for split, s in report["splits"].items():
        a = s["all"]
        print(f"{split}: CE base {a['base_ce']:.4f} teacher {a['teacher_ce']:.4f} psilm "
              f"{a['psilm_ce']:.4f} (ce_fraction {a['ce_fraction']}, student below the teacher on "
              f"{a['items_psilm_below_teacher']}/{s['n']}) | KL to the teacher: base {a['base_kl']:.4f} "
              f"psilm {a['psilm_kl']:.4f} (kl_fraction {a['kl_fraction']}, 95% "
              f"{(a.get('ci95') or {}).get('kl_fraction')}) | checks {json.dumps(s['checks'])}")
        t = a.get("temperature_crossfit") or a.get("temperature")
        if t:
            print(f"   temperature: base CE {t['base_ce']['at_1']:.4f} -> {t['base_ce']['at_tau']:.4f} at tau "
                  f"{t['base_ce']['tau']}; psilm {t['psilm_ce']['at_1']:.4f} -> {t['psilm_ce']['at_tau']:.4f} "
                  f"at tau {t['psilm_ce']['tau']} | CE gain raw {t['ce_gain_raw']:.4f}, at best tau "
                  f"{t['ce_gain_at_best_tau']:.4f} (sharpening share {t['sharpening_share']}) | KL to the "
                  f"teacher at best tau: base {t['base_kl']['at_tau']:.4f} psilm {t['psilm_kl']['at_tau']:.4f} "
                  f"(kl_fraction {t['kl_fraction_at_best_tau']})")
    print(f"TEACHER-CEILING {'OK' if ok_all else 'CHECKS FAILED'} -> {out_path}", flush=True)
    return 0 if ok_all else 1


def self_test():
    """The slicing, the cache and the builder's decode on tiny random models, CPU."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen2 import Model as Q2, ModelArgs as Q2A
    from mlx_lm.models.qwen3_5 import Model as Q35, ModelArgs as Q35A
    from build_constitution_data import greedy, make_cloner

    mx.random.seed(0)
    V = 256
    q2 = Q2(Q2A(model_type="qwen2", hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=V, num_key_value_heads=2,
                max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=True))
    tc = {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
          "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
          "head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": V, "linear_num_value_heads": 4,
          "linear_num_key_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4, "tie_word_embeddings": False, "full_attention_interval": 4,
          "attn_output_gate": True,
          "rope_parameters": {"mrope_interleaved": True, "mrope_section": [1, 1, 0],
                              "partial_rotary_factor": 0.25, "rope_theta": 10000,
                              "rope_type": "default"}}
    q35 = Q35(Q35A.from_dict({"model_type": "qwen3_5", "text_config": tc}))
    for name, model in (("kv-cache", q2), ("hybrid", q35)):
        model.eval()
        mx.eval(model.parameters())
        prefix = [int(x) for x in mx.random.randint(2, V, (40,)).tolist()]
        cache = make_prompt_cache(model)
        model(mx.array([prefix]), cache=cache)
        mx.eval([c.state for c in cache])
        clone, how = make_cloner(model, cache)
        worst = 0.0
        for k in range(3):                       # three prompts off ONE prefilled prefix
            tail = [int(x) for x in mx.random.randint(2, V, (5 + k,)).tolist()]
            cont = [int(x) for x in mx.random.randint(2, V, (9,)).tolist()]
            a = continuation_logits(model, tail, cont, clone())
            b = continuation_logits(model, prefix + tail, cont)
            mx.eval(a, b)
            assert a.shape == b.shape == (len(cont), V), (a.shape, b.shape)
            worst = max(worst, float(mx.max(mx.abs(a - b)) / mx.max(mx.abs(b))))
            # the slice: position j of the returned logits scores cont[j]
            ref = model(mx.array([prefix + tail + cont[:4]]))[0, -1]
            assert float(mx.max(mx.abs(ref - b[4]))) / float(mx.max(mx.abs(ref))) < 1e-4
            # the builder's token-by-token decode is what the one-pass scoring scores:
            # teacher-forced on the greedy continuation, every argmax is the stored token
            g = greedy(model, tail, 12, set(), clone())
            pos, _ = position_stats(continuation_logits(model, tail, g, clone()), g)
            assert sum(pos["hit"]) == len(g) and sum(pos["near"]) == len(g), (name, pos["hit"])
            one, _ = position_stats(continuation_logits(model, tail, g[:1], clone()), g[:1])
            assert one["hit"] == [1.0]                               # a one-token continuation
        assert worst < 1e-3, (name, worst)
        print(f"[self-test] {name}: cached == uncached continuation logits (worst relative "
              f"{worst:.1e}); the builder's greedy decode is reproduced (clone {how.split(' ')[0]})  OK")

    lg = mx.array([[2.0, 1.9, -5.0], [0.0, 3.0, 0.0]])
    pos, lp = position_stats(lg, [1, 1])
    assert pos["hit"] == [0.0, 1.0] and pos["near"] == [1.0, 1.0]       # 0.1 below the max: a near-tie
    kl_self, _ = position_stats(lg, [1, 1], lp)
    assert max(abs(x) for x in kl_self["kl"]) < 1e-6                    # KL to itself is 0

    def item(b, t, p, kb, kp, n=30):
        def arm(ce, kl):
            return span_stats({"ce": [ce] * n, "hit": [1.0] * n, "near": [1.0] * n, "p": [0.5] * n,
                               "kl": [kl] * n})
        return {"base": arm(b, kb), "base_stock": arm(b, kb), "teacher": arm(t, 0.0), "psilm": arm(p, kp)}
    rows = [item(2.0, 1.0, 1.25, 0.8, 0.2), item(2.0, 1.0, 0.75, 0.8, 0.2)]
    s = summarize(rows)["all"]
    assert s["ce_fraction"] == 1.0 and s["kl_fraction"] == 0.75 and s["items_psilm_below_teacher"] == 1
    assert rows[0]["base"]["first24"]["n"] == FIRST and rows[0]["base"]["all"]["n"] == 30
    ci = bootstrap(rows * 6, "all", draws=200)
    assert ci["kl_fraction"] == [0.75, 0.75] and ci["ce_fraction"][0] <= 1.0 <= ci["ce_fraction"][1]
    print("[self-test] near-ties, KL to self, fractions (the CE one can pass 1), bootstrap  OK")

    # temperature: a student that is the base made sharper gains CE and no content
    mx.random.seed(1)
    T_, V_ = 200, 50
    base_lg = mx.random.normal((T_, V_)) * 2.0
    teach_lg = base_lg + mx.random.normal((T_, V_)) * 1.0          # a teacher near the base
    tgt = [int(x) for x in mx.argmax(teach_lg, axis=-1).tolist()]
    _, lpt = position_stats(teach_lg, tgt)
    arms = {"base": base_lg, "base_stock": base_lg, "psilm": base_lg / 0.8}     # "bridge" = tau 0.8
    row = {"teacher": span_stats(position_stats(teach_lg, tgt)[0])}
    for k, lg_ in arms.items():
        row[k] = span_stats(position_stats(lg_, tgt, lpt, TAUS if k in ("base", "psilm") else None)[0])
    t = temperature([row], "all")
    assert t["ce_gain_raw"] > 0.01                                  # the sharpened copy "gains" CE
    assert abs(t["ce_gain_at_best_tau"]) < 0.02                     # ... and nothing at its best tau
    assert t["sharpening_share"] > 0.9, t
    assert TAUS[0] < t["base_ce"]["tau"] < TAUS[-1] and not t["base_ce"]["at_edge"]   # an interior optimum
    assert abs(t["psilm_ce"]["tau"] * 0.8 - t["base_ce"]["tau"]) < 0.06  # psilm's best tau = base's / 0.8
    assert abs(row["base"]["all"]["ce@1.0"] - row["base"]["all"]["ce"]) < 1e-6
    assert temperature([row], "all", taus_from=[row])["ce_gain_raw"] == t["ce_gain_raw"]
    print(f"[self-test] temperature: a sharpened base gains {t['ce_gain_raw']:.3f} raw and "
          f"{t['ce_gain_at_best_tau']:+.3f} at best tau (sharpening share {t['sharpening_share']})  OK")

    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "r.jsonl"
    tmp.write_text(json.dumps({"id": "a", "split": "s", "source": "1"}) + "\n"
                   + json.dumps({"id": "b", "split": "s", "source": "2"}) + "\n" + '{"id": "a", "spl')
    got = read_rows(tmp, "a")
    assert list(got) == [("s", "1")] and len(tmp.read_text().splitlines()) == 2
    print("[self-test] resume keeps this run's rows only and survives a torn line  OK")
    print("[self-test] eval/teacher_ceiling.py: all assertions passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--model", default=None, help="backbone (default: the checkpoint's own)")
    ap.add_argument("--hf-tokenizer", default=None)
    ap.add_argument("--const-model", default=None)
    ap.add_argument("--data", help="comma-separated data JSONs (each split scored separately)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--verify-n", type=int, default=2,
                    help="items per split whose stored continuation is re-decoded greedily")
    ap.add_argument("--excerpt", default="data/constitution/system_excerpt.md")
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/teacher_ceiling.json")
    ap.add_argument("--fresh", action="store_true", help="discard rows already written")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.ckpt and a.data):
        ap.error("--ckpt and --data are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
