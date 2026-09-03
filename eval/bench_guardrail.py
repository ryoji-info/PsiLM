"""Gate-selectivity guard-rail benchmark for PsiLM.

Claim under test: attaching the bridges does not damage the backbone's
general capability, and the injection gate is selective - near-closed on
non-physics questions, open on the PsiLM physics questions it was trained on.

Three arms on the SAME backbone weights, same prompts, greedy decoding:
    base    frozen LLM alone
    psilm   bridges attached, gate free
    zeroed  bridges attached, physics injection multiplied by 0
            (gate sigma still measured -> arm 3 vs arm 1 is the
            numerical-noise floor, arm 2 vs arm 3 is the injection's effect)

Datasets (one report, one table):
    gsm8k    openai/gsm8k test, seeded prefix of N      "Answer: <number>"
    mmlu     cais/mmlu, fixed 5-subject slice, N total  "Answer: <letter>"
    physics  data/stage2_qa_val.json, first N           trained reply (acc@0.05)
             (base arm uses the stage2_eval nudge protocol by default)

Usage
  # prompts only, no model weights (safe while the GPU is busy):
  python eval/bench_guardrail.py --tag v5_8b --dry-run
  # CPU-only decoder equivalence test on a synthetic tiny backbone:
  python eval/bench_guardrail.py --tag x --self-test
  # real run (loads the 4-bit backbone; resumable):
  python eval/bench_guardrail.py --tag v5_8b --l-rev 27 [--resume]

Outputs
  results/bench/<tag>_guardrail.json          summary + gate table + rows
  results/bench/<tag>_guardrail.rows.jsonl    one row per (question, arm), for --resume
  results/bench/<tag>_guardrail_dryrun.json   prompt manifest (dry run)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.bench_common import (  # noqa: E402
    ARMS, DEFAULT_CKPT, DEFAULT_FNO, DEFAULT_HF_TOKENIZER, DEFAULT_MODEL, MMLU_SUBJECTS,
    PHYSICS_DATA, StagedDecoder, Task, append_jsonl, build_tasks, eos_id_set, estimate_budget,
    format_table, load_backbone, load_gsm8k, load_mmlu, load_physics, load_physics_stack,
    parse_letter, parse_number, read_jsonl, score, sigma_stats, summarize, task_manifest,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="results/bench/<tag>_guardrail.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hf-tokenizer", default=DEFAULT_HF_TOKENIZER)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="bridges .npz (with .meta beside it)")
    ap.add_argument("--fno", default=DEFAULT_FNO)
    ap.add_argument("--l-fwd", type=int, default=None)
    ap.add_argument("--l-rev", type=int, default=None,
                    help="MUST match training (8B v5 used 27); default = PsiLMMLX rule")
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    ap.add_argument("--n", type=int, default=100, help="questions per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", default="physics,mmlu,gsm8k",
                    help="cheap/high-signal first so a partial run is already useful")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--mmlu-subjects", default=",".join(MMLU_SUBJECTS))
    ap.add_argument("--physics-data", default=PHYSICS_DATA)
    ap.add_argument("--physics-base-protocol", default="nudge", choices=["nudge", "trained"],
                    help="base arm on physics: stage2_eval nudge (+160 tok budget) or the trained prompt")
    ap.add_argument("--nonphys-span", default="whole", choices=["whole", "learned"],
                    help="x0 pointer on questions with no x0: whole-prompt mean "
                         "(= QABuilder fallback) or the learned attention pointer")
    ap.add_argument("--max-new-gsm8k", type=int, default=384)
    ap.add_argument("--max-new-mmlu", type=int, default=24)
    ap.add_argument("--max-new-physics", type=int, default=32)
    ap.add_argument("--max-new-physics-base", type=int, default=160)
    ap.add_argument("--gate-open-thresh", type=float, default=0.1,
                    help="a question counts as 'gate open' when its mean sigma exceeds this")
    ap.add_argument("--save-sigma-trace", action="store_true",
                    help="store per-position sigma lists in the rows (bigger JSON)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, print 3/dataset, no weights")
    ap.add_argument("--self-test", action="store_true", help="CPU-only decoder/parser tests, no weights")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="discard an existing rows file")
    ap.add_argument("--no-parity-check", action="store_true")
    ap.add_argument("--gsm8k-nudge", type=int, default=1,
                    help="0: drop the 'Answer:' nudge from GSM8K prompts (wrapper-cue probe for the gate)")
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--text-chars", type=int, default=1200)
    return ap.parse_args()


# ----------------------------------------------------------------------------
# task construction (tokenizer only)
# ----------------------------------------------------------------------------

def build_all(args, hf_tok):
    from psilm.stage2.qa import QABuilder
    builder = QABuilder(hf_tok)
    datasets = [d for d in args.datasets.split(",") if d]
    tasks = []
    for ds in datasets:
        t0 = time.time()
        if ds == "gsm8k":
            recs = load_gsm8k(args.n, args.seed)
            tasks += build_tasks("gsm8k", recs, hf_tok, args.max_new_gsm8k,
                                 nonphys_span=args.nonphys_span, gsm8k_nudge=bool(args.gsm8k_nudge))
        elif ds == "mmlu":
            recs = load_mmlu(args.n, args.seed, args.mmlu_subjects.split(","))
            tasks += build_tasks("mmlu", recs, hf_tok, args.max_new_mmlu,
                                 nonphys_span=args.nonphys_span)
        elif ds == "physics":
            recs = load_physics(args.n, args.physics_data)
            tasks += build_tasks("physics", recs, hf_tok, args.max_new_physics, builder=builder,
                                 physics_base_protocol=args.physics_base_protocol,
                                 max_new_physics_base=args.max_new_physics_base)
        else:
            raise SystemExit(f"unknown dataset {ds}")
        print(f"[tasks] {ds}: {sum(t.dataset == ds for t in tasks)} questions ({time.time() - t0:.1f}s)",
              flush=True)
    return datasets, tasks


def dataset_stats(tasks, ds):
    ts = [t for t in tasks if t.dataset == ds]
    lens = [len(t.prompt.ids) for t in ts]
    d = {"n": len(ts), "prompt_tokens_mean": round(sum(lens) / max(1, len(lens)), 1),
         "prompt_tokens_max": max(lens, default=0), "prompt_tokens_min": min(lens, default=0)}
    if ds == "physics":
        d["span_fallbacks"] = sum(t.prompt.span_fallback for t in ts)
        d["span_len_mean"] = round(sum(t.prompt.x0_span[1] - t.prompt.x0_span[0] for t in ts) / max(1, len(ts)), 2)
    if ds == "mmlu":
        d["per_subject"] = {}
        for t in ts:
            d["per_subject"][t.meta["subject"]] = d["per_subject"].get(t.meta["subject"], 0) + 1
    return d


def do_dry_run(args, tasks, datasets, hf_tok, out_path: Path):
    arms = args.arms.split(",")
    print("\n" + "=" * 78 + "\nDRY RUN - prompts only, no model weights loaded\n" + "=" * 78)
    for ds in datasets:
        st = dataset_stats(tasks, ds)
        print(f"\n### {ds}: {json.dumps(st)}")
        for k, t in enumerate([t for t in tasks if t.dataset == ds][:3]):
            m = task_manifest(t, hf_tok)
            print(f"\n--- [{ds} example {k + 1}] {t.qid}  gold={t.gold!r}  n_tok={m['n_tokens']}  "
                  f"protocol={m['protocol']}  max_new={m['max_new']}  x0_span={m['x0_span']}"
                  + (f"  span_text={m['span_text']!r}" if "span_text" in m else ""))
            txt = t.prompt.text
            print(txt if len(txt) <= 900 else txt[:600] + f"\n ... [{len(txt) - 900} chars] ...\n" + txt[-300:])
            if t.arm_prompts:
                for arm, p in t.arm_prompts.items():
                    print(f"   [{arm} arm override] protocol={p.protocol} n_tok={len(p.ids)} max_new={p.max_new}")
                    if k == 0:
                        print("   " + p.text.replace("\n", "\n   "))
    budget = estimate_budget(tasks, arms)
    print(f"\n[budget] {json.dumps(budget)}")
    print("[note] tps assumptions are placeholders for 8B-4bit; calibrate from the first rows.")
    fallbacks = sum(t.prompt.span_fallback for t in tasks if t.dataset == "physics")
    if fallbacks:
        print(f"[WARN] {fallbacks} physics prompts hit the QABuilder x0_span whole-prompt fallback")
    manifest = {"tag": args.tag, "dry_run": True, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": vars(args), "arms": arms, "datasets": datasets,
                "stats": {ds: dataset_stats(tasks, ds) for ds in datasets},
                "budget": budget,
                "tasks": [task_manifest(t, hf_tok, full_text=True) for t in tasks]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=1))
    print(f"[dry-run] manifest -> {out_path}  ({len(tasks)} tasks)")


# ----------------------------------------------------------------------------
# real run
# ----------------------------------------------------------------------------

def make_row(t: Task, arm: str, prompt, res, pred, ok, args):
    row = {"dataset": t.dataset, "qid": t.qid, "arm": arm, "gold": t.gold, "pred": pred, "ok": ok,
           "protocol": prompt.protocol, "n_prompt": len(prompt.ids), "n_gen": len(res.gen_ids),
           "stopped_eos": res.stopped_eos, "sec": round(res.sec, 3),
           "text": res.text[-args.text_chars:], "text_truncated": len(res.text) > args.text_chars,
           "sigma": sigma_stats(res.sigma_prompt, res.sigma_gen) if arm != "base" else None,
           "diag": res.diag, "meta": {k: v for k, v in t.meta.items() if k != "item"}}
    if t.dataset == "physics":
        row["meta"]["item"] = t.meta["item"]
        row["x0_span"] = prompt.x0_span
    if args.save_sigma_trace and arm != "base":
        row["sigma_prompt"] = [round(v, 5) for v in res.sigma_prompt]
        row["sigma_gen"] = [round(v, 5) for v in res.sigma_gen]
    return row


def write_report(path: Path, args, rows, datasets, arms, extra):
    s = summarize(rows, datasets, arms, args.gate_open_thresh)
    rep = {"tag": args.tag, "dry_run": False, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "config": vars(args), "arms": arms, "datasets": datasets, **extra,
           "summary": s["summary"], "gate_table": s["gate_table"],
           "table_text": format_table(s["summary"], arms), "n_rows": len(rows), "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rep, indent=1))
    tmp.replace(path)
    return rep


def coupling_from_log(ckpt: str):
    """The bridges .meta does not record the layer split; the trainer prints
    'coupling <l_fwd>/<l_rev> of <n>' into supervisor.log next to it."""
    import re
    log = Path(ckpt).parent / "supervisor.log"
    if not log.exists():
        return None
    hits = re.findall(r"coupling (\d+)/(\d+) of (\d+)", log.read_text())
    return tuple(int(v) for v in hits[-1]) if hits else None


def resolve_coupling(args, n_layers: int):
    rule = (round(n_layers * 10 / 24), round(n_layers * 15 / 24))
    logged = coupling_from_log(args.ckpt)
    # v6+ checkpoint meta records the coupling depths; prefer them over the log parse
    meta_p = Path(str(args.ckpt) + ".meta")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if "l_rev" in meta and "l_fwd" in meta and not logged:
        logged = (int(meta["l_fwd"]), int(meta["l_rev"]), n_layers)
    l_fwd = args.l_fwd if args.l_fwd is not None else (logged[0] if logged else rule[0])
    l_rev = args.l_rev if args.l_rev is not None else (logged[1] if logged else rule[1])
    if logged and logged[2] != n_layers:
        print(f"[WARN] supervisor.log coupling {logged} is for a {logged[2]}-layer model, backbone has {n_layers}")
    elif logged and (l_fwd, l_rev) != logged[:2]:
        print(f"[WARN] coupling {l_fwd}/{l_rev} differs from the training log {logged[0]}/{logged[1]}")
    elif not logged and (args.l_fwd is None or args.l_rev is None):
        print(f"[WARN] no --l-rev and no supervisor.log beside {args.ckpt}: using the rule {rule} "
              "(8B v5 trained with l_rev=27)")
    print(f"[couple] l_fwd={l_fwd} l_rev={l_rev} n_layers={n_layers} (log: {logged})", flush=True)
    return l_fwd, l_rev


def do_run(args, tasks, datasets, hf_tok, report_path: Path, rows_path: Path):
    arms = args.arms.split(",")
    if rows_path.exists() and not args.resume:
        if args.fresh:
            rows_path.unlink()
        else:
            raise SystemExit(f"{rows_path} exists: pass --resume to continue or --fresh to discard")
    rows = read_jsonl(rows_path) if args.resume else []
    done = {(r["dataset"], r["qid"], r["arm"]) for r in rows}
    if rows:
        print(f"[resume] {len(rows)} rows already done", flush=True)

    t0 = time.time()
    model, tok, _ = load_backbone(args.model, args.hf_tokenizer)
    d_model = model.args.hidden_size
    fno, bridges, meta = load_physics_stack(args.ckpt, d_model, args.gate_bias, args.fno)
    l_fwd, l_rev = resolve_coupling(args, len(model.model.layers))
    dec = StagedDecoder(model, hf_tok, fno, bridges, l_fwd, l_rev,
                        eos_ids=eos_id_set(hf_tok, tok))
    info = {"model": args.model, "ckpt": args.ckpt, "ckpt_step": meta.get("step"),
            "ckpt_model": meta.get("model"), "couple": [dec.l_fwd, dec.l_rev, dec.n_layers],
            "d_model": d_model, "eos_ids": sorted(dec.eos_ids), "load_sec": round(time.time() - t0, 1),
            "selection": {ds: [t.qid for t in tasks if t.dataset == ds] for ds in datasets},
            "protocols": {ds: sorted({t.prompt_for(a).protocol + f"@{t.prompt_for(a).max_new}"
                                      for t in tasks if t.dataset == ds for a in arms})
                          for ds in datasets}}
    print(f"[model] {args.model} loaded in {info['load_sec']}s | bridges step {meta.get('step')} | "
          f"couple {dec.l_fwd}/{dec.l_rev} of {dec.n_layers} | d_model {d_model}", flush=True)
    if meta.get("model") and meta["model"] != args.model:
        print(f"[WARN] checkpoint was trained on {meta['model']}, running {args.model}", flush=True)
    if not args.no_parity_check:
        diff, rel, same = dec.parity(tasks[0].prompt.ids)
        info["parity_max_abs"], info["parity_rel"], info["parity_argmax_same"] = diff, rel, same
        print(f"[parity] staged base vs stock model logits: max|diff| = {diff:.3e} "
              f"(relative {rel:.2e}, argmax same: {same})", flush=True)
        if rel > 1e-2 or not same:
            raise SystemExit("parity failed: staged base arm is not the stock model")

    for ti, t in enumerate(tasks):
        for arm in arms:
            key = (t.dataset, t.qid, arm)
            if key in done:
                continue
            p = t.prompt_for(arm)
            span = p.x0_span if arm != "base" else None
            res = dec.generate(p.ids, mode=arm, max_new=p.max_new, x0_span=span)
            pred, ok = score(p.protocol, res.text, t.gold)
            row = make_row(t, arm, p, res, pred, ok, args)
            append_jsonl(rows_path, row)
            rows.append(row)
            done.add(key)
        if (ti + 1) % args.print_every == 0 or ti == len(tasks) - 1:
            sub = [r for r in rows if r["dataset"] == t.dataset]
            run = {a: (round(sum(r["ok"] for r in sub if r["arm"] == a) /
                             max(1, sum(r["arm"] == a for r in sub)), 3)) for a in arms}
            sig = {a: (round(sum(r["sigma"]["all_mean"] for r in sub if r["arm"] == a and r["sigma"]) /
                             max(1, sum(r["arm"] == a for r in sub)), 4)) for a in arms if a != "base"}
            tps = sum(r["n_gen"] for r in rows) / max(1e-9, sum(r["sec"] for r in rows))
            print(f"  [{t.dataset} {ti + 1}/{len(tasks)}] acc={run} sigma={sig} "
                  f"{tps:.1f} tok/s  {(time.time() - t0) / 60:.1f} min", flush=True)
        if (ti + 1) % args.save_every == 0:
            write_report(report_path, args, rows, datasets, arms, info)

    rep = write_report(report_path, args, rows, datasets, arms, info)
    print("\n" + rep["table_text"])
    print(f"\nFINAL -> {report_path}", flush=True)


# ----------------------------------------------------------------------------
# self-test: CPU-only, synthetic tiny backbone, no weights
# ----------------------------------------------------------------------------

def self_test():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from psilm.mlx.bridges import PsiBridgesMLX
    from psilm.mlx.fno import FNO1dMLX
    from psilm.mlx.model import PsiLMMLX
    from psilm.mlx.staged import MlxStream

    class FakeTok:
        eos_token_id = 1
        unk_token_id = None

        def encode(self, s, add_special_tokens=False):
            return [0]

        def convert_tokens_to_ids(self, t):
            return -1

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    args = ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                     num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=256, num_key_value_heads=2,
                     max_position_embeddings=512, rope_theta=10000.0, head_dim=16, tie_word_embeddings=True)
    model = Model(args)
    model.freeze()
    fno = FNO1dMLX()
    bridges = PsiBridgesMLX(64, gate_bias=0.0)
    # make the injection material so cached-vs-full equivalence is a real test
    bridges.inject.to_out.weight = 0.3 * mx.random.normal(bridges.inject.to_out.weight.shape)
    bridges.inject.g2.weight = 0.02 * mx.random.normal(bridges.inject.g2.weight.shape)  # unsaturated gate
    mx.eval(model.parameters(), fno.parameters(), bridges.parameters())
    tok = FakeTok()
    dec = StagedDecoder(model, tok, fno, bridges, l_fwd=1, l_rev=3, eos_ids={1})
    prompt = [int(v) for v in mx.random.randint(2, 256, (23,)).tolist()]
    K = 12

    # 1. parity: staged base prefill == stock forward
    d, rel, same = dec.parity(prompt)
    assert d < 1e-3 and same, f"parity {d}"
    print(f"[self-test] parity base vs stock: {d:.2e}  OK")

    # 2. cached coupled decode == full-recompute (PsiLMMLX._couple) decode
    psi = PsiLMMLX(model, tok, fno, bridges, l_fwd=1, l_rev=3)
    for span in ((0, len(prompt)), (5, 9), None):
        res = dec.generate(prompt, mode="psilm", max_new=K, x0_span=span)
        ids = list(prompt)
        ref_ids, ref_sigma = [], None
        span_mx = None if span is None else mx.array([list(span)], dtype=mx.int32)
        for _ in range(K):
            t = mx.array([ids])
            pmask = mx.concatenate([mx.ones((1, len(prompt)), dtype=mx.bool_),
                                    mx.zeros((1, len(ids) - len(prompt)), dtype=mx.bool_)], axis=1)
            s = MlxStream(model, t)
            _, _, _, _, sigma, _ = psi._couple(s, pmask, span_mx)
            nxt = int(s.finish()[:, -1].argmax(-1).item())
            ref_sigma = sigma[0, :, 0]
            ids.append(nxt)
            ref_ids.append(nxt)
            if nxt == 1:
                break
        assert res.gen_ids == ref_ids, (span, res.gen_ids, ref_ids)
        n_in = len(prompt) + len(res.sigma_gen)
        ref_sig = [float(v) for v in ref_sigma[:n_in].tolist()]
        got_sig = res.sigma_prompt + res.sigma_gen
        err = max(abs(a - b) for a, b in zip(ref_sig, got_sig))
        assert err < 1e-3, (span, err)
        spread = max(got_sig) - min(got_sig)
        assert 0.02 < spread and 0.02 < sum(got_sig) / len(got_sig) < 0.98, f"gate saturated: {got_sig}"
        print(f"[self-test] cached==full-recompute span={span}: {len(res.gen_ids)} tokens, "
              f"sigma max|diff|={err:.2e}, mean sigma={sum(got_sig) / len(got_sig):.3f} "
              f"(spread {spread:.3f})  OK")

    # 3. zeroed arm reproduces the base arm exactly; psilm differs
    rb = dec.generate(prompt, mode="base", max_new=K)
    rz = dec.generate(prompt, mode="zeroed", max_new=K, x0_span=(0, len(prompt)))
    rp = dec.generate(prompt, mode="psilm", max_new=K, x0_span=(0, len(prompt)))
    assert rb.gen_ids == rz.gen_ids, (rb.gen_ids, rz.gen_ids)
    assert rz.sigma_prompt and rz.sigma_gen, "zeroed arm must still measure the gate"
    lb, *_ = dec.prefill_logits(prompt, "base")
    lz, *_ = dec.prefill_logits(prompt, "zeroed", (0, len(prompt)))
    lp, *_ = dec.prefill_logits(prompt, "psilm", (0, len(prompt)))
    dz = float(mx.abs(lb - lz).max().item())
    dp = float(mx.abs(lb - lp).max().item())
    assert dz == 0.0, f"zeroed arm changed the hidden state: {dz}"
    assert dp > 1e-3, f"psilm arm did not change the logits: {dp}"
    print(f"[self-test] zeroed == base (logits diff {dz:.1e}, {len(rb.gen_ids)} tokens); "
          f"psilm logits diff {dp:.2e}, tokens differ: {rp.gen_ids != rb.gen_ids}  OK")

    # 4. parsers / scoring
    assert parse_number("so 16-3-4=9 eggs.\nAnswer: 18") == 18.0
    assert parse_number("**Answer:** $1,234.50") == 1234.5
    assert parse_number("Answer: \\boxed{42}") == 42.0
    assert parse_number("blah 3 then 7.") == 7.0
    assert parse_number("u at x = 0.86 equals -0.10.", fallback="decimal") == -0.10
    assert parse_letter("Answer: C") == "C" and parse_letter("**Answer:** (b)") == "B"
    assert parse_letter("A duck walks.\nB. yes") == "B" and parse_letter("nothing") is None
    assert score("physics_trained", "u at x = 0.86 equals -0.10.", -0.095) == (-0.10, True)
    assert score("number", "Answer: 18.00", 18.0) == (18.0, True)
    assert score("letter", "The answer is option C.", "C") == ("C", True)
    print("[self-test] parsers OK")
    print("SELF-TEST OK")


# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    out_dir = Path("results/bench")
    report_path = out_dir / f"{args.tag}_guardrail.json"
    rows_path = out_dir / f"{args.tag}_guardrail.rows.jsonl"
    dry_path = out_dir / f"{args.tag}_guardrail_dryrun.json"

    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    datasets, tasks = build_all(args, hf_tok)
    for ds in datasets:
        print(f"[stats] {ds}: {json.dumps(dataset_stats(tasks, ds))}", flush=True)
    if args.dry_run:
        do_dry_run(args, tasks, datasets, hf_tok, dry_path)
        return
    do_run(args, tasks, datasets, hf_tok, report_path, rows_path)


if __name__ == "__main__":
    main()
