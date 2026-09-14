"""The causal check: zero the value neurons and watch GSM8K accuracy fall.

Stage 3 of the value-neuron identification of arXiv:2602.00986. The paper's
evidence that the top ~1% of dimensions really are the reward subsystem is not
the probe's AUC -- it is that zeroing them collapses accuracy (75 -> 20 on
MATH500) while zeroing the same number of random dimensions does nothing. This
script runs that contrast on the backbone itself.

Arms, all greedy, all on the same weights and the same prompts:
    base            untouched
    zeroV           the top-1% dimensions of layer<l>.json zeroed at layer l
    zeroV5          the top-5% ones (--also-top5pct)
    rand<k>         the same number of dimensions drawn uniformly from the
                    complement of the zeroed set, three seeds

"Zeroed at layer l" means: the residual stream ENTERING block l has those
dimensions multiplied by zero at EVERY position, prompt and generated, which is
where and how the constitution bridge will write. The shim that does it
(psilm.mlx.value_neurons.ZeroDimsShim) wraps block l-1 and counts its own calls;
every item asserts that the count and the positions seen match the decode exactly,
so an arm can never silently run as the base arm.

This is the ONE script in the value-neuron pipeline that reads the GSM8K test
split: it selects nothing and trains nothing (the layer and the dimensions come
from eval/vn_probe.py on the train split), it only reports. n=100 / seed 0 is the
guard-rail's own selection (eval/bench_common.load_gsm8k), so the base arm here is
comparable to the guard-rail's gsm8k row.

Resumable: rows are appended to ablate_layer<l>.jsonl as they are produced and an
(arm, qid) already there is skipped, so a five-arm run that dies at arm four
resumes rather than repeating 40 minutes of decoding.

Usage:
  .venv/bin/python eval/vn_ablate.py --tag qwen0.5b \
      --neurons results/value_neurons/qwen0.5b/layer15.json --n 100 --max-new 384
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402

from eval.bench_common import (  # noqa: E402
    append_jsonl, chat_ids, eos_id_set, gsm8k_user, load_gsm8k, paired, read_jsonl, score,
)
from psilm.mlx.value_neurons import (  # noqa: E402
    StreamDecoder, install_zero_shim, remove_zero_shim,
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_HF_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="qwen0.5b")
    ap.add_argument("--out-root", default="results/value_neurons")
    ap.add_argument("--neurons", required=True, help="results/value_neurons/<tag>/layer<l>.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hf-tokenizer", default=DEFAULT_HF_TOKENIZER)
    ap.add_argument("--n", type=int, default=100, help="GSM8K test questions (seeded prefix)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--random-seed-base", type=int, default=20_000)
    ap.add_argument("--random-exclude-sink", action="store_true",
                    help="keep attention-sink dims (huge at position 0 only) out of the random "
                         "pool; see psilm.mlx.value_neurons.sink_dims")
    ap.add_argument("--random-arm-prefix", default="rand")
    ap.add_argument("--out-suffix", default="",
                    help="write ablate_layer<l><suffix>.{json,jsonl} (a second run with other controls)")
    ap.add_argument("--also-top5pct", action="store_true")
    ap.add_argument("--arms", default="", help="restrict to these arms (comma separated)")
    ap.add_argument("--nudge", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    rec = json.loads(Path(args.neurons).read_text())
    layer = int(rec["layer"])
    d_model = int(rec["d_model"])
    top1 = sorted(int(i) for i in rec["top1pct"])
    top5 = sorted(int(i) for i in rec["top5pct"])
    root = Path(args.out_root) / args.tag
    root.mkdir(parents=True, exist_ok=True)
    rows_path = root / f"ablate_layer{layer}{args.out_suffix}.jsonl"

    # the random controls: same count, drawn from the complement of the zeroed set
    # so the control can never accidentally include a value neuron; optionally
    # also from the complement of the attention-sink dims (see sink_dims)
    excluded = set(top1)
    sinks: List[int] = []
    if args.random_exclude_sink:
        from psilm.mlx.value_neurons import sink_dims
        sinks = sink_dims(root, layer)
        excluded |= set(sinks)
        print(f"random pool excludes the sink dims {sinks}")
    pool = np.array([i for i in range(d_model) if i not in excluded], dtype=np.int64)
    arms: Dict[str, List[int]] = {"base": []}
    arms["zeroV"] = top1
    if args.also_top5pct:
        arms["zeroV5"] = top5
    for s in range(args.random_seeds):
        rs = np.random.default_rng(args.random_seed_base + s).choice(pool, size=len(top1), replace=False)
        arms[f"{args.random_arm_prefix}{s}"] = sorted(int(i) for i in rs)
    want = [a for a in args.arms.split(",") if a.strip()] or list(arms)
    arms = {a: arms[a] for a in want}

    from transformers import AutoTokenizer
    from psilm.mlx.gemma_loader import load_backbone_any

    items = load_gsm8k(args.n, args.seed)
    hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    model, _stock, mlx_tok = load_backbone_any(args.model)
    model.freeze()
    assert int(model.args.hidden_size) == d_model, (model.args.hidden_size, d_model)
    assert 1 <= layer <= len(model.model.layers), (layer, len(model.model.layers))
    dec = StreamDecoder(model, hf_tok, eos_id_set(hf_tok, mlx_tok))
    prompts = {it["qid"]: chat_ids(hf_tok, gsm8k_user(it["question"], nudge=bool(args.nudge)))
               for it in items}

    print(f"model={args.model} layer={layer} d_model={d_model} n={len(items)} "
          f"max_new={args.max_new} arms={list(arms)}")
    print(f"zeroV dims ({len(top1)}): {top1}")
    done = {(r["arm"], r["qid"]) for r in read_jsonl(rows_path)}
    if done:
        print(f"resuming: {len(done)} (arm, qid) rows already on disk")

    t_all = time.perf_counter()
    for arm, dims in arms.items():
        shim = install_zero_shim(model, layer, dims, d_model) if dims else None
        n_new, n_ok = 0, 0
        t0 = time.perf_counter()
        for it in items:
            if (arm, it["qid"]) in done:
                continue
            before = (shim.calls, shim.positions) if shim else (0, 0)
            out = dec.rollout(prompts[it["qid"]], max_new=args.max_new)
            pred, ok = score("number", out["text"], it["gold"])
            if shim:
                # the shim MUST be on the path: one call for the prefill plus one
                # per generated position fed back in, and every position of each
                calls = shim.calls - before[0]
                pos = shim.positions - before[1]
                assert calls == 1 + out["n_gen_states"], (arm, calls, out["n_gen_states"])
                assert pos == out["prompt_len"] + out["n_gen_states"], (arm, pos)
            append_jsonl(rows_path, {
                "arm": arm, "qid": it["qid"], "pred": pred, "gold": it["gold"],
                "ok": bool(ok), "n_gen": len(out["gen_ids"]),
                "stopped_eos": bool(out["stopped_eos"]), "sec": round(out["sec"], 2),
                "n_dims": len(dims), "text": out["text"][-220:]})
            n_new += 1
            n_ok += int(ok)
            if n_new % 10 == 0 or n_new == 1:
                el = time.perf_counter() - t0
                print(f"  {arm}: {n_new} items {el / 60:.1f}min acc={n_ok / n_new:.3f} "
                      f"{len(out['gen_ids']) / max(1e-9, out['sec']):.1f}tok/s", flush=True)
            del out
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        if shim:
            print(f"  {arm}: shim calls={shim.calls} positions={shim.positions} "
                  f"dims={len(dims)}")
            remove_zero_shim(model, shim)
        print(f"  {arm}: done new={n_new} {(time.perf_counter() - t0) / 60:.1f}min", flush=True)

    rows = read_jsonl(rows_path)
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in arms}
    summary = {"tag": args.tag, "layer": layer, "d_model": d_model, "model": args.model,
               "neurons": str(args.neurons), "n": len(items), "seed": args.seed,
               "max_new": args.max_new, "split": "gsm8k:test", "arms": {},
               "dims": {a: list(v) for a, v in arms.items()},
               "sec_total": round(time.perf_counter() - t_all, 1)}
    for a, rs in by_arm.items():
        if not rs:
            continue
        summary["arms"][a] = {
            "n": len(rs), "n_dims": len(arms[a]),
            "acc": round(sum(r["ok"] for r in rs) / len(rs), 4),
            "n_correct": int(sum(r["ok"] for r in rs)),
            "parse_rate": round(sum(r["pred"] is not None for r in rs) / len(rs), 4),
            "mean_n_gen": round(float(np.mean([r["n_gen"] for r in rs])), 1),
            "eos_rate": round(float(np.mean([r["stopped_eos"] for r in rs])), 3),
            "sec_per_q": round(float(np.mean([r["sec"] for r in rs])), 2)}
    summary["paired_vs_base"] = {a: paired(by_arm["base"], by_arm[a])
                                 for a in by_arm if a != "base" and by_arm.get("base") and by_arm[a]}
    summary["sink_dims_excluded"] = sinks
    rnd = [summary["arms"][a]["acc"] for a in summary["arms"] if a.startswith(args.random_arm_prefix)]
    summary["random_acc_mean"] = (round(float(np.mean(rnd)), 4) if rnd else None)
    summary["rows"] = [{k: v for k, v in r.items() if k != "text"} for r in rows]
    (root / f"ablate_layer{layer}{args.out_suffix}.json").write_text(json.dumps(summary, indent=1))

    print(f"\nlayer {layer}, GSM8K test n={len(items)} seed={args.seed}, greedy, "
          f"max_new={args.max_new}")
    print(f"{'arm':8s} {'dims':>5s} {'n':>4s} {'acc':>7s} {'parse':>7s} {'gen':>7s} "
          f"{'eos':>6s} {'McNemar p':>10s}")
    print("-" * 62)
    for a in summary["arms"]:
        b = summary["arms"][a]
        p = summary["paired_vs_base"].get(a, {}).get("mcnemar_p")
        print(f"{a:8s} {b['n_dims']:5d} {b['n']:4d} {b['acc']:7.3f} {b['parse_rate']:7.3f} "
              f"{b['mean_n_gen']:7.1f} {b['eos_rate']:6.3f} {p!s:>10s}")
    if rnd:
        print(f"random-dims mean acc = {summary['random_acc_mean']}")
    print(f"VN-ABLATE DONE layer={layer} arms={len(summary['arms'])}")


if __name__ == "__main__":
    main()
