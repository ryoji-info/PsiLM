#!/usr/bin/env python3
"""What must hold before a bridge is trained on Ternary Bonsai, checked once on the GPU.

Run by results/bonsai/constitution_bonsai.sh after the data and the no-harm set
exist and before any smoke run. Each check is one that cost, or would cost, days
if it failed only after training:

  data     the four splits are the 9B's prompts in the 9B's order, the teacher
           differs from the base often enough to learn from, and the no-harm set
           is the 9B's 1,194 prompts slot for slot (997 distinct) with Bonsai's
           own short targets
  stream   the training forward (MlxStream on a right-padded batch of 2) gives the
           stock model's last-position logits for each row, on the Metal scan and
           again with the differentiable ops scan above layer 48 (training numerics)
  grad     the recomputed backward (MlxStream.checkpoint_from) gives the taped
           gradient on the real model
  memory   recomputing actually lowers the backward peak on the GPU and keeps it
           flat as the window grows (reported; a failure here is logged, not fatal:
           the long-batch smokes still bound the peak)
  packed   the input gradient through a packed ternary projection, in the training
           dtype at a training-like row count, matches a dense dequantized reference
           on the GPU (and the CPU, where its kernels run)

  python results/bonsai/precheck.py               # writes results/bonsai/precheck.json
  python results/bonsai/precheck.py --no-model    # data half only -> precheck_data.json

Exit 0 when every fatal check passes, 1 otherwise.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PACK = ("/Users/rxiii/Documents/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/"
        "snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787")
TAG, REF = "bonsai27b", "qwen35"
SPLITS = {"train": 1000, "val": 100, "test": 100, "helpful_test": 100}
NOHARM_MAX_NEW = 32


def load(p):
    return json.loads((ROOT / p).read_text())


def check_data(rep):
    ok = True
    for s, n in SPLITS.items():
        mine, ref = load(f"data/constitution_{s}_{TAG}.json"), load(f"data/constitution_{s}_{REF}.json")
        same_order = [r["source"] for r in mine] == [r["source"] for r in ref]
        same_ids = sum(a["prompt_ids"] == b["prompt_ids"] for a, b in zip(mine, ref))
        empty = sum(1 for r in mine if not r.get("teacher_ids") or not r.get("base_ids"))
        good = len(mine) == n and same_order and same_ids == n and empty == 0
        rep["data"][s] = {"n": len(mine), "same_order_as_9b": same_order, "same_prompt_ids": same_ids,
                          "empty_continuations": empty, "ok": good}
        ok &= good
    st, st_ref = load(f"data/constitution_{TAG}_stats.json"), load(f"data/constitution_{REF}_stats.json")
    keys = ("teacher_differs_rate", "refusal_rate_base", "refusal_rate_teacher", "mean_len_teacher")
    rep["data"]["stats"] = {s: {k: v.get(k) for k in keys} for s, v in st["splits"].items()}
    rep["data"]["stats_9b"] = {s: {k: v.get(k) for k in keys} for s, v in st_ref["splits"].items()}
    differs = st["splits"]["train"]["teacher_differs_rate"]
    rep["data"]["teacher_signal_ok"] = differs >= 0.2          # the 9B's is 0.94
    ok &= rep["data"]["teacher_signal_ok"]

    neg, prior = load(f"data/noharm_{TAG}_all.json"), load(f"data/noharm_{REF}_all.json")
    slots = sum(a["source"] == b["source"] and a["prompt_ids"] == b["prompt_ids"] for a, b in zip(neg, prior))
    lens = [len(r["target_ids"]) for r in neg]
    eos = {248044, 248046}
    # a target shorter than the budget ends in the stop id the builder appends; one that
    # stops early mid-sentence would mean the text re-encoded shorter than it generated
    early = sum(1 for r in neg if 5 <= len(r["target_ids"]) <= NOHARM_MAX_NEW
                and r["target_ids"][-1] in eos and not r["target_text"].rstrip().endswith((".", "!", "?", ")")))
    good = (len(neg) == len(prior) and slots == len(prior)
            and len({tuple(r["prompt_ids"]) for r in neg}) == len({tuple(r["prompt_ids"]) for r in prior})
            and min(lens) >= 1 and max(lens) <= NOHARM_MAX_NEW + 1)
    rep["noharm"] = {"n": len(neg), "n_9b": len(prior), "slots_matching": slots,
                     "distinct_prompts": len({tuple(r["prompt_ids"]) for r in neg}),
                     "target_len_min": min(lens), "target_len_max": max(lens),
                     "target_len_hist": {str(k): lens.count(k) for k in sorted(set(lens))},
                     "early_eos_mid_sentence": early, "ok": good}
    return ok and good


def rel(a, b):
    import mlx.core as mx
    return float(mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(b)), 1e-9))


def check_model(rep):
    import mlx.core as mx
    from psilm.mlx.gemma_loader import load_backbone_any
    from psilm.mlx.staged import MlxStream
    from psilm.mlx.constitution import pad_batch

    t0 = time.time()
    tower, model, _ = load_backbone_any(PACK)
    n = len(tower.model.layers)
    rep["model"] = {"load_sec": round(time.time() - t0, 1), "n_layers": n,
                    "hidden_dtype": None, "active_gb": round(mx.get_active_memory() / 2**30, 2)}
    ok = True

    # -- stream: right-padded batch of two real training items of different lengths
    items = sorted(load(f"data/constitution_train_{TAG}.json")[:40],
                   key=lambda r: len(r["prompt_ids"]) + len(r["teacher_ids"]))
    pair = [items[0], items[-1]]
    batch = pad_batch(pair, 0, "teacher_ids")
    ids, attn = batch["p_ids"], batch["p_attn"]
    lens = [int(x) for x in attn.sum(axis=1).tolist()]
    # the reference: the stock model, each row alone and unpadded, on the Metal scan
    # (the deployment path: the eval, the chat app)
    tower.eval()
    for shim in tower.model.layers:
        shim.layer.train(False)
    refs = [tower(ids[b:b + 1, :L]).astype(mx.float32)[0, -1] for b, L in enumerate(lens)]
    mx.eval(refs)
    out = {}
    for label, window in (("kernel", None), ("ops_above_48", 48)):
        if window is not None:
            tower.set_grad_window(window)       # the training path: ops scan from 48 up
        s = MlxStream(tower, ids, attn)
        s.run(0, n)
        lg = s.finish().astype(mx.float32)
        rep["model"]["hidden_dtype"] = str(s.hidden.dtype)
        rows = []
        for b, L in enumerate(lens):
            got = lg[b, L - 1]
            mx.eval(got)
            rows.append({"len": L, "rel": rel(got, refs[b]),
                         "argmax_same": int(mx.argmax(got).item()) == int(mx.argmax(refs[b]).item())})
        out[label] = rows
    # the kernel path is the deployment path: it must match the stock model closely;
    # the ops scan is the training path: the 9B's ops-vs-kernel gap was 7.2e-3 in bf16
    ok_k = all(r["argmax_same"] and r["rel"] <= 2e-3 for r in out["kernel"])
    ok_o = all(r["argmax_same"] and r["rel"] <= 2e-2 for r in out["ops_above_48"])
    rep["stream"] = {**out, "ok": ok_k and ok_o}
    ok &= ok_k and ok_o

    # -- grad and memory: a delta added mid-stack, backward through the top of the stack
    item = max(load(f"data/constitution_train_{TAG}.json")[:40],
               key=lambda r: len(r["prompt_ids"]) + len(r["teacher_ids"]))
    seq = mx.array([(item["prompt_ids"] + item["teacher_ids"])[:64]])
    one = mx.ones(seq.shape, dtype=mx.int32)

    def run_loss(delta, lo, cp):
        s = MlxStream(tower, seq, one)
        s.checkpoint_from = cp
        s.run(0, lo)
        s.hidden = s.hidden + delta.astype(s.hidden.dtype)
        s.run(lo, n)
        lg = s.finish().astype(mx.float32)
        return mx.mean(mx.logsumexp(lg[0, :-1], axis=-1)
                       - mx.take_along_axis(lg[0, :-1], seq[0, 1:, None], axis=-1)[:, 0])

    mx.random.seed(0)
    d = mx.random.normal((1, seq.shape[1], tower.args.hidden_size)) * 0.01
    res = {}
    for lo in (56, 48):
        tower.set_grad_window(lo)
        for cp in (None, lo):
            f = mx.value_and_grad(lambda x: run_loss(x, lo, cp))
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            t1 = time.time()
            v, g = f(d)
            mx.eval(v, g)
            res[(lo, cp)] = (float(v), g, (mx.get_peak_memory() - base) / 2**30, time.time() - t1)
    grad_rel = rel(res[(56, 56)][1], res[(56, None)][1])
    grad_rel48 = rel(res[(48, 48)][1], res[(48, None)][1])
    rep["grad"] = {"window_from_56_rel": grad_rel, "window_from_48_rel": grad_rel48,
                   "loss": res[(56, None)][0], "ok": grad_rel <= 1e-5 and grad_rel48 <= 1e-5}
    ok &= rep["grad"]["ok"]
    taped, recomp = res[(48, None)][2], res[(48, 48)][2]
    t56, r56 = res[(56, None)][2], res[(56, 56)][2]
    rep["memory"] = {"seq": int(seq.shape[1]), "window_layers": [n - 56, n - 48],
                     "taped_peak_gb": [round(t56, 2), round(taped, 2)],
                     "recomputed_peak_gb": [round(r56, 2), round(recomp, 2)],
                     "taped_sec": round(res[(48, None)][3], 1), "recomputed_sec": round(res[(48, 48)][3], 1),
                     # lower than taped AND nearly flat as the window doubles: mx.checkpoint
                     # failed the second on the CPU and both on the GPU
                     "recompute_saves": recomp < 0.6 * taped and (recomp - r56) < 0.25 * max(taped - t56, 1e-9)}

    # -- packed: the input gradient through one ternary projection, as training takes
    # it (float32 activations, 64 rows), against the dense dequantized weight
    rt = str(Path(PACK) / "runtime")
    if rt not in sys.path:
        sys.path.insert(0, rt)
    from runtime import fwht
    proj = tower.model.layers[48].layer.mlp.down_proj
    Wd = mx.dequantize(proj.weight, proj.scales, proj.biases, group_size=128, bits=2).astype(mx.float32)
    width, out_w = Wd.shape[1], Wd.shape[0]
    x = mx.random.normal((1, 64, width)) * 0.1
    w = mx.random.normal((1, 64, out_w))

    def f(x_):
        return mx.sum(proj(x_).astype(mx.float32) * w)

    def f_dense(x_):
        h = fwht(x_, proj.block, proj.signs) if proj.block else x_
        return mx.sum((h @ Wd.T) * w)

    g, g_ref = mx.grad(f)(x), mx.grad(f_dense)(x)
    mx.eval(g, g_ref)
    r = rel(g, g_ref)
    rep["packed"] = {"module": "layers.48.mlp.down_proj", "shape": [out_w, width],
                     "input_dtype": str(x.dtype), "gpu_vs_dense_rel": r, "ok": r <= 1e-2}
    try:                                           # the CPU's kernels, where they exist
        with mx.stream(mx.cpu):
            g_cpu = mx.grad(f)(x)
            mx.eval(g_cpu)
        rep["packed"]["gpu_vs_cpu_rel"] = rel(g, g_cpu)
    except Exception as e:
        rep["packed"]["gpu_vs_cpu"] = f"skipped: {type(e).__name__}: {e}"[:300]
    ok &= rep["packed"]["ok"]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="data checks only")
    ap.add_argument("--out", default=None,
                    help="default results/bonsai/precheck.json, or precheck_data.json with "
                         "--no-model (the chain accepts only a full run)")
    a = ap.parse_args()
    out = a.out or str(ROOT / "results/bonsai" / ("precheck_data.json" if a.no_model else "precheck.json"))
    rep = {"created": time.strftime("%F %T"), "model_checked": not a.no_model, "data": {}}
    ok = False
    try:
        ok = check_data(rep)
        if not a.no_model:
            ok = check_model(rep) and ok
    except Exception:
        rep["error"] = traceback.format_exc()[-3000:]
        ok = False
    finally:
        rep["ok"] = ok
        Path(out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))
    print(f"PRECHECK {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
