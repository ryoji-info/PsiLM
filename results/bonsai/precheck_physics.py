#!/usr/bin/env python3
"""What must hold before the physics bridge is trained on Ternary Bonsai.

Run by results/bonsai/physics_bonsai.sh before any smoke run. Each check is one
that would cost days if it failed only after training:

  qa       every training and held-out question builds under Bonsai's tokenizer
           (its x0 span is found) to the ids, labels and span the 9B's tokenizer
           gives: the bridge sees the 9B bridge's prompts token for token
  noharm   the negatives are the 9B run's 1,194 prompts with Bonsai's own
           continuations (results/bonsai/precheck.json checked that; it must be
           ok and not older than the file), in the layout the trainer reads
  fno      results/stage2/fno.pt, which the trainer and the held-out evaluation
           load, and results/hf_export/physics/fno_burgers_singlemode.safetensors,
           which the chat app loads, give the same field
  infra    results/bonsai/precheck.json: the staged stream, the recomputed
           backward and the packed projections were checked at the write layer
  stream   (GPU) the training forward on a right-padded batch of two questions
           with their answers gives the stock model's last-position logits, on the
           Metal scan and with the differentiable ops scan above the write layer

  python results/bonsai/precheck_physics.py              # -> results/bonsai/precheck_physics.json
  python results/bonsai/precheck_physics.py --no-model   # the CPU checks only -> ..._cpu.json

Exit 0 when every check passes, 1 otherwise. Prints no prompt and no text.
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
REF_TOK = "/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx"
READ, WRITE = 26, 48
NEG = "data/noharm_bonsai27b_all.json"


def load(p):
    return json.loads((ROOT / p).read_text())


def check_qa(rep):
    from transformers import AutoTokenizer
    from psilm.stage2.qa import QABuilder
    mine, ref = QABuilder(AutoTokenizer.from_pretrained(PACK)), QABuilder(AutoTokenizer.from_pretrained(REF_TOK))
    out, ok = {}, True
    for split, n in (("train", 16384), ("val", 512)):
        items = load(f"data/stage2_qa_{split}.json")
        same = spans = 0
        lens, bad = [], 0
        for it in items:
            try:
                a, b = mine.build(it), ref.build(it)
            except ValueError:
                bad += 1
                continue
            lens.append(len(a["p_ids"]))
            same += a["p_ids"] == b["p_ids"] and a["p_labels"] == b["p_labels"]
            spans += tuple(a["x0_span"]) == tuple(b["x0_span"])
        good = len(items) == n and bad == 0 and same == n and spans == n
        out[split] = {"n": len(items), "x0_span_not_found": bad, "same_ids_and_labels_as_9b": same,
                      "same_x0_span_as_9b": spans, "tokens_min": min(lens), "tokens_max": max(lens), "ok": good}
        ok &= good
    rep["qa"] = {**out, "ok": ok}
    return ok


def check_noharm(rep):
    neg, pc = load(NEG), load("results/bonsai/precheck.json")
    fresh = (ROOT / "results/bonsai/precheck.json").stat().st_mtime >= (ROOT / NEG).stat().st_mtime
    layout = all(isinstance(r.get("prompt_ids"), list) and isinstance(r.get("target_ids"), list)
                 and r["prompt_ids"] and r["target_ids"] for r in neg)
    longest = max(len(r["prompt_ids"]) + len(r["target_ids"]) for r in neg)
    ok = len(neg) == 1194 and layout and bool(pc.get("noharm", {}).get("ok")) and fresh
    rep["noharm"] = {"n": len(neg), "layout": layout, "longest_tokens": longest,
                     "checked_by_precheck_json": bool(pc.get("noharm", {}).get("ok")),
                     "precheck_json_not_older": fresh, "ok": ok}
    infra = all(bool(pc.get(k, {}).get("ok")) for k in ("stream", "grad", "packed")) and bool(pc.get("ok"))
    at = "window_from_%d_rel" % WRITE in pc.get("grad", {}) and "ops_above_%d" % WRITE in pc.get("stream", {})
    rep["infra"] = {"precheck_json_ok": infra, "covers_write_layer": at, "write": WRITE,
                    "recomputed_peak_gb": pc.get("memory", {}).get("recomputed_peak_gb"), "ok": infra and at}
    return ok and infra and at


def check_fno(rep):
    import mlx.core as mx
    import numpy as np
    from psilm.mlx.fno import convert_from_torch, load_fno_safetensors
    mx.set_default_device(mx.cpu)
    a = convert_from_torch(str(ROOT / "results/stage2/fno.pt"))
    b = load_fno_safetensors(str(ROOT / "results/hf_export/physics/fno_burgers_singlemode.safetensors"))
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 128, endpoint=False, dtype=np.float32)
    u0 = mx.array(np.stack([amp * np.sin(2 * np.pi * x + ph) for amp, ph in
                            zip(rng.uniform(0.5, 1.5, 8), rng.uniform(0, 2 * np.pi, 8))]).astype(np.float32))
    ya, yb = a(u0), b(u0)
    mx.eval(ya, yb)
    d = float(mx.max(mx.abs(ya - yb)))
    rep["fno"] = {"max_abs_difference": d, "fields": 8, "ok": d <= 1e-5}
    mx.set_default_device(mx.gpu)
    return d <= 1e-5


def rel(a, b):
    import mlx.core as mx
    return float(mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(b)), 1e-9))


def check_stream(rep):
    import mlx.core as mx
    import random
    from transformers import AutoTokenizer
    from psilm.mlx.gemma_loader import load_backbone_any
    from psilm.mlx.staged import MlxStream
    from psilm.stage2.qa import QABuilder, make_batch
    t0 = time.time()
    tower, _, _ = load_backbone_any(PACK)
    n = len(tower.model.layers)
    builder = QABuilder(AutoTokenizer.from_pretrained(PACK))
    items = random.Random(3).sample(load("data/stage2_qa_train.json"), 2)
    tb = make_batch(builder, items, "cpu")
    ids, attn = mx.array(tb["p_ids"].numpy()), mx.array(tb["p_attn"].numpy().astype("int32"))
    lens = [int(v) for v in attn.sum(axis=1).tolist()]
    tower.eval()
    for shim in tower.model.layers:
        shim.layer.train(False)
    refs = [tower(ids[b:b + 1, :L]).astype(mx.float32)[0, -1] for b, L in enumerate(lens)]
    mx.eval(refs)
    out = {}
    for label, window in (("kernel", None), (f"ops_above_{WRITE}", WRITE)):
        if window is not None:
            tower.set_grad_window(window)
        s = MlxStream(tower, ids, attn)
        s.run(0, n)
        lg = s.finish().astype(mx.float32)
        rows = []
        for b, L in enumerate(lens):
            got = lg[b, L - 1]
            mx.eval(got)
            rows.append({"len": L, "rel": rel(got, refs[b]),
                         "argmax_same": int(mx.argmax(got).item()) == int(mx.argmax(refs[b]).item())})
        out[label] = rows
    ok = (all(r["argmax_same"] and r["rel"] <= 2e-3 for r in out["kernel"])
          and all(r["argmax_same"] and r["rel"] <= 2e-2 for r in out[f"ops_above_{WRITE}"]))
    rep["stream"] = {**out, "n_layers": n, "load_and_check_sec": round(time.time() - t0, 1),
                     "active_gb": round(mx.get_active_memory() / 2**30, 2), "ok": ok and n == 64}
    return ok and n == 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="the CPU checks only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "read": READ, "write": WRITE,
           "model_checked": not a.no_model}
    ok = True
    for name, fn in (("qa", check_qa), ("noharm", check_noharm), ("fno", check_fno)) + \
            (() if a.no_model else (("stream", check_stream),)):
        try:
            ok &= bool(fn(rep))
        except Exception as e:                        # noqa: BLE001  (a check that cannot run has failed)
            traceback.print_exc()
            rep[name] = {"ok": False, "error": type(e).__name__}
            ok = False
    rep["ok"] = ok
    out = a.out or ("results/bonsai/precheck_physics_cpu.json" if a.no_model else "results/bonsai/precheck_physics.json")
    (ROOT / out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
