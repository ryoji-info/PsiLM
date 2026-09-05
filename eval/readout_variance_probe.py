"""Readout-signal diagnostic: how much across-item variance survives in the
pooled x0-span vector at the readout layer, under the per-position RMS
normalization alone vs. calibrated per-dimension standardization.

This is the measurement behind --readout-norm dim (Gemma 4's constant
massive-activation dimensions squash the digit signal under RMS norm).

Usage: python eval/readout_variance_probe.py --model mlx-community/gemma-4-12B-it-4bit \
           --hf-tokenizer mlx-community/gemma-4-12B-it-4bit --layer-frac 0.4167 --tag gemma12b
"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from psilm.mlx.staged import MlxStream  # noqa: E402
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from psilm.mlx.bridges import ForwardBridgeMLX, _rms  # noqa: E402
from psilm.stage2.qa import QABuilder, make_batch  # noqa: E402
from eval.mlx_stage2_train import to_mlx_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--layer", type=int, default=None, help="readout layer (default: round(n*10/24))")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-items", type=int, default=8)
    ap.add_argument("--calib-n", type=int, default=32)
    args = ap.parse_args()
    tower, stock, tok = load_backbone_any(args.model)
    hf = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    n = len(tower.model.layers)
    layer = args.layer if args.layer is not None else round(n * 10 / 24)
    items = json.loads(Path("data/stage2_qa_train.json").read_text())
    b = QABuilder(hf)
    # per-dimension energy at the readout layer (first n_items prompts)
    H, spans = [], []
    for it in items[: args.n_items]:
        p = b.prompt_ids(it); s = b.x0_span(p, it)
        st = MlxStream(tower, mx.array([p])); st.run(0, layer); H.append(st.hidden[0].astype(mx.float32)); spans.append(s)
    allh = mx.concatenate(H, axis=0)
    dim_rms = mx.sqrt((allh ** 2).mean(axis=0)); top = mx.argsort(-dim_rms)[:5]
    top5_share = float((dim_rms[top] ** 2).sum() / (dim_rms ** 2).sum())
    # calibration (as the trainer does it)
    cb = to_mlx_batch(make_batch(b, random.Random(7).sample(items, args.calib_n), "cpu"))
    cs = MlxStream(tower, cb["p_ids"], cb["p_attn"]); cs.run(0, layer)
    fb = ForwardBridgeMLX(tower.args.hidden_size, readout_norm="dim"); fb.calibrate_readout(cs.hidden, cb["prompt_mask"])
    out = {"model": args.model, "layer": layer, "n_layers": n, "n_items": args.n_items, "calib_n": args.calib_n,
           "top5_dim_energy_share": round(top5_share, 4), "dim_rms_max": round(float(dim_rms.max()), 2),
           "dim_rms_median": round(float(mx.sort(dim_rms)[dim_rms.shape[0] // 2]), 4)}
    for name, f in (("rms", lambda h: _rms(h)), ("dim_standardized_rms", lambda h: fb._normalize(h))):
        pooled = mx.stack([f(h)[s[0]:s[1]].mean(axis=0) for h, s in zip(H, spans)]); v = pooled.var(axis=0)
        out[f"pooled_span_var_total_{name}"] = round(float(v.sum()), 3)
        out[f"pooled_span_var_maxdim_{name}"] = round(float(v.max()), 5)
    Path("results/readout_variance").mkdir(parents=True, exist_ok=True)
    Path(f"results/readout_variance/{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
