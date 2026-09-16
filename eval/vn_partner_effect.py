#!/usr/bin/env python3
"""Does the partner's knowledge of the constitution matter, and does that depend on width?

Pairs each recorded 9B width with its plain-partner twin -- same backbone, same
coupling depths, same cap, same written coordinates, same teacher targets, and a
partner that has never read the document. The partner is the same base model
either way (Qwen2.5-0.5B, d_model 896), so no bridge shape changes and the only
difference is the fine-tune.

At 0.5B this control was negative and width-dependent: the plain partner had
LOWER held-out CE by 0.0040 at nine written dimensions, with a bootstrap interval
excluding zero, and by 0.0096 at full width with an interval spanning it. A
negative number here means the fine-tuned partner is not earning its place.

  python3 eval/vn_partner_effect.py

Writes results/constitution/partner_effect_qwen35.json.
"""
import argparse, json
from pathlib import Path

PAIRS = [("vn", "vnplain", 41), ("vn5", "vn5plain", 205), ("all", "allplain", 4096)]


def read(tag):
    out = {}
    f = Path(f"results/stage2c_qwen35_{tag}/train_log.jsonl")
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("eval") and r.get("step") == 1000:
                out["val_ce"] = r["eval"]["psilm"]["ce"]
                out["val_base"] = r["eval"]["base"]["ce"]
    for split in ("test", "helpful"):
        g = Path(f"results/stage2c_qwen35_{tag}/eval_{split}.json")
        if g.exists():
            d = json.loads(g.read_text())
            a = d.get("arms", d)
            out[f"{split}_ce"] = a["psilm"]["ce"]
            out[f"{split}_base"] = a["base"]["ce"]
            out[f"{split}_refusal"] = a["psilm"]["refusal"]
    m = Path(f"results/stage2c_qwen35_{tag}/bridges.npz.meta")
    if m.exists():
        out["const_model"] = json.loads(m.read_text())["const_model"].split("/")[-1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/constitution/partner_effect_qwen35.json")
    a = ap.parse_args()
    rows, recs = [], {}
    for fine, plain, dims in PAIRS:
        f, p = read(fine), read(plain)
        if not f or not p:
            continue
        rec = {"dims": dims, "fine_partner": f.get("const_model"),
               "plain_partner": p.get("const_model")}
        for split in ("val", "test", "helpful"):
            kf, kp = f.get(f"{split}_ce"), p.get(f"{split}_ce")
            if kf is None or kp is None:
                continue
            # negative: the PLAIN partner has lower CE, i.e. the fine-tune costs
            rec[f"{split}_delta"] = round(kp - kf, 4)
            rec[f"{split}_fine"] = kf
            rec[f"{split}_plain"] = kp
        recs[f"{fine} vs {plain}"] = rec
        rows.append((fine, plain, dims, rec))
    hdr = (f"{'width':>6s} {'fine CE':>9s} {'plain CE':>9s} {'delta':>8s}   "
           f"{'test fine':>9s} {'test plain':>10s} {'delta':>8s}")
    print(hdr); print("-" * len(hdr))
    for fine, plain, dims, r in rows:
        print(f"{dims:6d} {r.get('val_fine', float('nan')):9.4f} "
              f"{r.get('val_plain', float('nan')):9.4f} {r.get('val_delta', float('nan')):+8.4f}   "
              f"{r.get('test_fine', float('nan')):9.4f} {r.get('test_plain', float('nan')):10.4f} "
              f"{r.get('test_delta', float('nan')):+8.4f}")
    print("\ndelta = plain minus fine-tuned; NEGATIVE means the fine-tuned partner "
          "is not earning its place")
    if not rows:
        print("(no pairs complete yet)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"pairs": recs}, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
