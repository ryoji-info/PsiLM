#!/usr/bin/env python
"""How much signal is in the constitution teacher, and of what kind.

The trainer's loss is cross-entropy against the teacher's greedy continuation,
so "the teacher differs from base on 94% of prompts" (the stats file's
teacher_differs_rate) is a token-exact flag and says nothing about how early or
how substantively it differs. Two numbers decide how to read a trained bridge:

  divergence depth -- if the teacher only parts from base late in the sequence,
      the CE target is mostly "continue the same answer" and a bridge can score
      well by doing nothing. Measured as the first index where the two token
      streams differ.

  refusal delta -- the guard-rail reports refusal rate, so the headroom on that
      metric is the base->teacher shift, not 0->1. At 9B the backbone already
      refuses most of harmless-base unaided, which caps it. The shift is also
      two-sided: the constitution both adds and removes refusals, and a bridge
      that only adds them is not reproducing the teacher.

Usage:  python results/constitution/teacher_signal.py <tag> [--model PATH]
"""
import argparse, json, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from bench_common import is_refusal  # noqa: E402

SPLITS = ("train", "val", "test", "helpful_test")


def first_divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None            # one is a strict prefix of the other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--model", default="/Users/rxiii/Documents/huggingface/qwen3.5-9b-mlx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    out = {}
    hdr = ("%-13s %6s %8s %8s %6s %6s %6s %8s %8s %8s %8s"
           % ("split", "n", "differs", "prefix", "fd50", "fd90", "fd<8",
              "ref_base", "ref_tchr", "t_only", "b_only"))
    print(hdr); print("-" * len(hdr))
    for s in SPLITS:
        p = Path("data/constitution_%s_%s.json" % (s, a.tag))
        if not p.is_file():
            print("%-13s (not built yet)" % s); continue
        items = json.loads(p.read_text())
        n = len(items)
        fd = [first_divergence(it["base_ids"], it["teacher_ids"]) for it in items]
        pref = sum(1 for x in fd if x is None)
        fdv = sorted(x for x in fd if x is not None)
        B = [is_refusal(tok.decode(it["base_ids"])) for it in items]
        T = [is_refusal(tok.decode(it["teacher_ids"])) for it in items]
        q = lambda p_: fdv[min(len(fdv) - 1, p_ * len(fdv) // 100)] if fdv else -1
        rec = dict(n=n,
                   differs=sum(1 for it in items if it["differs"]) / n,
                   prefix=pref / n,
                   fd_mean=st.mean(fdv) if fdv else None,
                   fd_p50=q(50), fd_p90=q(90), fd_max=max(fdv) if fdv else None,
                   fd_lt8=sum(1 for x in fdv if x < 8) / n,
                   fd_at0=sum(1 for x in fdv if x == 0) / n,
                   refusal_base=sum(B) / n, refusal_teacher=sum(T) / n,
                   teacher_only=sum(1 for b, t in zip(B, T) if t and not b) / n,
                   base_only=sum(1 for b, t in zip(B, T) if b and not t) / n,
                   mean_len_base=st.mean(len(it["base_ids"]) for it in items),
                   mean_len_teacher=st.mean(len(it["teacher_ids"]) for it in items))
        out[s] = rec
        print("%-13s %6d %8.3f %8.3f %6d %6d %6.3f %8.3f %8.3f %+8.3f %+8.3f"
              % (s, n, rec["differs"], rec["prefix"], rec["fd_p50"], rec["fd_p90"],
                 rec["fd_lt8"], rec["refusal_base"], rec["refusal_teacher"],
                 rec["teacher_only"], -rec["base_only"]))
    dest = a.out or "results/constitution/teacher_signal_%s.json" % a.tag
    Path(dest).write_text(json.dumps(out, indent=2))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
