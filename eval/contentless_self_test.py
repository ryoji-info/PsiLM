#!/usr/bin/env python3
"""Assertions for the contentless control, on synthetic weights, on the CPU.

The control's whole value is that it differs from the psilm arm in exactly one
respect -- the payload's content -- and matches it in every other. Two ways that
could be silently wrong, both of which this project has hit before in other
guises: the arm could degrade into the psilm arm (writing the real payload after
all) or into the zeroed arm (writing nothing), and either would produce a
plausible table that means nothing. So the properties are asserted rather than
assumed.

Runs on mx.cpu deliberately: a second MLX process on the GPU can take the Metal
watchdog down with whatever training or guard-rail job is holding it.

  python3 eval/contentless_self_test.py
"""
import sys
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)                      # never contend for the GPU
mx.random.seed(0)                                  # the module's Linear layers draw from the global RNG
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from psilm.mlx.constitution import MaskedGatedCrossAttentionMLX    # noqa: E402


def rms_over(x, mask, n_write):
    return float(mx.sqrt((x * x * mask).sum(axis=-1) * (1.0 / n_write) + 1e-12)[0, 0].item())


def main() -> int:
    d_model, T, L = 64, 4, 6
    write = sorted(range(0, d_model, 3))           # 22 of 64 coordinates
    # to_k/to_v are Linear(d_model, d_attn), so the partner's tokens arrive already
    # mapped into the backbone's width by the reverse bridge.
    inj = MaskedGatedCrossAttentionMLX(d_model, write_idx=write, d_attn=16,
                                       g_hidden=8, inj_cap=0.2, gate_bias=0.0)
    # g2 ships zeroed so sigma is exactly 1/2 everywhere; give it a real gate so
    # the test would notice the payload leaking into it.
    inj.g2.weight = mx.random.normal(inj.g2.weight.shape, key=mx.random.key(3)) * 0.5
    h = mx.random.normal((1, L, d_model), key=mx.random.key(1))
    tok = mx.random.normal((1, T, d_model), key=mx.random.key(2))
    mask = inj.write_mask.astype(mx.float32)
    n_w = len(write)
    fails = []

    def check(n, cond, detail=""):
        print(f"  {n:2d}. {'PASS' if cond else 'FAIL'}  {detail}")
        if not cond:
            fails.append(n)

    inj.contentless = None
    h_p, sig_p, ratio_p = inj(h, tok, return_ratio=True)
    d_p = h_p - h
    inj.contentless = 11
    h_c, sig_c, ratio_c = inj(h, tok, return_ratio=True)
    d_c = h_c - h
    inj.contentless = None

    check(1, float(mx.abs(d_p).max().item()) > 0,
          "the psilm arm writes something at all")
    check(2, float(mx.abs(d_c).max().item()) > 0,
          "the contentless arm WRITES -- it is not the zeroed arm")
    check(3, float(mx.abs(d_c - d_p).max().item()) > 1e-6,
          "the contentless arm differs from the psilm arm -- it is not a duplicate")
    check(4, float(mx.abs(sig_c - sig_p).max().item()) < 1e-7,
          f"the gate is untouched by the payload (max |dsigma| "
          f"{float(mx.abs(sig_c - sig_p).max().item()):.2e})")
    rp, rc = rms_over(d_p, mask, n_w), rms_over(d_c, mask, n_w)
    # 1e-3 relative: float32 cannot resolve tighter at a write RMS of ~1e-3, and the
    # failure this guards against (a masked standard normal, RMS ~1) is six orders off.
    check(5, abs(rp - rc) / max(rp, 1e-12) < 1e-3,
          f"per-coordinate write RMS matches: psilm {rp:.6f} vs contentless {rc:.6f}")
    off_c = float(mx.abs(d_c * (1.0 - mask)).max().item())
    check(6, off_c == 0.0, f"the contentless write respects the mask (max off-mask {off_c})")

    # a direction, not noise: identical across positions up to the receiver scale
    col = [d_c[0, i] / (mx.linalg.norm(d_c[0, i]) + 1e-12) for i in range(L)]
    cos = min(float((col[0] * col[i]).sum().item()) for i in range(1, L))
    check(7, cos > 0.999, f"one fixed direction at every position (min cosine {cos:.5f})")

    inj.contentless = 11
    h_again, _, _ = inj(h, tok, return_ratio=True)
    inj.contentless = 12
    h_other, _, _ = inj(h, tok, return_ratio=True)
    inj.contentless = None
    check(8, float(mx.abs(h_again - h_c).max().item()) == 0.0,
          "the same seed reproduces the same write exactly")
    check(9, float(mx.abs(h_other - h_c).max().item()) > 1e-6,
          "a different seed gives a different direction")

    # The flag must not survive the call, or a later arm inherits it. That is
    # the COUPLER's job (its try/finally), so test it there -- including when the
    # write raises, which is the case a plain assertion after a clean call misses.
    from psilm.mlx.constitution import ConstitutionCoupler
    class _Bridges:                       # only .inject is touched by inject()
        pass
    br = _Bridges(); br.inject = inj
    coup = ConstitutionCoupler.__new__(ConstitutionCoupler); coup.phi = br
    coup.inject(h, tok, "psilm", contentless=99)
    cleared_ok = inj.contentless is None
    real_call = type(inj).__call__
    def _boom(self, *a, **k):
        raise RuntimeError("boom")
    type(inj).__call__ = _boom
    raised = False
    try:
        coup.inject(h, tok, "psilm", contentless=99)
    except RuntimeError:
        raised = True
    finally:
        type(inj).__call__ = real_call
    check(10, cleared_ok and raised and inj.contentless is None,
          "the coupler clears the flag after use, and after a raising write")

    print(f"\n{10 - len(fails)} of 10 assertions pass")
    if fails:
        print(f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
