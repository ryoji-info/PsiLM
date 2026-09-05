"""MLX bridges for the 2D Fisher-KPP task (Stage 2d) -- hybrid stack.

Language side of the coupling only; the physics (DPOT-Tiny) lives in torch
behind psilm/mlx/physics2d.py. The design that made the 8B / 12B 1D results
work is kept verbatim:

  * every numeric quantity of the prompt (a, cx, cy, w, x0, y0) is read by
    DETERMINISTIC span pooling over the quantity's token span, computed by
    QA2DBuilder.spans (prefix-anchored sublist match, widened by one token);
  * each pooled vector goes through its own Linear-GELU-Linear head into
    N_BINS = 100 bins on the 0.01 grid (bins = arange(100)/100, the buffer
    psilm/stage2d/bridges2d.py defines). Positions (cx, cy, x0, y0) live on
    the periodic unit interval and are estimated by the circular mean of the
    bin distribution; a and w are not periodic and use the linear
    expectation over the same bins (a in [0.3, 0.9] uses bins 30..90, w in
    [0.04, 0.10] bins 4..10 -- every bin that occurs in training is covered);
  * the physics->language channel carries ONLY the looked-up value at the
    predicted query point: ValueTokensMLX(u) -> GatedCrossAttentionMLX with
    inj_cap. No field tokens, no learned lookup.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from ..stage2d.qa2d import QUANTITIES
from .bridges import ForwardBridgeMLX, GatedCrossAttentionMLX, ValueTokensMLX

N_BINS = 100
PERIODIC = {"cx", "cy", "x0", "y0"}


class ForwardBridge2DMLX(ForwardBridgeMLX):
    """language -> physics: six span-pooled 100-bin classifiers.

    Subclasses ForwardBridgeMLX for its readout normalization only
    (readout_norm 'rms' | 'dim', dim_mu / dim_sigma, calibrate_readout,
    _normalize); the 1D bridge's learned pooling heads are not instantiated.
    """

    QUANTITIES = QUANTITIES

    def __init__(self, d_model: int, d_hidden: int = 256, readout_norm: str = "rms"):
        nn.Module.__init__(self)
        self.readout_norm = readout_norm
        self.dim_mu = mx.zeros((d_model,))
        self.dim_sigma = mx.ones((d_model,))
        self.freeze(keys=["dim_mu", "dim_sigma"], recurse=False)
        self.h1 = [nn.Linear(d_model, d_hidden) for _ in QUANTITIES]
        self.h2 = [nn.Linear(d_hidden, N_BINS) for _ in QUANTITIES]
        self._bins = mx.arange(N_BINS, dtype=mx.float32) / N_BINS

    @staticmethod
    def span_pool(h, spans):
        """h (B, L, d) normalized hidden; spans (B, Q, 2) int -> (B, Q, d) mean
        over each token span (the normalized span mask is the pooling weight)."""
        L = h.shape[1]
        pos = mx.arange(L)[None, None, :]
        m = ((pos >= spans[:, :, 0:1]) & (pos < spans[:, :, 1:2])).astype(mx.float32)   # (B, Q, L)
        w = m / mx.maximum(m.sum(axis=-1, keepdims=True), mx.array(1.0))
        return w @ h                                                                    # (B, Q, d)

    def estimate(self, logits, name):
        p = mx.softmax(logits, axis=-1)
        if name in PERIODIC:
            ang = 2 * math.pi * self._bins
            v = mx.arctan2(p @ mx.sin(ang), p @ mx.cos(ang)) / (2 * math.pi)
            return v - mx.floor(v)                       # wrap into [0, 1)
        return p @ self._bins

    def __call__(self, hidden, prompt_mask, spans):
        """hidden (B, L, d) at the readout layer; prompt_mask unused (kept for the
        PsiLMMLX call convention); spans (B, 6, 2). Returns a dict with one
        (B,) estimate per quantity, 'logits' (list of six (B, 100)) and
        'params' (B, 4) = [a, cx, cy, w] for the physics side."""
        h = self._normalize(hidden)
        pooled = self.span_pool(h, spans)
        out, logits = {}, []
        for i, name in enumerate(QUANTITIES):
            lg = self.h2[i](nn.gelu(self.h1[i](pooled[:, i])))
            logits.append(lg)
            out[name] = self.estimate(lg, name)
        out["logits"] = logits
        out["params"] = mx.stack([out["a"], out["cx"], out["cy"], out["w"]], axis=-1)
        return out


class PsiBridges2DMLX(nn.Module):
    """ForwardBridge2DMLX + ValueTokensMLX + GatedCrossAttentionMLX. The channel
    is 'value' only (the 2D port never had a field channel)."""

    def __init__(self, d_model: int, gate_bias: float = -2.0, inj_cap=None,
                 channel: str = "value", readout_norm: str = "rms"):
        super().__init__()
        if channel != "value":
            raise ValueError("PsiBridges2DMLX supports channel='value' only")
        self.channel = channel
        self.fwd = ForwardBridge2DMLX(d_model, readout_norm=readout_norm)
        self.val = ValueTokensMLX(d_model)
        self.inject = GatedCrossAttentionMLX(d_model, gate_bias=gate_bias, inj_cap=inj_cap)

    def reinit_channel(self, gate_bias: float = -2.0, inj_cap=None, channel: str = None):
        """Fresh physics->language channel (value tokens + gated injection),
        keeping the trained readouts."""
        d_model = self.inject.to_q.weight.shape[1]
        self.val = ValueTokensMLX(d_model)
        self.inject = GatedCrossAttentionMLX(d_model, gate_bias=gate_bias, inj_cap=inj_cap)
