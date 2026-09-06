"""Stage 2b, span readout: read every number from its own token span.

Why this exists. The pooled readout of psilm/mlx/multimode.py reaches 100% on
the training family and then fails to transfer: on Gemma 4 12B it scores 25.0%
on the held-out mode combination and 52.1% on amplitude extrapolation, while
the FNO is exact on both families (MAE 0.0008) and the oracle arm is 100%. The
whole loss is in the readout, and the two failures have one cause:

  * a plain MLP regressing 6 values from ONE pooled vector has no way to say
    "there is a second term here" when every training prompt had one term;
  * a regression head clamps to the amplitudes it saw -- on val_amp the
    implied amplitude is below the true one for 71% of items, median ratio
    0.68, which lands inside the mode-1 training range (0.3-0.7).

So each quantity gets read from its own deterministic token span (QA2Builder.
spans, the pointer that already made x0 work), and the amplitude and phase
heads are SHARED across mode slots. Sharing is what buys extrapolation:
mode-2 amplitudes are drawn from 0.5-1.0 in training, so the shared head has
seen the digits "0.8x" and "0.9x" and reads them the same way in the mode-1
slot at test time. An absent mode is an empty span (lo == hi): it pools to
zero and its amplitude is forced to zero, so single-mode and two-mode prompts
are the same computation.

Nothing else changes: the value channel, the gate, the injection cap and the
training phases are those of the 8B/12B single-mode result.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .bridges import ForwardBridgeMLX, PsiBridgesMLX
from .model import PsiLMMLX, cross_entropy_masked
from .staged import MlxStream
from .multimode import N_MODES, N_PARAMS, build_ic_multi_mlx

A_BINS = 101                     # amplitudes on the 0.01 grid, 0.00 .. 1.00
P_BINS = 100                     # phases on [0, 2 pi), periodic
X0_BINS = 100                    # x0 on [0, 1), periodic

# Slot order matches psilm.stage2.qa2.QA2Builder.SLOTS: (a, phi) per mode, x0 last.
AMP_SLOTS = [2 * m for m in range(N_MODES)]
PHI_SLOTS = [2 * m + 1 for m in range(N_MODES)]
X0_SLOT = 2 * N_MODES


class _Stash:
    """Plain object for tensors that must NOT join the module tree: mlx.nn.Module
    files every dict/array attribute into its parameter dict, and a stashed
    activation there would desynchronize the optimizer state across chunks."""


class ForwardBridgeMultiSpanMLX(ForwardBridgeMLX):
    """language -> physics: span-pooled bins, one shared head per quantity kind.

    Subclasses ForwardBridgeMLX for its readout normalization only (dim_mu /
    dim_sigma, calibrate_readout, _normalize); the pooled bridge's learned
    pooling heads are not instantiated.
    """

    def __init__(self, d_model: int, d_hidden: int = 256, readout_norm: str = "rms"):
        nn.Module.__init__(self)
        self.readout_norm = readout_norm
        self.dim_mu = mx.zeros((d_model,))
        self.dim_sigma = mx.ones((d_model,))
        self.freeze(keys=["dim_mu", "dim_sigma"], recurse=False)
        self.a_h1 = nn.Linear(d_model, d_hidden)
        self.a_h2 = nn.Linear(d_hidden, A_BINS)
        self.phi_h1 = nn.Linear(d_model, d_hidden)
        self.phi_h2 = nn.Linear(d_hidden, P_BINS)
        self.x0_h1 = nn.Linear(d_model, d_hidden)
        self.x0_h2 = nn.Linear(d_hidden, X0_BINS)
        self._a_bins = mx.arange(A_BINS, dtype=mx.float32) / (A_BINS - 1)
        self._p_ang = 2 * math.pi * mx.arange(P_BINS, dtype=mx.float32) / P_BINS
        self._x_ang = 2 * math.pi * mx.arange(X0_BINS, dtype=mx.float32) / X0_BINS
        self._stash = _Stash()

    @staticmethod
    def span_masks(spans, L):
        """spans (B, Q, 2) -> (B, Q, L) float mask of each span's tokens."""
        pos = mx.arange(L)[None, None, :]
        return ((pos >= spans[:, :, 0:1]) & (pos < spans[:, :, 1:2])).astype(mx.float32)

    def __call__(self, hidden, prompt_mask, spans=None):
        """hidden (B, L, d) at the readout layer; spans (B, Q, 2) with Q =
        2*N_MODES+1 in QA2Builder.SLOTS order. An absent mode's span is (0, 0).
        Returns (params, x0_hat, x0_logits, w_x0) -- the PsiLMMLX contract --
        and stashes the per-quantity logits for the readout loss."""
        if spans is None:
            raise ValueError("the span readout needs QA2Builder.spans; none was passed")
        h = self._normalize(hidden)
        m = self.span_masks(spans, h.shape[1])                       # (B, Q, L)
        w = m / mx.maximum(m.sum(axis=-1, keepdims=True), mx.array(1.0))
        pooled = w @ h                                               # (B, Q, d)
        present = (spans[:, :, 1] > spans[:, :, 0]).astype(mx.float32)

        a_in = pooled[:, AMP_SLOTS]                                  # (B, M, d)
        a_logits = self.a_h2(nn.gelu(self.a_h1(a_in)))               # (B, M, A_BINS)
        a_hat = (mx.softmax(a_logits, axis=-1) @ self._a_bins) * present[:, AMP_SLOTS]

        p_logits = self.phi_h2(nn.gelu(self.phi_h1(pooled[:, PHI_SLOTS])))
        pp = mx.softmax(p_logits, axis=-1)
        sin_phi = pp @ mx.sin(self._p_ang)
        cos_phi = pp @ mx.cos(self._p_ang)

        x0_logits = self.x0_h2(nn.gelu(self.x0_h1(pooled[:, X0_SLOT])))
        px = mx.softmax(x0_logits, axis=-1)
        x0_hat = mx.arctan2(px @ mx.sin(self._x_ang), px @ mx.cos(self._x_ang)) / (2 * math.pi)
        x0_hat = x0_hat - mx.floor(x0_hat)

        params = mx.concatenate(
            [mx.stack([a_hat[:, i], sin_phi[:, i], cos_phi[:, i]], axis=-1)
             for i in range(N_MODES)], axis=-1)                      # (B, 3*N_MODES)
        self._stash.logits = {"a": a_logits, "phi": p_logits, "present": present}
        return params, x0_hat, x0_logits, m[:, X0_SLOT]


class PsiBridgesMultiSpanMLX(PsiBridgesMLX):
    """PsiBridgesMLX with the span readout in the forward seat; reverse bridge,
    value tokens and gated injection are the released design, unchanged."""

    def __init__(self, d_model: int, gate_bias: float = -2.0, inj_cap=None,
                 channel: str = "value", readout_norm: str = "rms"):
        super().__init__(d_model=d_model, n_params=N_PARAMS, gate_bias=gate_bias,
                         inj_cap=inj_cap, channel=channel, readout_norm=readout_norm)
        self.fwd = ForwardBridgeMultiSpanMLX(d_model, readout_norm=readout_norm)


def make_bridges_multi_span(d_model: int, **kw) -> PsiBridgesMultiSpanMLX:
    return PsiBridgesMultiSpanMLX(d_model=d_model, **kw)


class PsiLMMLXMultiSpan(PsiLMMLX):
    """Multi-mode PsiLM whose readout loss is cross-entropy over the amplitude
    and phase bins of each present mode, instead of MSE on the pooled vector."""

    def __init__(self, model, tokenizer, fno, bridges, **kw):
        if not isinstance(bridges.fwd, ForwardBridgeMultiSpanMLX):
            raise ValueError("PsiLMMLXMultiSpan needs bridges from make_bridges_multi_span")
        super().__init__(model, tokenizer, fno, bridges,
                         ic_fn=build_ic_multi_mlx, param_loss_fn=self._param_loss, **kw)

    def loss_fn(self, batch, lam_param=1.0, lam_x0=None, lam_u=2.0, lam_attn=0.5):
        """The base loss without the attention term: the span pointer is
        deterministic, so supervising it against its own mask is vacuous."""
        if lam_x0 is None:
            lam_x0 = self.lam_x0
        if self.readout_only or batch.get("noharm"):
            return super().loss_fn(batch, lam_param, lam_x0, lam_u, lam_attn)
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        params_hat, x0_hat, x0_logits, u_hat, sigma, _ = self._couple(
            s, batch["prompt_mask"], batch["x0_span"])
        logits = s.finish()
        loss_ans = cross_entropy_masked(logits, batch["p_labels"],
                                        self.digit_ids, self.digit_weight)
        loss_param = self.param_loss_fn(params_hat, batch)
        loss_x0 = nn.losses.cross_entropy(x0_logits, batch["x0_bins"], reduction="mean")
        loss_u = ((u_hat - batch["u_true"]) ** 2).mean()
        loss = loss_ans + lam_param * loss_param + lam_x0 * loss_x0 + lam_u * loss_u
        resp = (batch["p_labels"] != -100).astype(mx.float32)
        n_resp = resp.sum() + 1e-6
        gate_ans = (sigma[..., 0] * resp).sum() / n_resp
        ratio_ans = (self._last_ratio * resp).sum() / n_resp
        x0_exact = (x0_logits.argmax(-1) == batch["x0_bins"]).astype(mx.float32).mean()
        d = mx.abs(x0_hat - batch["x0"])
        return loss, (loss_ans, loss_param, loss_x0, loss_u,
                      mx.minimum(d, 1 - d).mean(), sigma.mean(), mx.array(0.0),
                      gate_ans, ratio_ans, x0_exact)

    def generate(self, builder, item, max_new=24):
        """As PsiLMMLX.generate, with the full span set instead of the x0 span."""
        prompt = builder.prompt_ids(item)
        ids = list(prompt)
        eos = self.tok.eos_token_id
        spans = mx.array([[list(s) for s in builder.spans(prompt, item)[0]]], dtype=mx.int32)
        for _ in range(max_new):
            t = mx.array([ids])
            pmask = mx.concatenate([mx.ones((1, len(prompt)), dtype=mx.bool_),
                                    mx.zeros((1, len(ids) - len(prompt)), dtype=mx.bool_)], axis=1)
            s = MlxStream(self.model, t)
            self._couple(s, pmask, spans)
            nxt = int(s.finish()[:, -1].argmax(-1).item())
            ids.append(nxt)
            if nxt == eos:
                break
        return self.tok.decode(ids[len(prompt):])

    def _param_loss(self, params_hat, batch):
        """CE over the amplitude and phase bins of every present mode, plus the
        absent modes' amplitude pinned to bin 0 (the 'no second term' case the
        pooled readout could not express)."""
        lg = self.phi.fwd._stash.logits
        pres = batch["mode_present"]                                  # (B, M)
        a_ce = nn.losses.cross_entropy(
            lg["a"].reshape(-1, A_BINS), batch["amp_bins"].reshape(-1), reduction="none"
        ).reshape(pres.shape)
        p_ce = nn.losses.cross_entropy(
            lg["phi"].reshape(-1, P_BINS), batch["phi_bins"].reshape(-1), reduction="none"
        ).reshape(pres.shape)
        n = mx.maximum(pres.sum(), mx.array(1.0))
        return a_ce.mean() + (p_ce * pres).sum() / n


def batch_extras(builder, items):
    """Span-readout inputs for a batch of QA items: the token spans of every
    number, the amplitude/phase bin targets, and which modes are present.

    Prompts are re-tokenized here rather than threaded through the torch batch
    builder, which keeps psilm/stage2/qa2.py's pooled path untouched; the cost
    is tokenization, against a 12B forward pass."""
    spans, amp_bins, phi_bins, present = [], [], [], []
    for it in items:
        sp, _ = builder.spans(builder.prompt_ids(it), it)
        spans.append([list(s) for s in sp])
        a = [0.0] * N_MODES
        ph = [0.0] * N_MODES
        pr = [0.0] * N_MODES
        for m, av, pv in it["modes"]:
            a[m - 1], ph[m - 1], pr[m - 1] = av, pv, 1.0
        amp_bins.append([min(A_BINS - 1, round(v * (A_BINS - 1))) for v in a])
        phi_bins.append([round(v / (2 * math.pi) * P_BINS) % P_BINS for v in ph])
        present.append(pr)
    return {
        "spans": mx.array(spans, dtype=mx.int32),
        "amp_bins": mx.array(amp_bins, dtype=mx.int32),
        "phi_bins": mx.array(phi_bins, dtype=mx.int32),
        "mode_present": mx.array(present, dtype=mx.float32),
    }


def empty_spans(batch_size):
    """All-absent spans for the no-harm arm: a non-physics prompt has no
    numbers to read, so every span pools nothing and every amplitude is zero."""
    return mx.zeros((batch_size, 2 * N_MODES + 1, 2), dtype=mx.int32)
