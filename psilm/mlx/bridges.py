"""MLX port of the PsiLM bridges (single-mode forward, lookup reverse)."""

import math

import mlx.core as mx
import mlx.nn as nn

N_BINS = 100


def _rms(h):
    """Parameter-free RMS normalization: makes bridge inputs O(1) regardless
    of backbone width/scale. Large backbones (8B, 4096-dim) otherwise
    saturate the pooling softmax and gates, starving classifiers of
    gradient (observed empirically at Qwen3-8B)."""
    return h * mx.rsqrt((h * h).mean(axis=-1, keepdims=True) + 1e-6)


class ForwardBridgeMLX(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 256, n_params: int = 3,
                 readout_norm: str = "rms"):
        super().__init__()
        # readout_norm "dim": standardize each hidden dimension with calibration
        # statistics (mean/std over positions of a calibration batch at the
        # readout layer) BEFORE the per-position RMS norm. Backbones whose
        # massive-activation dimensions are constant across prompts (Gemma 4:
        # top-5 dims hold 95% of the energy) otherwise leave the pooled span
        # vector with ~26x less across-item variance than Qwen's, and the x0
        # classifier crawls. The buffers are frozen parameters (saved with the
        # bridges); calibrate_readout() fills them.
        self.readout_norm = readout_norm
        self.dim_mu = mx.zeros((d_model,))
        self.dim_sigma = mx.ones((d_model,))
        self.freeze(keys=["dim_mu", "dim_sigma"], recurse=False)
        self.query = mx.random.normal((d_model,)) / math.sqrt(d_model)
        self.key = nn.Linear(d_model, d_model)
        self.mlp1 = nn.Linear(d_model, d_hidden)
        self.mlp2 = nn.Linear(d_hidden, n_params)
        self.x0_query = mx.random.normal((d_model,)) / math.sqrt(d_model)
        self.x0_key = nn.Linear(d_model, d_model)
        self.x0_h1 = nn.Linear(d_model, d_hidden)
        self.x0_h2 = nn.Linear(d_hidden, N_BINS)
        self._bins = mx.arange(N_BINS, dtype=mx.float32) / N_BINS

    def _pool(self, h, mask, query, key):
        scores = (key(h) * query).sum(-1) / math.sqrt(h.shape[-1])
        scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
        w = mx.softmax(scores, axis=-1)
        return (w[..., None] * h).sum(axis=1), w

    def calibrate_readout(self, hidden, valid):
        """hidden (B, L, d) at the readout layer, valid (B, L) bool: set per-dim
        mean/std over the valid positions."""
        h = hidden.astype(mx.float32)
        w = valid.astype(mx.float32)[..., None]
        n = w.sum()
        mu = (h * w).sum(axis=(0, 1)) / n
        var = (((h - mu) ** 2) * w).sum(axis=(0, 1)) / n
        self.dim_mu = mu
        self.dim_sigma = mx.sqrt(var + 1e-6)
        self.freeze(keys=["dim_mu", "dim_sigma"], recurse=False)
        mx.eval(self.dim_mu, self.dim_sigma)

    def _normalize(self, hidden):
        h = hidden.astype(mx.float32)
        if self.readout_norm == "dim":
            h = (h - self.dim_mu) / self.dim_sigma
        return _rms(h)

    def __call__(self, hidden, prompt_mask, x0_span=None):
        h = self._normalize(hidden)
        pooled_p, _ = self._pool(h, prompt_mask, self.query, self.key)
        params = self.mlp2(nn.gelu(self.mlp1(pooled_p)))
        if x0_span is not None:
            # deterministic pointer: mean over the x0 token span the QA
            # builder computes exactly. At 8B the learned attention pointer
            # never leaves uniform (CE = ln 100 at any budget) while span
            # pooling reaches sub-bin precision — so we stop learning where
            # to look. w_x0 is the normalized span mask (attention loss = 0).
            L = h.shape[1]
            pos = mx.arange(L)[None, :]
            m = ((pos >= x0_span[:, 0:1]) & (pos < x0_span[:, 1:2])).astype(mx.float32)
            w_x0 = m / m.sum(axis=-1, keepdims=True)
            pooled_x = (w_x0[..., None] * h).sum(axis=1)
        else:
            pooled_x, w_x0 = self._pool(h, prompt_mask, self.x0_query, self.x0_key)
        x0_logits = self.x0_h2(nn.gelu(self.x0_h1(pooled_x)))
        # circular mean of the bin distribution: x0 lives on the periodic
        # domain, so a linear expectation collapses toward 0.5 while the
        # classifier is broad and mis-wraps at 0/1; the circular mean does
        # neither and agrees with the linear one once the distribution peaks.
        p = mx.softmax(x0_logits, axis=-1)
        ang = 2 * math.pi * self._bins
        x0_hat = mx.arctan2(p @ mx.sin(ang), p @ mx.cos(ang)) / (2 * math.pi)
        x0_hat = x0_hat - mx.floor(x0_hat)          # wrap into [0, 1)
        return params, x0_hat, x0_logits, w_x0


def build_ic_mlx(params, n: int = 128):
    a = params[:, 0:1]
    sc = params[:, 1:3]
    sc = sc / (mx.sqrt((sc ** 2).sum(axis=-1, keepdims=True)) + 1e-6)
    x = mx.arange(n, dtype=mx.float32) / n
    tp = 2 * math.pi * x
    return a * (mx.sin(tp)[None, :] * sc[:, 1:2] + mx.cos(tp)[None, :] * sc[:, 0:1])


class ReverseBridgeMLX(nn.Module):
    def __init__(self, d_model: int, fno_width: int = 32, k_tokens: int = 16,
                 d_attn: int = 128, n_pos: int = 8):
        super().__init__()
        self.n_pos = n_pos
        self.feat = nn.Linear(fno_width + 1 + 2 * n_pos, d_attn)
        self.queries = mx.random.normal((k_tokens, d_attn)) / math.sqrt(d_attn)
        self.out = nn.Linear(d_attn, d_model)
        self.log_kappa = mx.array(5.0)
        self.lookup_out = nn.Linear(d_attn, d_model)
        self.u_head = nn.Linear(d_attn, 1)

    def __call__(self, feats, u_field, x0_hat):
        B, N, _ = feats.shape
        x = mx.arange(N, dtype=mx.float32) / N
        ks = mx.arange(1, self.n_pos + 1, dtype=mx.float32)
        pos = mx.concatenate([mx.sin(2 * math.pi * ks[None, :] * x[:, None]),
                              mx.cos(2 * math.pi * ks[None, :] * x[:, None])], axis=-1)
        f = mx.concatenate([feats, u_field[..., None],
                            mx.broadcast_to(pos[None], (B,) + pos.shape)], axis=-1)
        kv = self.feat(f)
        attn = mx.softmax(self.queries @ kv.transpose(0, 2, 1) / math.sqrt(kv.shape[-1]), axis=-1)
        tokens = self.out(attn @ kv)
        kappa = mx.exp(self.log_kappa)
        w = mx.softmax(kappa * mx.cos(2 * math.pi * (x[None, :] - x0_hat[:, None])), axis=-1)
        pooled = (w[..., None] * kv).sum(axis=1)
        lookup = self.lookup_out(pooled)[:, None, :]
        u_hat = self.u_head(pooled).squeeze(-1)
        return mx.concatenate([lookup, tokens], axis=1), u_hat


class ValueTokensMLX(nn.Module):
    """Physics->language channel in the form the 8B copy probe proved readable:
    the looked-up value u(x0) -> Fourier features -> K soft tokens. The field
    tokens of ReverseBridgeMLX let an 8B collapse to per-trajectory constants
    (it reads the amplitude and ignores the one x0-dependent token); this
    channel can only carry what depends on x0."""

    def __init__(self, d_model: int, k_tokens: int = 8, d_hidden: int = 256, n_freq: int = 16):
        super().__init__()
        self.k = k_tokens
        self.d_model = d_model
        self.n_freq = n_freq
        self.enc1 = nn.Linear(1 + 2 * n_freq, d_hidden)
        self.enc2 = nn.Linear(d_hidden, k_tokens * (d_model // 8))
        self.proj = nn.Linear(d_model // 8, d_model)

    def __call__(self, u):                       # u: (B,)
        ks = mx.arange(1, self.n_freq + 1, dtype=mx.float32)
        feats = mx.concatenate([u[:, None],
                                mx.sin(ks[None, :] * u[:, None]),
                                mx.cos(ks[None, :] * u[:, None])], axis=-1)
        h = nn.gelu(self.enc1(feats))
        h = self.enc2(h).reshape(u.shape[0], self.k, self.d_model // 8)
        return self.proj(h)


class GatedCrossAttentionMLX(nn.Module):
    def __init__(self, d_model: int, d_attn: int = 256, g_hidden: int = 256,
                 gate_bias: float = -2.0, inj_cap=None):
        super().__init__()
        # inj_cap: hard cap on the injection's RMS relative to the receiver's
        # stream RMS (before the gate). 8B v6 showed Adam inflating to_out until
        # the injection ran at 30-50% of the residual stream at every position,
        # which garbled the frozen model even though the value it carried was
        # right; the successful 8B copy probe and v5 both operated near 5%.
        self.inj_cap = inj_cap
        self.to_q = nn.Linear(d_model, d_attn)
        self.to_k = nn.Linear(d_model, d_attn)
        self.to_v = nn.Linear(d_model, d_attn)
        self.to_out = nn.Linear(d_attn, d_model)
        self.to_out.weight = 1e-3 * mx.random.normal(self.to_out.weight.shape)
        self.to_out.bias = mx.zeros_like(self.to_out.bias)
        self.g1 = nn.Linear(d_model, g_hidden)
        self.g2 = nn.Linear(g_hidden, 1)
        self.g2.weight = mx.zeros_like(self.g2.weight)
        self.g2.bias = mx.full(self.g2.bias.shape, gate_bias)

    def __call__(self, hidden, phys_tokens, return_ratio=False):
        dtype = hidden.dtype
        h_raw = hidden.astype(mx.float32)
        h = _rms(h_raw)
        q, k, v = self.to_q(h), self.to_k(phys_tokens), self.to_v(phys_tokens)
        attn = mx.softmax(q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1]), axis=-1)
        inj = self.to_out(attn @ v)
        if self.inj_cap is not None:
            r = mx.sqrt((inj * inj).mean(axis=-1, keepdims=True) + 1e-12)
            inj = inj * mx.minimum(mx.array(1.0), self.inj_cap / r)
        sigma = mx.sigmoid(self.g2(nn.relu(self.g1(h))))
        # scale the injection to the receiver's local stream magnitude, so
        # the channel is scale-free across backbone widths
        scale = mx.sqrt((h_raw * h_raw).mean(axis=-1, keepdims=True) + 1e-6)
        delta = sigma * inj * scale
        out = (h_raw + delta).astype(dtype)
        if return_ratio:
            # effective channel strength per position: RMS(injection)/RMS(stream).
            # sigma alone is not the channel (to_out can absorb any fixed sigma).
            ratio = mx.sqrt((delta * delta).mean(axis=-1) + 1e-12) / scale[..., 0]
            return out, sigma, ratio
        return out, sigma


class PsiBridgesMLX(nn.Module):
    def __init__(self, d_model: int, n_params: int = 3, gate_bias: float = -2.0,
                 inj_cap=None, channel: str = "field", readout_norm: str = "rms"):
        super().__init__()
        self.channel = channel          # "field": lookup + K field tokens; "value": ValueTokensMLX(u_hat)
        self.fwd = ForwardBridgeMLX(d_model, n_params=n_params, readout_norm=readout_norm)
        self.rev = ReverseBridgeMLX(d_model)
        if channel == "value":
            self.val = ValueTokensMLX(d_model)
        self.inject = GatedCrossAttentionMLX(d_model, gate_bias=gate_bias, inj_cap=inj_cap)

    def reinit_channel(self, gate_bias: float = -2.0, inj_cap=None, channel: str = None):
        """Fresh physics->language channel (token heads and the gated injection)
        while keeping the trained readouts and lookup: fwd, rev.feat, rev.u_head,
        rev.log_kappa. Optionally switch the channel form."""
        d_model = self.inject.to_q.weight.shape[1]
        if channel is not None:
            self.channel = channel
        fresh = ReverseBridgeMLX(d_model)
        self.rev.out = fresh.out
        self.rev.lookup_out = fresh.lookup_out
        self.rev.queries = fresh.queries
        if self.channel == "value":
            self.val = ValueTokensMLX(d_model)
        self.inject = GatedCrossAttentionMLX(d_model, gate_bias=gate_bias, inj_cap=inj_cap)
