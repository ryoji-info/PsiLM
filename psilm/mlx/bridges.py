"""MLX port of the PsiLM bridges (single-mode forward, lookup reverse)."""

import math

import mlx.core as mx
import mlx.nn as nn

N_BINS = 100


class ForwardBridgeMLX(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 256, n_params: int = 3):
        super().__init__()
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
        w = mx.softmax(scores, axis=-1)[..., None]
        return (w * h).sum(axis=1)

    def __call__(self, hidden, prompt_mask):
        h = hidden.astype(mx.float32)
        params = self.mlp2(nn.gelu(self.mlp1(self._pool(h, prompt_mask, self.query, self.key))))
        x0_logits = self.x0_h2(nn.gelu(self.x0_h1(self._pool(h, prompt_mask, self.x0_query, self.x0_key))))
        x0_hat = mx.softmax(x0_logits, axis=-1) @ self._bins
        return params, x0_hat, x0_logits


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


class GatedCrossAttentionMLX(nn.Module):
    def __init__(self, d_model: int, d_attn: int = 256, g_hidden: int = 256):
        super().__init__()
        self.to_q = nn.Linear(d_model, d_attn)
        self.to_k = nn.Linear(d_model, d_attn)
        self.to_v = nn.Linear(d_model, d_attn)
        self.to_out = nn.Linear(d_attn, d_model)
        self.to_out.weight = 1e-3 * mx.random.normal(self.to_out.weight.shape)
        self.to_out.bias = mx.zeros_like(self.to_out.bias)
        self.g1 = nn.Linear(d_model, g_hidden)
        self.g2 = nn.Linear(g_hidden, 1)
        self.g2.weight = mx.zeros_like(self.g2.weight)
        self.g2.bias = mx.full(self.g2.bias.shape, -2.0)

    def __call__(self, hidden, phys_tokens):
        dtype = hidden.dtype
        h = hidden.astype(mx.float32)
        q, k, v = self.to_q(h), self.to_k(phys_tokens), self.to_v(phys_tokens)
        attn = mx.softmax(q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1]), axis=-1)
        inj = self.to_out(attn @ v)
        sigma = mx.sigmoid(self.g2(nn.relu(self.g1(h))))
        return (h + sigma * inj).astype(dtype), sigma


class PsiBridgesMLX(nn.Module):
    def __init__(self, d_model: int, n_params: int = 3):
        super().__init__()
        self.fwd = ForwardBridgeMLX(d_model, n_params=n_params)
        self.rev = ReverseBridgeMLX(d_model)
        self.inject = GatedCrossAttentionMLX(d_model)
