"""Trainable bridges for the 2D DPOT configuration. All fp32 internally."""

import math

import torch
import torch.nn as nn

from ..stage2.bridges import GatedCrossAttention  # reused as-is

N_BINS = 100
GRID = 128
PATCH_GRID = 16


class ForwardBridge2D(nn.Module):
    """language -> physics: six pooled readouts from layer-10 hidden states.

    (a, w) are regressions; (cx, cy, x0, y0) are 100-bin classifications with
    softmax-expectation — every bin occurs in training (the Stage-2b support-
    coverage law), so these generalize.
    """

    QUANTITIES = ("a", "w", "cx", "cy", "x0", "y0")

    def __init__(self, d_model: int = 896, d_hidden: int = 256):
        super().__init__()
        self.key = nn.Linear(d_model, d_model)
        self.queries = nn.Parameter(torch.randn(6, d_model) / math.sqrt(d_model))
        self.reg_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
            for _ in range(2)
        )
        self.cls_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, N_BINS))
            for _ in range(4)
        )
        self.register_buffer("bins", torch.arange(N_BINS, dtype=torch.float32) / N_BINS)

    def forward(self, hidden, prompt_mask):
        h = hidden.float()
        keys = self.key(h)
        pooled = []
        for q in self.queries:
            scores = (keys * q).sum(-1) / math.sqrt(h.shape[-1])
            scores = scores.masked_fill(~prompt_mask, float("-inf"))
            pooled.append((torch.softmax(scores, dim=-1).unsqueeze(-1) * h).sum(dim=1))
        a = self.reg_heads[0](pooled[0]).squeeze(-1)
        w = self.reg_heads[1](pooled[1]).squeeze(-1).clamp(min=0.02)
        logits = [head(p) for head, p in zip(self.cls_heads, pooled[2:])]
        vals = [torch.softmax(lg, dim=-1) @ self.bins for lg in logits]
        cx, cy, x0, y0 = vals
        return {"a": a, "w": w, "cx": cx, "cy": cy, "x0": x0, "y0": y0,
                "cls_logits": logits}


def build_ic_2d(a, cx, cy, w, n: int = GRID):
    """Differentiable periodic Gaussian bump (B,) params -> (B, n, n)."""
    x = torch.linspace(0, 1, n + 1, device=a.device)[:-1]
    dx = torch.remainder(x[None, :] - cx[:, None] + 0.5, 1.0) - 0.5   # (B, n)
    dy = torch.remainder(x[None, :] - cy[:, None] + 0.5, 1.0) - 0.5
    d2 = dx[:, :, None] ** 2 + dy[:, None, :] ** 2                     # (B, n, n)
    return a[:, None, None] * torch.exp(-d2 / (2 * (w[:, None, None] ** 2)))


class ReverseBridge2D(nn.Module):
    """physics -> language: K tokens over DPOT patch features + a 2D
    position-lookup token sampling the predicted field at (x0, y0)."""

    def __init__(self, feat_dim: int = 512, d_model: int = 896, k_tokens: int = 16,
                 d_attn: int = 128, n_pos: int = 4):
        super().__init__()
        self.n_pos = n_pos
        self.feat = nn.Linear(feat_dim + 4 * n_pos, d_attn)
        self.queries = nn.Parameter(torch.randn(k_tokens, d_attn) / math.sqrt(d_attn))
        self.out = nn.Linear(d_attn, d_model)
        self.log_kappa = nn.Parameter(torch.tensor(5.0))
        # lookup token: pooled field value + locally pooled coarse features
        self.lookup_out = nn.Linear(1 + d_attn, d_model)
        self.u_head = nn.Linear(1 + d_attn, 1)

    @staticmethod
    def _pos_feats(n, n_pos, device):
        x = torch.linspace(0, 1, n + 1, device=device)[:-1]
        ks = torch.arange(1, n_pos + 1, device=device, dtype=torch.float32)
        return torch.cat([torch.sin(2 * math.pi * ks * x[:, None]),
                          torch.cos(2 * math.pi * ks * x[:, None])], dim=-1)  # (n, 2P)

    def _kernel(self, n, center, kappa):
        x = torch.linspace(0, 1, n + 1, device=center.device)[:-1]
        return torch.softmax(kappa * torch.cos(2 * math.pi * (x[None, :] - center[:, None])), dim=-1)

    def forward(self, patch_feats, field, x0, y0):
        """patch_feats (B, 256, 512); field (B, 128, 128); x0, y0 (B,)."""
        B = patch_feats.shape[0]
        pf = self._pos_feats(PATCH_GRID, self.n_pos, patch_feats.device)     # (16, 2P)
        pos = torch.cat([
            pf[:, None, :].expand(-1, PATCH_GRID, -1),
            pf[None, :, :].expand(PATCH_GRID, -1, -1),
        ], dim=-1).reshape(PATCH_GRID * PATCH_GRID, -1)                       # (256, 4P)
        kv = self.feat(torch.cat([patch_feats.float(),
                                  pos[None].expand(B, -1, -1)], dim=-1))     # (B, 256, d)
        attn = torch.softmax(self.queries @ kv.transpose(1, 2) / math.sqrt(kv.shape[-1]), dim=-1)
        tokens = self.out(attn @ kv)                                          # (B, K, d_model)

        kappa = torch.exp(self.log_kappa)
        wx = self._kernel(GRID, x0, kappa)                                    # (B, 128)
        wy = self._kernel(GRID, y0, kappa)
        u_pool = torch.einsum("bi,bij,bj->b", wx, field.float(), wy)          # (B,)
        wxp = self._kernel(PATCH_GRID, x0, kappa / 4)
        wyp = self._kernel(PATCH_GRID, y0, kappa / 4)
        kv_grid = kv.reshape(B, PATCH_GRID, PATCH_GRID, -1)
        feat_pool = torch.einsum("bi,bijd,bj->bd", wxp, kv_grid, wyp)         # (B, d)
        packed = torch.cat([u_pool.unsqueeze(-1), feat_pool], dim=-1)
        lookup = self.lookup_out(packed).unsqueeze(1)
        u_hat = self.u_head(packed).squeeze(-1)
        return torch.cat([lookup, tokens], dim=1), u_hat


class PsiBridges2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd = ForwardBridge2D()
        self.rev = ReverseBridge2D()
        self.inject = GatedCrossAttention()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
