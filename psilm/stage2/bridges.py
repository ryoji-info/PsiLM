"""The two trainable bridges of PsiLM — the corpus callosum, Stage-2 form.

All bridge computation is fp32 regardless of backbone dtype (Stage-1 lesson);
outputs are cast back to the receiver's dtype.
"""

import math

import torch
import torch.nn as nn


class ForwardBridge(nn.Module):
    """language -> physics: read (a, sin phi, cos phi) from LLM hidden states.

    Attention-pools the prompt positions of the layer-10 residual stream with
    a learned query, then regresses the IC parameters. The IC field is built
    differentiably, so gradients from the answer loss flow back through the
    frozen FNO into this bridge.
    """

    def __init__(self, d_model: int = 896, d_hidden: int = 256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.key = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 3)
        )

    def forward(self, hidden, prompt_mask):
        h = hidden.float()
        scores = (self.key(h) * self.query).sum(-1) / math.sqrt(h.shape[-1])
        scores = scores.masked_fill(~prompt_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (w * h).sum(dim=1)
        return self.mlp(pooled)                      # (B, 3): a, sin, cos


def build_ic(params, n: int = 128):
    """Differentiable u0(x) = a sin(2 pi x + phi) from (a, sin phi, cos phi)."""
    a = params[:, 0:1]
    sc = params[:, 1:3]
    sc = sc / (sc.norm(dim=-1, keepdim=True) + 1e-6)  # project to unit circle
    x = torch.linspace(0, 1, n + 1, device=params.device)[:-1]
    two_pi_x = 2 * math.pi * x
    return a * (torch.sin(two_pi_x) * sc[:, 1:2] + torch.cos(two_pi_x) * sc[:, 0:1])


class ReverseBridge(nn.Module):
    """physics -> language: compress the FNO's latent field into K soft tokens.

    K learned queries cross-attend over the operator's per-gridpoint features
    (augmented with Fourier positional features of x), producing K vectors in
    the LLM's residual-stream dimensionality.
    """

    def __init__(self, fno_width: int = 32, d_model: int = 896, k_tokens: int = 16,
                 d_attn: int = 128, n_pos: int = 8):
        super().__init__()
        self.n_pos = n_pos
        d_in = fno_width + 2 * n_pos
        self.feat = nn.Linear(d_in, d_attn)
        self.queries = nn.Parameter(torch.randn(k_tokens, d_attn) / math.sqrt(d_attn))
        self.out = nn.Linear(d_attn, d_model)

    def forward(self, fno_feats):                    # (B, N, W) fp32
        B, N, _ = fno_feats.shape
        x = torch.linspace(0, 1, N + 1, device=fno_feats.device)[:-1]
        ks = torch.arange(1, self.n_pos + 1, device=fno_feats.device, dtype=torch.float32)
        pos = torch.cat([torch.sin(2 * math.pi * ks * x[:, None]),
                         torch.cos(2 * math.pi * ks * x[:, None])], dim=-1)
        feats = torch.cat([fno_feats, pos.expand(B, -1, -1)], dim=-1)
        kv = self.feat(feats)                        # (B, N, d_attn)
        attn = torch.softmax(self.queries @ kv.transpose(1, 2) / math.sqrt(kv.shape[-1]), dim=-1)
        tokens = attn @ kv                           # (B, K, d_attn)
        return self.out(tokens)                      # (B, K, d_model)


class GatedCrossAttention(nn.Module):
    """Inject physics tokens into the primary stream (layer 15), gated.

    h <- h + sigma(g(h)) * CrossAttn(h -> physics tokens): additive residual
    injection with a suppression gate that reads the receiver, per position.
    """

    def __init__(self, d_model: int = 896, d_attn: int = 256, g_hidden: int = 256):
        super().__init__()
        self.to_q = nn.Linear(d_model, d_attn)
        self.to_k = nn.Linear(d_model, d_attn)
        self.to_v = nn.Linear(d_model, d_attn)
        self.to_out = nn.Linear(d_attn, d_model)
        nn.init.normal_(self.to_out.weight, std=1e-3)
        nn.init.zeros_(self.to_out.bias)
        self.gate = nn.Sequential(
            nn.Linear(d_model, g_hidden), nn.ReLU(), nn.Linear(g_hidden, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, hidden, phys_tokens):
        dtype = hidden.dtype
        h = hidden.float()
        q, k, v = self.to_q(h), self.to_k(phys_tokens), self.to_v(phys_tokens)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]), dim=-1)
        inj = self.to_out(attn @ v)
        sigma = torch.sigmoid(self.gate(h))
        return (h + sigma * inj).to(dtype), sigma


class PsiBridges(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.fwd = ForwardBridge()
        self.rev = ReverseBridge(**kw)
        self.inject = GatedCrossAttention()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
