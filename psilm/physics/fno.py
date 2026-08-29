"""A compact 1D Fourier Neural Operator: u(., 0) -> u(., T) in one shot.

Self-contained (no external operator library): spectral convolution layers
with learned complex weights on the low Fourier modes, pointwise channel
mixing, GELU. Predicts the full field at t = T directly — the
"whole rollout at once" operator style rather than autoregressive stepping.
Trained once on solver data, then FROZEN: it is the physics hemisphere.
"""

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / channels
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):                       # x: (B, C, N)
        x_hat = torch.fft.rfft(x)
        out = torch.zeros_like(x_hat)
        out[:, :, : self.modes] = torch.einsum(
            "bim,iom->bom", x_hat[:, :, : self.modes], self.weight
        )
        return torch.fft.irfft(out, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, width: int = 32, modes: int = 16, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(2, width)          # (u0, x) -> width
        self.spectral = nn.ModuleList(SpectralConv1d(width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.act = nn.GELU()
        self.proj = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.width = width

    def features(self, u0):                      # u0: (B, N) -> (B, N, width)
        n = u0.shape[-1]
        x = torch.linspace(0, 1, n + 1, device=u0.device)[:-1]
        h = torch.stack([u0, x.expand_as(u0)], dim=-1)   # (B, N, 2)
        h = self.lift(h).permute(0, 2, 1)                # (B, W, N)
        for spec, pw in zip(self.spectral, self.pointwise):
            h = self.act(spec(h) + pw(h))
        return h.permute(0, 2, 1)                        # (B, N, W)

    def forward(self, u0):                       # (B, N) -> (B, N)
        return self.proj(self.features(u0)).squeeze(-1)
