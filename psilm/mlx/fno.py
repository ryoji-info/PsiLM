"""MLX port of the FNO physics hemisphere, with exact weight conversion.

Complex spectral weights are stored as (real, imag) pairs and the complex
multiply is done in real arithmetic, so only rfft/irfft touch complex
dtypes. `convert_from_torch` maps a trained psilm.physics.fno.FNO1d
checkpoint so the physics model is numerically the same network across the
PyTorch and MLX stacks.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SpectralConv1dMLX(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / channels
        self.wr = scale * mx.random.normal((channels, channels, modes))
        self.wi = scale * mx.random.normal((channels, channels, modes))

    def __call__(self, x):                    # x: (B, N, C) float32
        n = x.shape[1]
        x_hat = mx.fft.rfft(x, axis=1)        # (B, Nf, C) complex64
        xm = x_hat[:, : self.modes, :]
        xr, xi = mx.real(xm), mx.imag(xm)
        outr = mx.einsum("bmi,iom->bmo", xr, self.wr) - mx.einsum("bmi,iom->bmo", xi, self.wi)
        outi = mx.einsum("bmi,iom->bmo", xr, self.wi) + mx.einsum("bmi,iom->bmo", xi, self.wr)
        nf = x_hat.shape[1]
        pad = nf - self.modes
        outr = mx.pad(outr, [(0, 0), (0, pad), (0, 0)])
        outi = mx.pad(outi, [(0, 0), (0, pad), (0, 0)])
        out_hat = outr.astype(mx.complex64) + outi.astype(mx.complex64) * mx.array(1j, dtype=mx.complex64)
        return mx.fft.irfft(out_hat, n=n, axis=1)


class FNO1dMLX(nn.Module):
    def __init__(self, width: int = 32, modes: int = 16, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.spectral = [SpectralConv1dMLX(width, modes) for _ in range(layers)]
        self.pointwise = [nn.Linear(width, width) for _ in range(layers)]
        self.proj1 = nn.Linear(width, 64)
        self.proj2 = nn.Linear(64, 1)
        self.width = width

    def features(self, u0):                   # (B, N) -> (B, N, W)
        n = u0.shape[-1]
        x = mx.arange(n, dtype=mx.float32) / n
        h = mx.stack([u0, mx.broadcast_to(x[None, :], u0.shape)], axis=-1)
        h = self.lift(h)
        for spec, pw in zip(self.spectral, self.pointwise):
            h = nn.gelu(spec(h) + pw(h))
        return h

    def proj(self, feats):
        return self.proj2(nn.gelu(self.proj1(feats)))

    def __call__(self, u0):
        return self.proj(self.features(u0)).squeeze(-1)


def convert_from_torch(pt_path: str) -> FNO1dMLX:
    import torch
    sd = torch.load(pt_path, map_location="cpu")
    fno = FNO1dMLX()
    fno.lift.weight = mx.array(sd["lift.weight"].numpy())
    fno.lift.bias = mx.array(sd["lift.bias"].numpy())
    for i in range(4):
        w = sd[f"spectral.{i}.weight"].numpy()          # (C, C, M) complex64
        fno.spectral[i].wr = mx.array(np.ascontiguousarray(w.real))
        fno.spectral[i].wi = mx.array(np.ascontiguousarray(w.imag))
        cw = sd[f"pointwise.{i}.weight"].numpy()        # (Cout, Cin, 1)
        fno.pointwise[i].weight = mx.array(cw[:, :, 0])
        fno.pointwise[i].bias = mx.array(sd[f"pointwise.{i}.bias"].numpy())
    fno.proj1.weight = mx.array(sd["proj.0.weight"].numpy())
    fno.proj1.bias = mx.array(sd["proj.0.bias"].numpy())
    fno.proj2.weight = mx.array(sd["proj.2.weight"].numpy())
    fno.proj2.bias = mx.array(sd["proj.2.bias"].numpy())
    return fno
