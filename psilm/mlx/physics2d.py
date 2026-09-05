"""Torch physics side of the hybrid Stage-2d MLX stack.

The language hemisphere runs in MLX (4-bit backbones); the physics hemisphere
stays in torch: DPOT-Tiny (vendor/dpot_model.py) with the Stage-2d fine-tune
results/stage2d/dpot_ft.pt, frozen, on MPS (CPU fallback). No gradient crosses
the boundary -- the readout hands over a detached (B, 4) numpy array of IC
parameters and gets back the predicted field as an mx.array, exactly the
field psilm/stage2d/model2d.py computes (IC from the torch build_ic_2d,
replicated across DPOT's 10-step input history, out[..., 0, 0]).

`lookup` is the periodic bilinear interpolation of psilm/physics/fisher2d.py
as a pure MLX function of fractional coordinates (differentiable in x0, y0;
the model detaches them anyway). It is the only physics->language channel:
the value at the queried point.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from ..physics.dpot_wrapper import DPOTPhysics
from ..stage2d.bridges2d import GRID, build_ic_2d

DPOT_FT = Path("results/stage2d/dpot_ft.pt")


def pick_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TorchPhysics2D:
    """Frozen fine-tuned DPOT-Tiny behind a numpy/MLX boundary."""

    def __init__(self, ft_path=DPOT_FT, device="mps"):
        self.device = pick_device(device)
        self.phys = DPOTPhysics(device=self.device)
        sd = torch.load(ft_path, map_location=self.device)
        self.phys.net.load_state_dict(sd, strict=True)
        self.phys.eval()
        for p in self.phys.parameters():
            p.requires_grad_(False)
        self.dtype = next(self.phys.parameters()).dtype

    @torch.no_grad()
    def field_torch(self, params):
        """params (B, 4) float tensor [a, cx, cy, w] -> torch field (B, GRID, GRID)."""
        p = torch.as_tensor(np.asarray(params, dtype=np.float32), device=self.device)
        # the torch ForwardBridge2D clamps w >= 0.02 before build_ic_2d; the MLX
        # readout's bin expectation is unclamped, and w -> 0 makes the Gaussian
        # (and DPOT's output) non-finite -- e.g. on a no-harm prompt whose
        # readouts see non-physics text. Same floor here, at the boundary.
        ic = build_ic_2d(p[:, 0], p[:, 1], p[:, 2], p[:, 3].clamp(min=0.02))
        _, field = self.phys.features_and_field(ic.to(self.dtype))
        return field

    def __call__(self, params):
        """params: numpy / mx (B, 4) [a, cx, cy, w] (detached) -> mx.array (B, GRID, GRID) f32."""
        if isinstance(params, mx.array):
            params = np.array(params)
        field = self.field_torch(params)
        return mx.array(field.detach().float().cpu().numpy())


def lookup(field, x0, y0, n: int = GRID):
    """Periodic bilinear interpolation of field (B, n, n) [x index first, as
    build_ic_2d / fisher2d.bilinear_periodic] at fractional coordinates
    x0, y0 (B,) in [0, 1). Pure MLX; matches fisher2d.bilinear_periodic."""
    fx, fy = x0 * n, y0 * n
    ix, iy = mx.stop_gradient(mx.floor(fx)), mx.stop_gradient(mx.floor(fy))   # gather indices carry no VJP
    tx, ty = fx - ix, fy - iy
    i0 = ix.astype(mx.int32) % n
    j0 = iy.astype(mx.int32) % n
    i1, j1 = (i0 + 1) % n, (j0 + 1) % n
    b = mx.arange(field.shape[0])
    f00 = field[b, i0, j0]
    f10 = field[b, i1, j0]
    f01 = field[b, i0, j1]
    f11 = field[b, i1, j1]
    return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty)
            + f01 * (1 - tx) * ty + f11 * tx * ty)


def lookup_numpy_reference(field, x0, y0):
    """Batched fisher2d.bilinear_periodic on numpy inputs, for tests."""
    from ..physics.fisher2d import bilinear_periodic
    return np.array([bilinear_periodic(f, float(a), float(c)) for f, a, c in zip(field, x0, y0)],
                    dtype=np.float32)
