"""DPOT-Tiny as the Stage-2d physics hemisphere.

Wraps the pretrained 7.5M-parameter DPOT-Tiny (hzk17/DPOT, Apache-2.0;
AFNO architecture, pretrained across 12 PDE datasets) for our scalar
Fisher-KPP task: the IC is replicated across the model's 10-timestep input
history in channel 0 (other channels zero), and the model is briefly
fine-tuned to predict u(., T_FINAL) in one shot. A forward hook on the last
AFNO block exposes the (B, 512, 16, 16) latent patch features the reverse
bridge attends over.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
from dpot_model import DPOTNet  # noqa: E402

CKPT = Path("results/stage2d/model_Ti.pth")


class DPOTPhysics(torch.nn.Module):
    def __init__(self, ckpt_path=CKPT, device="cpu"):
        super().__init__()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.net = DPOTNet(
            img_size=128, patch_size=8, mixing_type="afno",
            in_channels=4, out_channels=4, in_timesteps=10, out_timesteps=1,
            n_blocks=4, embed_dim=512, out_layer_dim=32, depth=4, modes=32,
            mlp_ratio=1, act="gelu", normalize=False,
        )
        self.net.load_state_dict(sd["model"], strict=True)
        self._feats = None
        self.net.blocks[-1].register_forward_hook(self._hook)
        self.to(device)

    def _hook(self, module, inp, out):
        self._feats = out

    @staticmethod
    def prepare_input(u0):
        """(B, 128, 128) scalar IC -> (B, 128, 128, 10, 4) DPOT input."""
        x = torch.zeros(u0.shape[0], 128, 128, 10, 4,
                        device=u0.device, dtype=u0.dtype)
        x[..., 0] = u0.unsqueeze(-1)
        return x

    def features_and_field(self, u0):
        """Returns (patch_feats (B, 256, 512), field (B, 128, 128))."""
        out, _ = self.net(self.prepare_input(u0))
        field = out[..., 0, 0]                     # (B, 128, 128)
        feats = self._feats                        # (B, 512, 16, 16)
        feats = feats.flatten(2).transpose(1, 2)   # (B, 256, 512)
        return feats, field

    def forward(self, u0):
        return self.features_and_field(u0)[1]
