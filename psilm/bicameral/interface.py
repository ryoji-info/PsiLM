"""The trainable neural interface phi (the corpus callosum).

Each direction is a coupling operator on the receiver's residual-stream state:

    h_recv <- (1 - sigma) * h_recv + sigma * f(h_send)
    sigma   = Sigmoid(g(h_recv))            # gate reads the RECEIVER (pull)

PullStandard: f is an MLP translation network, g a smaller MLP.
ScalarIdentity: f is the identity (same hidden size), only g is trained.
Gates start near closed (bias init negative) so coupling strength begins
negligible and must be learned, as in the paper.
"""

import torch
import torch.nn as nn


class CouplingOperator(nn.Module):
    def __init__(self, d_model: int, kind: str = "pull_standard",
                 f_hidden: int = 1024, f_layers: int = 2,
                 g_hidden: int = 256, gate: str = "scalar"):
        super().__init__()
        self.kind = kind
        if kind == "pull_standard":
            blocks, d_in = [], d_model
            for _ in range(f_layers):
                blocks += [nn.Linear(d_in, f_hidden), nn.ReLU()]
                d_in = f_hidden
            blocks += [nn.Linear(d_in, d_model)]
            self.f = nn.Sequential(*blocks)
            # near-identity start: last layer small
            nn.init.normal_(self.f[-1].weight, std=1e-3)
            nn.init.zeros_(self.f[-1].bias)
        elif kind == "identity":
            self.f = nn.Identity()
        else:
            raise ValueError(kind)
        k = 1 if gate == "scalar" else d_model
        self.g = nn.Sequential(
            nn.Linear(d_model, g_hidden), nn.ReLU(), nn.Linear(g_hidden, k)
        )
        nn.init.zeros_(self.g[-1].weight)
        nn.init.constant_(self.g[-1].bias, -2.0)  # sigma ~= 0.12 at start

    def forward(self, h_send: torch.Tensor, h_recv: torch.Tensor):
        """Both (…, d_model). Returns (updated h_recv, sigma, translated).

        The interface computes in fp32 regardless of the backbone dtype and
        returns the receiver's dtype, so fp16 backbones compose cleanly.
        """
        dtype = h_recv.dtype
        hs, hr = h_send.float(), h_recv.float()
        sigma = torch.sigmoid(self.g(hr))
        translated = self.f(hs)
        out = (1 - sigma) * hr + sigma * translated
        return out.to(dtype), sigma, translated


class BicameralInterface(nn.Module):
    """Both directions: forward (p->a) and reverse (a->p)."""

    def __init__(self, d_model: int = 896, kind: str = "pull_standard",
                 f_hidden: int = 1024, f_layers: int = 2, gate: str = "scalar"):
        super().__init__()
        self.fwd = CouplingOperator(d_model, kind, f_hidden, f_layers, gate=gate)
        self.rev = CouplingOperator(d_model, kind, f_hidden, f_layers, gate=gate)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
