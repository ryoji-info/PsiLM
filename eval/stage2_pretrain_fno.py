"""Pretrain the FNO physics hemisphere on solver data, then freeze it.

Generates (u0, uT) Burgers pairs with the spectral solver, trains the FNO to
map u0 -> u(T) in one shot, reports held-out relative L2 error, and saves
results/stage2/fno.pt.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.physics.burgers import make_dataset  # noqa: E402
from psilm.physics.fno import FNO1d  # noqa: E402


def main(device="mps", n_train=4096, n_val=256, steps=1500, batch=64):
    print("generating solver data...", flush=True)
    t0 = time.time()
    u0, ut, _ = make_dataset(n_train + n_val, seed=0)
    print(f"  {n_train + n_val} trajectories in {time.time() - t0:.0f}s")
    u0 = torch.tensor(u0, dtype=torch.float32)
    ut = torch.tensor(ut, dtype=torch.float32)
    tr0, trT = u0[:n_train].to(device), ut[:n_train].to(device)
    va0, vaT = u0[n_train:].to(device), ut[n_train:].to(device)

    fno = FNO1d().to(device)
    n_params = sum(p.numel() for p in fno.parameters())
    print(f"FNO params: {n_params/1e6:.2f}M")
    opt = torch.optim.Adam(fno.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch,), generator=g)
        pred = fno(tr0[idx])
        loss = ((pred - trT[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 300 == 0:
            with torch.no_grad():
                vp = fno(va0)
                rel = (vp - vaT).norm(dim=-1) / vaT.norm(dim=-1)
            print(f"step {step+1}: train mse={loss.item():.2e} val relL2={rel.mean().item():.4f}")

    out = Path("results/stage2"); out.mkdir(parents=True, exist_ok=True)
    torch.save(fno.state_dict(), out / "fno.pt")
    print(f"saved {out/'fno.pt'}  (final val relL2 {rel.mean().item():.4f})")


if __name__ == "__main__":
    main()
