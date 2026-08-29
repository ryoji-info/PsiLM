"""Pretrain a general-purpose FNO on broad multi-mode Burgers data.

The operator sees random Fourier initial conditions up to mode 4 with random
amplitudes — deliberately broader than any QA family, including the families
held out from bridge training. Generalization tests then probe the BRIDGES,
not the physics model.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.physics.burgers import initial_condition_multi, solve  # noqa: E402
from psilm.physics.fno import FNO1d  # noqa: E402


def random_modes(rng):
    n_active = rng.integers(1, 4)                     # 1..3 active modes
    ms = rng.choice([1, 2, 3, 4], size=n_active, replace=False)
    return [(int(m), float(rng.uniform(0.1, 0.9)), float(rng.uniform(0, 6.28)))
            for m in ms]


def main(device="mps", n_train=6144, n_val=256, steps=2500, batch=64):
    rng = np.random.default_rng(7)
    print("generating multi-mode solver data...", flush=True)
    t0 = time.time()
    u0s, uts = [], []
    for _ in range(n_train + n_val):
        u0 = initial_condition_multi(random_modes(rng))
        u0s.append(u0)
        uts.append(solve(u0))
    print(f"  {n_train + n_val} trajectories in {time.time() - t0:.0f}s")
    u0 = torch.tensor(np.stack(u0s), dtype=torch.float32)
    ut = torch.tensor(np.stack(uts), dtype=torch.float32)
    tr0, trT = u0[:n_train].to(device), ut[:n_train].to(device)
    va0, vaT = u0[n_train:].to(device), ut[n_train:].to(device)

    fno = FNO1d().to(device)
    opt = torch.optim.Adam(fno.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch,), generator=g)
        loss = ((fno(tr0[idx]) - trT[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                rel = ((fno(va0) - vaT).norm(dim=-1) / vaT.norm(dim=-1)).mean()
            print(f"step {step+1}: mse={loss.item():.2e} val relL2={rel.item():.4f}")

    out = Path("results/stage2b"); out.mkdir(parents=True, exist_ok=True)
    torch.save(fno.state_dict(), out / "fno.pt")
    print(f"saved {out/'fno.pt'}")


if __name__ == "__main__":
    main()
