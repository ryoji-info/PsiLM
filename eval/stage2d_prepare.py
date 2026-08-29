"""Stage-2d preparation: solver data, DPOT fine-tune, QA datasets.

1. Generate Fisher-KPP trajectories with the spectral solver (ground truth).
2. Fine-tune pretrained DPOT-Tiny to map the replicated IC history to
   u(., T_FINAL) in one shot; freeze thereafter.
3. Build QA items (question params + query point + interpolated answer),
   reusing the same solver fields.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.physics.dpot_wrapper import DPOTPhysics  # noqa: E402
from psilm.physics.fisher2d import (  # noqa: E402
    bilinear_periodic, initial_condition, sample_ic, sample_query, solve)

OUT = Path("results/stage2d")
N_TRAIN, N_VAL = 3072, 256
QA_TRAIN_TRAJ, QA_PER_TRAJ = 1536, 6


def main(device="mps"):
    rng = random.Random(42)
    t0 = time.time()
    params, u0s, uts = [], [], []
    for i in range(N_TRAIN + N_VAL):
        a, cx, cy, w = sample_ic(rng)
        u0 = initial_condition(a, cx, cy, w)
        params.append((a, cx, cy, w))
        u0s.append(u0.astype(np.float32))
        uts.append(solve(u0).astype(np.float32))
        if (i + 1) % 500 == 0:
            print(f"  solver {i + 1}/{N_TRAIN + N_VAL} ({time.time() - t0:.0f}s)", flush=True)
    u0s, uts = np.stack(u0s), np.stack(uts)
    np.savez_compressed(OUT / "fisher2d_data.npz", u0=u0s, ut=uts,
                        params=np.array(params, dtype=np.float32))
    print(f"solver data done in {time.time() - t0:.0f}s")

    # ---- fine-tune DPOT ----
    phys = DPOTPhysics(device=device)
    opt = torch.optim.AdamW(phys.parameters(), lr=1e-4, weight_decay=1e-5)
    steps, batch = 1200, 8
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tr0 = torch.tensor(u0s[:N_TRAIN])
    trT = torch.tensor(uts[:N_TRAIN])
    va0 = torch.tensor(u0s[N_TRAIN:]).to(device)
    vaT = torch.tensor(uts[N_TRAIN:]).to(device)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, N_TRAIN, (batch,), generator=g)
        pred = phys(tr0[idx].to(device))
        loss = ((pred - trT[idx].to(device)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                errs = []
                for j in range(0, N_VAL, 32):
                    vp = phys(va0[j:j + 32])
                    errs.append(((vp - vaT[j:j + 32]).norm(dim=(1, 2))
                                 / vaT[j:j + 32].norm(dim=(1, 2))))
                rel = torch.cat(errs).mean().item()
            print(f"  ft step {step + 1}: mse={loss.item():.2e} val relL2={rel:.4f} "
                  f"({(time.time() - t0) / (step + 1):.2f}s/step)", flush=True)
    torch.save(phys.net.state_dict(), OUT / "dpot_ft.pt")
    print(f"fine-tuned DPOT saved (val relL2 {rel:.4f})")

    # ---- QA datasets ----
    rng_q = random.Random(777)
    def build(idx_range, per, path):
        items = []
        for i in idx_range:
            a, cx, cy, w = params[i]
            for _ in range(per):
                x0, y0 = sample_query(rng_q, cx, cy)
                items.append({"a": a, "cx": cx, "cy": cy, "w": w,
                              "x0": x0, "y0": y0,
                              "u": round(bilinear_periodic(uts[i], x0, y0), 4)})
        Path(path).write_text(json.dumps(items))
        return items

    tr = build(range(QA_TRAIN_TRAJ), QA_PER_TRAJ, "data/stage2d_qa_train.json")
    va = build(range(N_TRAIN, N_TRAIN + N_VAL), 4, "data/stage2d_qa_val.json")
    us = np.array([it["u"] for it in tr])
    print(f"QA: train {len(tr)}, val {len(va)}; mean|u|={np.abs(us).mean():.3f} "
          f"zero-acc={float((np.abs(us) <= 0.05).mean()):.3f}")


if __name__ == "__main__":
    main()
