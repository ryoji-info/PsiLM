"""Stage 2b on MLX: the multi-mode Burgers task over the 4-bit-capable stack.

Port of the torch Stage-2b pieces (psilm/stage2/bridges.py build_ic_multi,
psilm/stage2/model.py with ic_fn=build_ic_multi and the param_mask MSE) onto
PsiLMMLX. Nothing of the coupling is copied: PsiLMMLX exposes the IC builder
and the readout-parameter loss as pluggable hooks (ic_fn, param_loss_fn) and
PsiLMMLXMulti only fills them in, so _couple / loss_fn / _readout_only_loss /
_noharm_loss / generate are the exact code that produced the Qwen3-8B and
Gemma 4 12B single-mode results.

What is multi-mode here and what is not:
  * the forward readout regresses 3 * N_MODES = 6 values (a, sin phi, cos phi)
    per mode from the deterministic prompt pool, absent modes supervised to
    a = 0 through batch["param_mask"] exactly as the torch version;
  * the IC is u0 = sum_m a_m sin(2 pi m x + phi_m) on the 128-point grid, fed
    to the multi-mode FNO of results/stage2b/fno.pt (same FNO1d architecture
    as the single-mode one, so psilm/mlx/fno.convert_from_torch applies);
  * everything downstream -- span-pooled x0 classifier with circular mean,
    lookup at x0, the value-token channel carrying only u(x0), gated
    injection -- is untouched.

Run on Gemma 4 12B (the settings of results/stage2_gemma12b/bridges.npz.meta):
  python eval/mlx_stage2b_train.py --model mlx-community/gemma-4-12B-it-4bit \
      --hf-tokenizer mlx-community/gemma-4-12B-it-4bit --tag _gemma12b_2b \
      --batch 4 --steps 500 --fresh --readout-only 2000 --detach-x0 \
      --clip module --lam-x0 1.0 --channel value --readout-norm dim \
      --calib-n 32 --inj-cap 0.2 --gate-bias 0.0 --eval-n 48
"""

import math

import mlx.core as mx

from .bridges import PsiBridgesMLX
from .model import PsiLMMLX

N_MODES = 2                      # psilm.stage2.qa2.N_MODES
N_PARAMS = 3 * N_MODES


def build_ic_multi_mlx(params, n: int = 128):
    """u0 = sum_m a_m sin(2 pi m x + phi_m); params (B, 3*n_modes) as
    (a, sin phi, cos phi) triplets. Mirror of psilm.stage2.bridges.build_ic_multi:
    the (sin, cos) pair is projected to the unit circle, and an absent mode's
    a ~ 0 zeroes its term regardless of the (sin, cos) readout."""
    n_modes = params.shape[1] // 3
    x = mx.arange(n, dtype=mx.float32) / n
    u0 = mx.zeros((params.shape[0], n), dtype=mx.float32)
    for m in range(1, n_modes + 1):
        a = params[:, 3 * m - 3: 3 * m - 2]
        sc = params[:, 3 * m - 2: 3 * m]
        sc = sc / (mx.sqrt((sc ** 2).sum(axis=-1, keepdims=True)) + 1e-6)
        arg = 2 * math.pi * m * x
        u0 = u0 + a * (mx.sin(arg)[None, :] * sc[:, 1:2] + mx.cos(arg)[None, :] * sc[:, 0:1])
    return u0


def param_loss_multi(params_hat, batch):
    """Masked MSE against batch['params'] (6 values), exactly the torch Stage-2b
    'pooled' loss: mask covers present modes' (a, sin, cos) and absent modes'
    amplitude (supervised to 0); absent modes' phases are unsupervised."""
    m = batch["param_mask"]
    return (m * (params_hat - batch["params"]) ** 2).sum() / mx.maximum(m.sum(), mx.array(1.0))


def make_bridges_multi(d_model: int, **kw) -> PsiBridgesMLX:
    """PsiBridgesMLX with the 6-parameter forward readout; kw as PsiBridgesMLX
    (gate_bias, inj_cap, channel, readout_norm)."""
    return PsiBridgesMLX(d_model=d_model, n_params=N_PARAMS, **kw)


class PsiLMMLXMulti(PsiLMMLX):
    """PsiLMMLX with the multi-mode IC builder and the masked 6-value readout
    loss. All coupling, loss, and generation code is inherited verbatim."""

    def __init__(self, model, tokenizer, fno, bridges, **kw):
        n_out = bridges.fwd.mlp2.weight.shape[0]
        if n_out != N_PARAMS:
            raise ValueError(f"multi-mode bridges need a {N_PARAMS}-value forward readout, "
                             f"got {n_out} (build them with make_bridges_multi)")
        super().__init__(model, tokenizer, fno, bridges,
                         ic_fn=build_ic_multi_mlx, param_loss_fn=param_loss_multi, **kw)
