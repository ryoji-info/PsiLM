"""PsiLM 2D over MLX (hybrid): frozen MLX LLM + torch DPOT-Tiny + MLX bridges.

Mirror of psilm/mlx/model.py for the Stage-2d task. The coupling:

  layers 0..l_fwd  -> ForwardBridge2DMLX (span-pooled six-way readout)
                   -> TorchPhysics2D(params_hat, detached, numpy)   [no grad]
                   -> lookup(field, x0_hat, y0_hat), both detached  [no grad]
                   -> ValueTokensMLX(u)  -> GatedCrossAttentionMLX at l_rev
  layers l_rev..n  -> logits

Losses: digit-weighted answer CE + lam_cls * sum of the six readout CEs.
Phase A (readout_only) trains the six classifiers through layers 0..l_fwd
with no physics call. The no-harm arm follows psilm/mlx/model.py with every
span = the whole prompt.
"""

import mlx.core as mx
import mlx.nn as nn

from ..stage2d.qa2d import QUANTITIES
from .model import cross_entropy_masked
from .physics2d import lookup
from .staged import MlxStream

PERIODIC_IDX = [QUANTITIES.index(k) for k in ("cx", "cy", "x0", "y0")]


def _bin_err(est, target_bins, periodic):
    """mean |estimate - target| on the unit grid (periodic distance for positions)."""
    d = mx.abs(est - target_bins.astype(mx.float32) / 100)
    return (mx.minimum(d, 1 - d) if periodic else d).mean()


class PsiLM2DMLX:
    def __init__(self, model, tokenizer, phys, bridges, l_fwd=None, l_rev=None,
                 digit_weight=5.0, lam_cls=1.0):
        self.model = model
        self.tok = tokenizer
        self.phys = phys                    # TorchPhysics2D (frozen, torch side)
        self.phi = bridges
        n = len(model.model.layers)
        self.n_layers = n
        self.l_fwd = l_fwd if l_fwd is not None else round(n * 10 / 24)
        self.l_rev = l_rev if l_rev is not None else round(n * 15 / 24)
        model.freeze()
        self.digit_weight = digit_weight
        self.lam_cls = lam_cls
        self.readout_only = False
        self.detach_x0 = True               # the lookup coordinates are always detached here
        self.lam_gate = 0.0
        toks = [str(d) for d in range(10)] + [".", "-", " -"]
        ids = set()
        for t in toks:
            enc = tokenizer.encode(t)
            if len(enc) == 1:
                ids.add(enc[0])
        self.digit_ids = mx.array(sorted(ids), dtype=mx.int64)

    # ---- coupling -----------------------------------------------------------
    def _physics(self, r):
        """Detached physics call: DPOT on the read IC parameters, lookup at the
        read query point. Returns (u_val (B,), field (B, G, G)); nothing here
        carries gradient."""
        params = mx.stop_gradient(r["params"])
        field = self.phys(params)                                  # torch -> mx, no grad
        x0, y0 = mx.stop_gradient(r["x0"]), mx.stop_gradient(r["y0"])
        u_val = lookup(field, x0, y0)
        return u_val, field

    def _couple(self, stream, prompt_mask, spans):
        stream.run(0, self.l_fwd)
        r = self.phi.fwd(stream.hidden, prompt_mask, spans)
        u_val, field = self._physics(r)
        # the language side learns to READ the physics side's answer; the value
        # is a constant of the graph (the readouts train on their own CEs)
        tokens = self.phi.val(u_val)
        stream.run(self.l_fwd, self.l_rev)
        stream.hidden, sigma, ratio = self.phi.inject(stream.hidden, tokens, return_ratio=True)
        stream.run(self.l_rev, self.n_layers)
        self._last_ratio = ratio
        self._last_field = field
        return r, u_val, sigma

    # ---- losses -------------------------------------------------------------
    def _readout_losses(self, r, batch):
        ces = [nn.losses.cross_entropy(lg, batch["qbins"][:, i], reduction="mean")
               for i, lg in enumerate(r["logits"])]
        exact = mx.stack([(lg.argmax(-1) == batch["qbins"][:, i]).astype(mx.float32).mean()
                          for i, lg in enumerate(r["logits"])])
        pos_err = sum(_bin_err(r[QUANTITIES[i]], batch["qbins"][:, i], True)
                      for i in PERIODIC_IDX) / len(PERIODIC_IDX)
        return ces, exact, pos_err

    def _noharm_loss(self, batch):
        """Gate-selectivity arm (see psilm/mlx/model.py): a non-physics prompt
        through the full coupled pipeline (every span = whole prompt, DPOT on
        whatever the readouts say), trained to reproduce the frozen backbone's
        own continuation with a plain CE; optional mean-gate penalty."""
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        _, _, sigma = self._couple(s, batch["prompt_mask"], batch["spans"])
        logits = s.finish()
        ce = cross_entropy_masked(logits, batch["p_labels"], None, 1.0)
        valid = batch["p_attn"].astype(mx.float32)
        gate_mean = (sigma[..., 0] * valid).sum() / (valid.sum() + 1e-6)
        loss = ce + self.lam_gate * gate_mean
        resp = (batch["p_labels"] != -100).astype(mx.float32)
        n_resp = resp.sum() + 1e-6
        zero = mx.array(0.0)
        aux = {"loss_ans": ce, "loss_cls": zero, "u_err": zero, "u_err_oracle": zero,
               "pos_err": zero, "gate": gate_mean, "gate_ans": (sigma[..., 0] * resp).sum() / n_resp,
               "inj_ratio_ans": (self._last_ratio * resp).sum() / n_resp,
               "exact": mx.zeros((len(QUANTITIES),))}
        return loss, aux

    def _readout_only_loss(self, batch, lam_cls):
        """Phase A: the six readout CEs through layers 0..l_fwd; no physics."""
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        s.run(0, self.l_fwd)
        r = self.phi.fwd(s.hidden, batch["prompt_mask"], batch["spans"])
        ces, exact, pos_err = self._readout_losses(r, batch)
        loss_cls = sum(ces)
        zero = mx.array(0.0)
        aux = {"loss_ans": zero, "loss_cls": loss_cls, "u_err": zero, "u_err_oracle": zero,
               "pos_err": pos_err, "gate": zero, "gate_ans": zero, "inj_ratio_ans": zero,
               "exact": exact, "ces": mx.stack(ces)}
        return lam_cls * loss_cls, aux

    def loss_fn(self, batch, lam_cls=None):
        if lam_cls is None:
            lam_cls = self.lam_cls
        if self.readout_only:
            return self._readout_only_loss(batch, lam_cls)
        if batch.get("noharm"):
            return self._noharm_loss(batch)
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        r, u_val, sigma = self._couple(s, batch["prompt_mask"], batch["spans"])
        logits = s.finish()
        loss_ans = cross_entropy_masked(logits, batch["p_labels"], self.digit_ids, self.digit_weight)
        ces, exact, pos_err = self._readout_losses(r, batch)
        loss_cls = sum(ces)
        loss = loss_ans + lam_cls * loss_cls
        # diagnostics: the channel's cargo vs the true answer (at the read
        # point, and at the TRUE point on the same predicted field = the
        # physics side's own error), gate and channel strength at the answer
        u_err = mx.abs(u_val - batch["u_true"]).mean()
        u_oracle = lookup(self._last_field, batch["x0"], batch["y0"])
        u_err_oracle = mx.abs(u_oracle - batch["u_true"]).mean()
        resp = (batch["p_labels"] != -100).astype(mx.float32)
        n_resp = resp.sum() + 1e-6
        aux = {"loss_ans": loss_ans, "loss_cls": loss_cls, "u_err": u_err,
               "u_err_oracle": u_err_oracle, "pos_err": pos_err, "gate": sigma.mean(),
               "gate_ans": (sigma[..., 0] * resp).sum() / n_resp,
               "inj_ratio_ans": (self._last_ratio * resp).sum() / n_resp,
               "exact": exact, "ces": mx.stack(ces)}
        return loss, aux

    # ---- generation ---------------------------------------------------------
    def generate(self, builder, item, max_new=32):    # the 2D reply is 22 tokens (+eos) on Qwen/Gemma
        prompt = builder.prompt_ids(item)
        ids = list(prompt)
        eos = self.tok.eos_token_id
        spans = mx.array([builder.spans(prompt, item)], dtype=mx.int32)      # (1, 6, 2)
        for _ in range(max_new):
            t = mx.array([ids])
            pmask = mx.concatenate([mx.ones((1, len(prompt)), dtype=mx.bool_),
                                    mx.zeros((1, len(ids) - len(prompt)), dtype=mx.bool_)], axis=1)
            s = MlxStream(self.model, t)
            self._couple(s, pmask, spans)
            nxt = int(s.finish()[:, -1].argmax(-1).item())
            ids.append(nxt)
            if nxt == eos:
                break
        return self.tok.decode(ids[len(prompt):])
