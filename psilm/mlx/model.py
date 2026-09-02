"""PsiLM over MLX: frozen (possibly 4-bit) LLM + MLX FNO + trainable bridges."""

import mlx.core as mx
import mlx.nn as nn

from .bridges import build_ic_mlx
from .staged import MlxStream


def cross_entropy_masked(logits, labels, digit_ids=None, digit_weight=5.0):
    """logits (B, L, V), labels (B, L) with -100 = ignore. Shifted CE.
    Targets in digit_ids get digit_weight: the channel's cargo is the digit
    tokens, and strong backbones' answer CE is otherwise dominated by
    already-solved format tokens."""
    lg = logits[:, :-1].astype(mx.float32)
    lb = labels[:, 1:]
    valid = lb != -100
    lb_safe = mx.where(valid, lb, mx.zeros_like(lb))
    ce = nn.losses.cross_entropy(lg.reshape(-1, lg.shape[-1]), lb_safe.reshape(-1),
                                 reduction="none").reshape(lb.shape)
    w = valid.astype(mx.float32)
    if digit_ids is not None:
        is_digit = (lb[..., None] == digit_ids[None, None, :]).any(axis=-1)
        w = w * mx.where(is_digit, mx.array(digit_weight), mx.array(1.0))
    return (ce * w).sum() / w.sum()


class PsiLMMLX:
    def __init__(self, model, tokenizer, fno, bridges, l_fwd=None, l_rev=None,
                 digit_weight=5.0, lam_x0=0.3):
        self.model = model
        self.tok = tokenizer
        self.fno = fno
        self.phi = bridges
        n = len(model.model.layers)
        self.n_layers = n
        self.l_fwd = l_fwd if l_fwd is not None else round(n * 10 / 24)
        self.l_rev = l_rev if l_rev is not None else round(n * 15 / 24)
        model.freeze()
        fno.freeze()
        self.digit_weight = digit_weight
        self.lam_x0 = lam_x0
        # v6 training controls (set by the trainer):
        #  detach_x0    - stop_gradient on the pointer fed to the reverse bridge, so the
        #                 x0 head sees only its CE (the regime the readout probe converged in)
        #  readout_only - phase A: train fwd + lookup on the TRUE x0 through layers 0..l_fwd
        #                 only; no injection, no answer loss (pointer locks before the
        #                 channel ever opens, at ~half the step cost)
        self.detach_x0 = False
        self.readout_only = False
        toks = [str(d) for d in range(10)] + [".", "-", " -"]
        ids = set()
        for t in toks:
            enc = tokenizer.encode(t)
            if len(enc) == 1:
                ids.add(enc[0])
        self.digit_ids = mx.array(sorted(ids), dtype=mx.int64)

    def _couple(self, stream, prompt_mask, x0_span=None):
        stream.run(0, self.l_fwd)
        params_hat, x0_hat, x0_logits, w_x0 = self.phi.fwd(stream.hidden, prompt_mask, x0_span)
        ic = build_ic_mlx(params_hat)
        feats = self.fno.features(ic)
        u_field = self.fno.proj(feats).squeeze(-1)
        x0_in = mx.stop_gradient(x0_hat) if self.detach_x0 else x0_hat
        tokens, u_hat = self.phi.rev(feats, u_field, x0_in)
        stream.run(self.l_fwd, self.l_rev)
        stream.hidden, sigma, ratio = self.phi.inject(stream.hidden, tokens, return_ratio=True)
        stream.run(self.l_rev, self.n_layers)
        self._last_ratio = ratio
        return params_hat, x0_hat, x0_logits, u_hat, sigma, w_x0

    def _readout_only_loss(self, batch, lam_param, lam_x0, lam_u):
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        s.run(0, self.l_fwd)
        params_hat, x0_hat, x0_logits, _ = self.phi.fwd(s.hidden, batch["prompt_mask"],
                                                        batch.get("x0_span"))
        ic = build_ic_mlx(params_hat)
        feats = self.fno.features(ic)
        u_field = self.fno.proj(feats).squeeze(-1)
        _, u_hat = self.phi.rev(feats, u_field, batch["x0"])   # lookup at the TRUE x0
        loss_param = ((params_hat - batch["params"]) ** 2).mean()
        loss_x0 = nn.losses.cross_entropy(x0_logits, batch["x0_bins"], reduction="mean")
        loss_u = ((u_hat - batch["u_true"]) ** 2).mean()
        loss = lam_param * loss_param + lam_x0 * loss_x0 + lam_u * loss_u
        zero = mx.array(0.0)
        x0_exact = (x0_logits.argmax(-1) == batch["x0_bins"]).astype(mx.float32).mean()
        d = mx.abs(x0_hat - batch["x0"])
        return loss, (zero, loss_param, loss_x0, loss_u,
                      mx.minimum(d, 1 - d).mean(), zero, zero, zero, zero, x0_exact)

    def loss_fn(self, batch, lam_param=1.0, lam_x0=None, lam_u=2.0, lam_attn=0.5):
        if lam_x0 is None:
            lam_x0 = self.lam_x0
        if self.readout_only:
            return self._readout_only_loss(batch, lam_param, lam_x0, lam_u)
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        params_hat, x0_hat, x0_logits, u_hat, sigma, w_x0 = self._couple(
            s, batch["prompt_mask"], batch.get("x0_span"))
        logits = s.finish()
        loss_ans = cross_entropy_masked(logits, batch["p_labels"], self.digit_ids, self.digit_weight)
        loss_param = ((params_hat - batch["params"]) ** 2).mean()
        x0_tgt = batch["x0_bins"]
        loss_x0 = nn.losses.cross_entropy(x0_logits, x0_tgt, reduction="mean")
        loss_u = ((u_hat - batch["u_true"]) ** 2).mean()
        # attention supervision: the x0 pool must place its mass on the
        # x0 token span (the cold-start fix for wide backbones)
        L = w_x0.shape[1]
        pos = mx.arange(L)[None, :]
        span = batch["x0_span"]
        in_span = (pos >= span[:, 0:1]) & (pos < span[:, 1:2])
        mass = (w_x0 * in_span).sum(axis=-1)
        loss_attn = -mx.log(mass + 1e-6).mean()
        loss = (loss_ans + lam_param * loss_param + lam_x0 * loss_x0
                + lam_u * loss_u + lam_attn * loss_attn)
        # diagnostics: the gate and the effective channel strength AT THE ANSWER
        # positions (a mean over all positions hides a selective gate), and
        # exact-bin pointer accuracy
        resp = (batch["p_labels"] != -100).astype(mx.float32)
        n_resp = resp.sum() + 1e-6
        gate_ans = (sigma[..., 0] * resp).sum() / n_resp
        ratio_ans = (self._last_ratio * resp).sum() / n_resp
        x0_exact = (x0_logits.argmax(-1) == x0_tgt).astype(mx.float32).mean()
        d = mx.abs(x0_hat - batch["x0"])
        x0_err = mx.minimum(d, 1 - d).mean()           # periodic distance
        return loss, (loss_ans, loss_param, loss_x0, loss_u,
                      x0_err, sigma.mean(), loss_attn,
                      gate_ans, ratio_ans, x0_exact)

    def generate(self, builder, item, max_new=24):
        prompt = builder.prompt_ids(item)
        ids = list(prompt)
        eos = self.tok.eos_token_id
        span = mx.array([list(builder.x0_span(prompt, item))], dtype=mx.int32)
        for _ in range(max_new):
            t = mx.array([ids])
            pmask = mx.concatenate([mx.ones((1, len(prompt)), dtype=mx.bool_),
                                    mx.zeros((1, len(ids) - len(prompt)), dtype=mx.bool_)], axis=1)
            s = MlxStream(self.model, t)
            self._couple(s, pmask, span)
            nxt = int(s.finish()[:, -1].argmax(-1).item())
            ids.append(nxt)
            if nxt == eos:
                break
        return self.tok.decode(ids[len(prompt):])
