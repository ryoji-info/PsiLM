"""PsiLM over MLX: frozen (possibly 4-bit) LLM + MLX FNO + trainable bridges."""

import mlx.core as mx
import mlx.nn as nn

from .bridges import build_ic_mlx
from .staged import MlxStream


def cross_entropy_masked(logits, labels):
    """logits (B, L, V), labels (B, L) with -100 = ignore. Shifted CE."""
    lg = logits[:, :-1].astype(mx.float32)
    lb = labels[:, 1:]
    valid = lb != -100
    lb_safe = mx.where(valid, lb, mx.zeros_like(lb))
    ce = nn.losses.cross_entropy(lg.reshape(-1, lg.shape[-1]), lb_safe.reshape(-1),
                                 reduction="none").reshape(lb.shape)
    return (ce * valid).sum() / valid.sum().astype(mx.float32)


class PsiLMMLX:
    def __init__(self, model, tokenizer, fno, bridges, l_fwd=None, l_rev=None):
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

    def _couple(self, stream, prompt_mask):
        stream.run(0, self.l_fwd)
        params_hat, x0_hat, x0_logits, w_x0 = self.phi.fwd(stream.hidden, prompt_mask)
        ic = build_ic_mlx(params_hat)
        feats = self.fno.features(ic)
        u_field = self.fno.proj(feats).squeeze(-1)
        tokens, u_hat = self.phi.rev(feats, u_field, x0_hat)
        stream.run(self.l_fwd, self.l_rev)
        stream.hidden, sigma = self.phi.inject(stream.hidden, tokens)
        stream.run(self.l_rev, self.n_layers)
        return params_hat, x0_hat, x0_logits, u_hat, sigma, w_x0

    def loss_fn(self, batch, lam_param=1.0, lam_x0=0.3, lam_u=2.0, lam_attn=0.5):
        s = MlxStream(self.model, batch["p_ids"], batch["p_attn"])
        params_hat, x0_hat, x0_logits, u_hat, sigma, w_x0 = self._couple(s, batch["prompt_mask"])
        logits = s.finish()
        loss_ans = cross_entropy_masked(logits, batch["p_labels"])
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
        return loss, (loss_ans, loss_param, loss_x0, loss_u,
                      mx.abs(x0_hat - batch["x0"]).mean(), sigma.mean(), loss_attn)

    def generate(self, builder, item, max_new=24):
        prompt = builder.prompt_ids(item)
        ids = list(prompt)
        eos = self.tok.eos_token_id
        for _ in range(max_new):
            t = mx.array([ids])
            pmask = mx.concatenate([mx.ones((1, len(prompt)), dtype=mx.bool_),
                                    mx.zeros((1, len(ids) - len(prompt)), dtype=mx.bool_)], axis=1)
            s = MlxStream(self.model, t)
            self._couple(s, pmask)
            nxt = int(s.finish()[:, -1].argmax(-1).item())
            ids.append(nxt)
            if nxt == eos:
                break
        return self.tok.decode(ids[len(prompt):])
