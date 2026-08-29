"""PsiLM 2D: frozen LLM + frozen fine-tuned DPOT-Tiny + trainable bridges."""

import torch
import torch.nn.functional as F

from ..bicameral.staged import StreamState
from .bridges2d import build_ic_2d

N_LAYERS = 24
L_FWD = 10
L_REV = 15


class PsiLM2D:
    def __init__(self, model, tokenizer, phys, bridges):
        self.model = model
        self.tok = tokenizer
        self.phys = phys                     # DPOTPhysics, frozen
        self.phi = bridges
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.phys.parameters():
            p.requires_grad_(False)

    def _physics_tokens(self, stream, prompt_mask):
        r = self.phi.fwd(stream.hidden, prompt_mask)
        ic = build_ic_2d(r["a"], r["cx"], r["cy"], r["w"])
        feats, field = self.phys.features_and_field(ic.to(next(self.phys.parameters()).dtype))
        tokens, u_hat = self.phi.rev(feats, field, r["x0"], r["y0"])
        return r, tokens, u_hat

    def train_forward(self, batch, lam_reg=1.0, lam_cls=0.3, lam_u=2.0):
        p = StreamState(self.model, batch["p_ids"], batch["p_attn"])
        p.run(0, L_FWD)
        r, tokens, u_hat = self._physics_tokens(p, batch["prompt_mask"])
        p.run(L_FWD, L_REV)
        p.hidden, sigma = self.phi.inject(p.hidden, tokens)
        p.run(L_REV, N_LAYERS)
        logits = p.finish()

        loss_ans = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            batch["p_labels"][:, 1:].reshape(-1), ignore_index=-100)
        loss_reg = F.mse_loss(r["a"], batch["reg"][:, 0]) + F.mse_loss(r["w"], batch["reg"][:, 1])
        loss_cls = sum(
            F.cross_entropy(lg, batch["bins"][:, i])
            for i, lg in enumerate(r["cls_logits"])
        ) / 4
        loss_u = F.mse_loss(u_hat, batch["u_true"])
        pos_err = (
            (r["x0"].detach() - batch["bins"][:, 2].float() / 100).abs().mean()
            + (r["y0"].detach() - batch["bins"][:, 3].float() / 100).abs().mean()
        ) / 2
        mask = batch["p_attn"].bool()
        return {
            "loss": loss_ans + lam_reg * loss_reg + lam_cls * loss_cls + lam_u * loss_u,
            "loss_ans": loss_ans.detach(), "loss_reg": loss_reg.detach(),
            "loss_cls": loss_cls.detach(), "loss_u": loss_u.detach(),
            "pos_err": pos_err,
            "gate": sigma.detach()[mask.unsqueeze(-1).expand_as(sigma)].mean(),
        }

    @torch.no_grad()
    def generate(self, builder, item, max_new=24):
        """The physics side depends only on prompt positions, whose hidden
        states are fixed under causal attention — so DPOT runs ONCE per
        question and the injected tokens are cached across decoding steps."""
        device = next(self.model.parameters()).device
        prompt = builder.prompt_ids(item)
        ids = list(prompt)
        eos = self.tok.eos_token_id

        t = torch.tensor([ids], device=device)
        pmask = torch.ones_like(t, dtype=torch.bool)
        s = StreamState(self.model, t)
        s.run(0, L_FWD)
        _, tokens, _ = self._physics_tokens(s, pmask)

        for _ in range(max_new):
            tt = torch.tensor([ids], device=device)
            s = StreamState(self.model, tt)
            s.run(0, L_FWD)
            s.run(L_FWD, L_REV)
            s.hidden, _ = self.phi.inject(s.hidden, tokens)
            s.run(L_REV, N_LAYERS)
            nxt = int(s.finish()[:, -1].argmax(-1))
            ids.append(nxt)
            if nxt == eos:
                break
        return self.tok.decode(ids[len(prompt):], skip_special_tokens=True)
