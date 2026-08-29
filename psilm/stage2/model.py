"""PsiLM Stage 2: frozen LLM + frozen FNO + trainable bridges."""

import torch
import torch.nn.functional as F

from ..bicameral.staged import StreamState
from .bridges import build_ic

N_LAYERS = 24
L_FWD = 10    # read language hidden states here
L_REV = 15    # inject physics tokens here


class PsiLM:
    def __init__(self, model, tokenizer, fno, bridges, ic_fn=build_ic):
        self.model = model
        self.tok = tokenizer
        self.fno = fno
        self.phi = bridges
        self.ic_fn = ic_fn
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.fno.parameters():
            p.requires_grad_(False)

    def _couple(self, stream, prompt_mask):
        """Run the coupled pass up to the final layers.

        Returns (params_hat, x0_hat, x0_logits, u_hat, gate) for losses.
        """
        stream.run(0, L_FWD)
        params_hat, x0_hat, x0_logits = self.phi.fwd(stream.hidden, prompt_mask)
        ic = self.ic_fn(params_hat)
        feats = self.fno.features(ic)                       # (B, N, W) fp32
        u_field = self.fno.proj(feats).squeeze(-1)          # (B, N)
        phys_tokens, u_hat = self.phi.rev(feats, u_field, x0_hat)
        stream.run(L_FWD, L_REV)
        stream.hidden, sigma = self.phi.inject(stream.hidden, phys_tokens)
        stream.run(L_REV, N_LAYERS)
        return params_hat, x0_hat, x0_logits, u_hat, sigma

    def train_forward(self, batch, lam_param=1.0, lam_x0=0.3, lam_u=2.0):
        p = StreamState(self.model, batch["p_ids"], batch["p_attn"])
        params_hat, x0_hat, x0_logits, u_hat, sigma = self._couple(p, batch["prompt_mask"])
        logits = p.finish()
        loss_ans = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            batch["p_labels"][:, 1:].reshape(-1),
            ignore_index=-100,
        )
        mask_p = batch["param_mask"]
        loss_param = (mask_p * (params_hat - batch["params"]) ** 2).sum() / mask_p.sum()
        x0_target = (batch["x0"] * 100).round().long().clamp(0, 99)
        loss_x0 = F.cross_entropy(x0_logits, x0_target)
        loss_u = F.mse_loss(u_hat, batch["u_true"])
        mask = batch["p_attn"].bool()
        return {
            "loss": loss_ans + lam_param * loss_param + lam_x0 * loss_x0 + lam_u * loss_u,
            "loss_ans": loss_ans.detach(),
            "loss_param": loss_param.detach(),
            "loss_x0": loss_x0.detach(),
            "loss_u": loss_u.detach(),
            "x0_err": (x0_hat.detach() - batch["x0"]).abs().mean(),
            "gate": sigma.detach()[mask.unsqueeze(-1).expand_as(sigma)].mean(),
        }

    @torch.no_grad()
    def generate(self, builder, item, max_new=24):
        device = next(self.model.parameters()).device
        prompt = builder.prompt_ids(item)
        ids = list(prompt)
        eos = self.tok.eos_token_id
        for _ in range(max_new):
            t = torch.tensor([ids], device=device)
            pmask = torch.zeros_like(t, dtype=torch.bool)
            pmask[:, : len(prompt)] = True
            s = StreamState(self.model, t)
            self._couple(s, pmask)
            nxt = int(s.finish()[:, -1].argmax(-1))
            ids.append(nxt)
            if nxt == eos:
                break
        return self.tok.decode(ids[len(prompt):], skip_special_tokens=True)
