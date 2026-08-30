"""PsiLM with looped coupling: multiple read->rollout->inject passes.

Pass k reads the residual stream at depth r_k, produces a physics readout,
rolls the frozen operator, and injects the result at depth w_k, with
r_1 < w_1 < r_2 < w_2 < ... — so every later readout sees the stream AFTER
the previous injection and can revise it. This is the 'more paths from
language to physics and back' design: the ablation question is whether the
revision loop buys generalization (the combination family's headroom) at
matched training budget.

`shared=True` reuses one bridge set across passes (a recurrence);
`shared=False` gives each pass its own bridges (independent paths).
All passes' readouts are supervised; the answer loss is computed once.
"""

import copy

import torch
import torch.nn.functional as F

from ..bicameral.staged import StreamState
from .bridges import PsiBridges, build_ic


class PsiLMLoop:
    # read/write depths as fractions of backbone depth, one pair per pass
    PASS_FRACS = [(8 / 24, 11 / 24), (15 / 24, 18 / 24)]

    def __init__(self, model, tokenizer, fno, bridges, ic_fn=build_ic,
                 n_passes=2, shared=True):
        self.model = model
        self.tok = tokenizer
        self.fno = fno
        self.ic_fn = ic_fn
        self.n_passes = n_passes
        n = model.config.num_hidden_layers
        self.n_layers = n
        self.depths = [(round(r * n), round(w * n))
                       for r, w in self.PASS_FRACS[:n_passes]]
        if shared:
            self.phis = [bridges] * n_passes
        else:
            self.phis = [bridges] + [copy.deepcopy(bridges) for _ in range(n_passes - 1)]
        self._modules = torch.nn.ModuleList(
            self.phis if not shared else [bridges])
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.fno.parameters():
            p.requires_grad_(False)

    def parameters(self):
        return self._modules.parameters()

    def n_params(self):
        return sum(p.numel() for p in self._modules.parameters())

    def _couple(self, stream, prompt_mask):
        """Run all coupling passes; return per-pass readouts and gates."""
        passes = []
        pos = 0
        for k, (r, w) in enumerate(self.depths):
            stream.run(pos, r)
            phi = self.phis[k]
            params_hat, x0_hat, x0_logits, amp_logits = phi.fwd(stream.hidden, prompt_mask)
            ic = self.ic_fn(params_hat)
            feats = self.fno.features(ic)
            u_field = self.fno.proj(feats).squeeze(-1)
            phys_tokens, u_hat = phi.rev(feats, u_field, x0_hat)
            stream.run(r, w)
            stream.hidden, sigma = phi.inject(stream.hidden, phys_tokens)
            passes.append({"params": params_hat, "x0": x0_hat,
                           "x0_logits": x0_logits, "amp_logits": amp_logits,
                           "u_hat": u_hat, "sigma": sigma})
            pos = w
        stream.run(pos, self.n_layers)
        return passes

    def train_forward(self, batch, lam_param=1.0, lam_x0=0.3, lam_u=2.0):
        p = StreamState(self.model, batch["p_ids"], batch["p_attn"])
        passes = self._couple(p, batch["prompt_mask"])
        logits = p.finish()
        loss_ans = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            batch["p_labels"][:, 1:].reshape(-1), ignore_index=-100)

        loss_param = loss_x0 = loss_u = 0.0
        mask_p = batch["param_mask"]
        x0_target = (batch["x0"] * 100).round().long().clamp(0, 99)
        for r in passes:
            if r["amp_logits"] is not None:
                n_modes = r["amp_logits"].shape[1]
                la = F.cross_entropy(r["amp_logits"].reshape(-1, r["amp_logits"].shape[-1]),
                                     batch["amp_bins"].reshape(-1))
                sc = [i for m in range(n_modes) for i in (3 * m + 1, 3 * m + 2)]
                scm = mask_p[:, sc]
                ls = (scm * (r["params"][:, sc] - batch["params"][:, sc]) ** 2
                      ).sum() / scm.sum().clamp(min=1)
                loss_param = loss_param + la + ls
            else:
                loss_param = loss_param + (mask_p * (r["params"] - batch["params"]) ** 2
                                           ).sum() / mask_p.sum()
            loss_x0 = loss_x0 + F.cross_entropy(r["x0_logits"], x0_target)
            loss_u = loss_u + F.mse_loss(r["u_hat"], batch["u_true"])
        k = len(passes)
        loss_param, loss_x0, loss_u = loss_param / k, loss_x0 / k, loss_u / k

        last = passes[-1]
        mask = batch["p_attn"].bool()
        return {
            "loss": loss_ans + lam_param * loss_param + lam_x0 * loss_x0 + lam_u * loss_u,
            "loss_ans": loss_ans.detach(),
            "loss_param": loss_param.detach(),
            "loss_x0": loss_x0.detach(),
            "loss_u": loss_u.detach(),
            "x0_err": (last["x0"].detach() - batch["x0"]).abs().mean(),
            "gate": last["sigma"].detach()[mask.unsqueeze(-1).expand_as(last["sigma"])].mean(),
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
