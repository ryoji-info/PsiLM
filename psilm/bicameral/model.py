"""The Bicameral model: two frozen Qwen streams + the trainable interface.

Training runs both streams teacher-forced over full sequences with coupling
applied position-wise on the aligned window (the paper's parallel-SFT trick).
Inference runs the two streams in lockstep, one token each per step, with the
calculator tool forcing results into the auxiliary stream only.
"""

import torch
import torch.nn.functional as F

from .calculator import check_call
from .staged import StreamState

N_LAYERS = 24


class Bicameral:
    def __init__(self, model, tokenizer, interface, l_fwd=10, l_rev=15):
        self.model = model
        self.tok = tokenizer
        self.phi = interface
        self.l_fwd, self.l_rev = l_fwd, l_rev
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---------------- training ----------------

    def train_forward(self, batch):
        Pa = batch["aux_prompt_len"]
        Lp = batch["p_ids"].shape[1]

        p = StreamState(self.model, batch["p_ids"], batch["p_attn"])
        a = StreamState(self.model, batch["a_ids"], batch["a_attn"])
        p.run(0, self.l_fwd)
        a.run(0, self.l_fwd)

        # forward coupling p->a on the aligned window
        aw = a.hidden[:, Pa:Pa + Lp]
        new_aw, sig_f, _ = self.phi.fwd(p.hidden[:, :Lp], aw)
        a.hidden = torch.cat([a.hidden[:, :Pa], new_aw, a.hidden[:, Pa + Lp:]], dim=1)

        a.run(self.l_fwd, self.l_rev)
        p.run(self.l_fwd, self.l_rev)

        # reverse coupling a->p
        p15 = p.hidden[:, :Lp]
        new_p, sig_r, _ = self.phi.rev(a.hidden[:, Pa:Pa + Lp], p15)
        p.hidden = torch.cat([new_p, p.hidden[:, Lp:]], dim=1)

        p.run(self.l_rev, N_LAYERS)
        a.run(self.l_rev, N_LAYERS)
        logits_p, logits_a = p.finish(), a.finish()

        loss_p = _ce(logits_p, batch["p_labels"])
        loss_a = _ce(logits_a, batch["a_labels"])
        window_mask = batch["p_attn"][:, :Lp].bool()
        with torch.no_grad():
            # paper Fig-4 metrics: mean gated perturbation magnitude per direction
            pert_f = (sig_f * (new_aw.float() - aw.float())).norm(dim=-1)[window_mask].mean()
            pert_r = (sig_r * (new_p.float() - p15.float())).norm(dim=-1)[window_mask].mean()
        return {
            "loss": loss_p + loss_a,
            "loss_p": loss_p.detach(),
            "loss_a": loss_a.detach(),
            "gate_fwd": sig_f.detach()[window_mask.unsqueeze(-1).expand_as(sig_f)].mean(),
            "gate_rev": sig_r.detach()[window_mask.unsqueeze(-1).expand_as(sig_r)].mean(),
            "pert_fwd": pert_f,
            "pert_rev": pert_r,
        }

    # ---------------- inference ----------------

    @torch.no_grad()
    def _step_logits(self, p_ids, a_ids, aux_prompt_len):
        """Full recompute (cacheless v1): coupled staged pass, last-position logits."""
        Lp = p_ids.shape[1]
        p = StreamState(self.model, p_ids)
        a = StreamState(self.model, a_ids)
        p.run(0, self.l_fwd)
        a.run(0, self.l_fwd)
        aw = a.hidden[:, aux_prompt_len:aux_prompt_len + Lp]
        new_aw, _, _ = self.phi.fwd(p.hidden[:, :Lp], aw)
        a.hidden = torch.cat([a.hidden[:, :aux_prompt_len], new_aw], dim=1)
        a.run(self.l_fwd, self.l_rev)
        p.run(self.l_fwd, self.l_rev)
        new_p, _, _ = self.phi.rev(a.hidden[:, aux_prompt_len:aux_prompt_len + Lp], p.hidden)
        p.hidden = new_p
        p.run(self.l_rev, N_LAYERS)
        a.run(self.l_rev, N_LAYERS)
        return p.finish()[:, -1], a.finish()[:, -1]

    @torch.no_grad()
    def generate(self, builder, a_val, b_val, max_new=48):
        """Lockstep generation with the calculator tool. Greedy.

        Invariant: before step t, the primary stream holds tokens 0..t-1 and
        the auxiliary stream holds its prompt plus window tokens 0..t-1. At
        step t both models emit their position-t token from logits at t-1,
        with coupling between aligned positions (primary i <-> window i).
        """
        device = next(self.model.parameters()).device
        p_prompt = builder.prompt_ids(a_val, b_val)
        Pa = len(builder.aux_prompt_ids)

        p_list, a_list = [], list(builder.aux_prompt_ids)
        aux_generated = []          # (token_id, forced?) in the aligned window
        force_queue = []            # pending tool-result ids
        eos_id = self.tok.eos_token_id

        for t in range(len(p_prompt) + max_new):
            if t == 0:
                # phase 1 boundary: first window token from the aux prompt
                # alone, no coupling yet (the window is empty).
                out = self.model(torch.tensor([a_list], device=device))
                logits_a = out.logits[:, -1]
                logits_p = None
            else:
                p_ids = torch.tensor([p_list], device=device)
                a_ids = torch.tensor([a_list], device=device)
                logits_p, logits_a = self._step_logits(p_ids, a_ids, Pa)

            # auxiliary token for position t
            if force_queue:
                a_next, forced = force_queue.pop(0), True
            else:
                a_next, forced = int(logits_a.argmax(-1)), False
            a_list.append(a_next)
            aux_generated.append((a_next, forced))
            if not forced:
                text = self.tok.decode([tid for tid, _ in aux_generated])
                call = check_call(text)
                if call is not None:
                    force_queue = self.tok.encode(call, add_special_tokens=False)

            # primary token for position t
            if t < len(p_prompt):
                p_list.append(p_prompt[t])
            else:
                p_next = int(logits_p.argmax(-1))
                p_list.append(p_next)
                if p_next == eos_id:
                    break

        answer = self.tok.decode(p_list[len(p_prompt):], skip_special_tokens=True)
        aux_text = self.tok.decode([tid for tid, _ in aux_generated])
        return answer, aux_text


def _ce(logits, labels):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
