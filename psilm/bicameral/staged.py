"""Staged forward passes through a frozen Qwen2 model.

The Bicameral coupling needs to pause a forward pass at an arbitrary layer,
exchange residual-stream state with the other stream, and resume. This module
drives `model.model.layers` manually in segments, matching the internals of
transformers' Qwen2Model.forward (verified against transformers 5.16).

Parity is enforced by tests: a full staged pass with no coupling must
reproduce the stock model's logits.
"""

import torch
from transformers.masking_utils import create_causal_mask


class StreamState:
    """Everything one segmented forward pass needs, built once per call."""

    def __init__(self, model, input_ids, attention_mask=None, past_key_values=None,
                 use_cache=False):
        base = model.model
        self.model = model
        self.use_cache = use_cache
        self.cache = past_key_values
        self.hidden = base.embed_tokens(input_ids)
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        self.position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device) + past
        ).unsqueeze(0)
        self.mask = create_causal_mask(
            config=base.config,
            inputs_embeds=self.hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=self.position_ids,
        )
        self.pos_emb = base.rotary_emb(self.hidden, self.position_ids)

    def run(self, lo: int, hi: int):
        """Run decoder layers [lo, hi) on the current hidden state."""
        for layer in self.model.model.layers[lo:hi]:
            self.hidden = layer(
                self.hidden,
                attention_mask=self.mask,
                position_embeddings=self.pos_emb,
                position_ids=self.position_ids,
                past_key_values=self.cache,
                use_cache=self.use_cache,
            )
        return self.hidden

    def finish(self):
        """Final norm + LM head -> logits."""
        h = self.model.model.norm(self.hidden)
        return self.model.lm_head(h)
