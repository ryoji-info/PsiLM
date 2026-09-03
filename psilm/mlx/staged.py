"""Segmented forward passes through a frozen mlx-lm model.

Mirror of psilm/bicameral/staged.py for MLX: drive `model.model.layers`
manually so a forward pass can pause at any depth, exchange residual-stream
state with the bridges, and resume. The additive attention mask (causal +
right-padding) is built explicitly for full control.

Parity contract: a full staged pass with no coupling must reproduce the
stock model's logits (tested per backbone).
"""

import mlx.core as mx


def padded_causal_mask(attn, dtype):
    """attn: (B, L) 0/1 array -> additive mask (B, 1, L, L)."""
    B, L = attn.shape
    causal = mx.tril(mx.ones((L, L), dtype=mx.bool_))
    keys_ok = attn.astype(mx.bool_)[:, None, None, :]      # (B,1,1,L)
    ok = causal[None, None, :, :] & keys_ok
    neg = mx.array(-65504.0 if dtype == mx.float16 else -1e9, dtype=dtype)
    return mx.where(ok, mx.zeros((), dtype=dtype), neg)


class MlxStream:
    def __init__(self, model, input_ids, attn=None):
        self.model = model
        inner = model.model
        self.inner = inner
        self.hidden = inner.embed_tokens(input_ids)
        if attn is None:
            attn = mx.ones(input_ids.shape, dtype=mx.int32)
        self.mask = padded_causal_mask(attn, self.hidden.dtype)

    def run(self, lo: int, hi: int):
        for layer in self.inner.layers[lo:hi]:
            self.hidden = layer(self.hidden, mask=self.mask, cache=None)
        return self.hidden

    def finish(self):
        h = self.inner.norm(self.hidden)
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(h)
        else:
            logits = self.inner.embed_tokens.as_linear(h)
        # backbones with a final logit transform (Gemma: tanh soft-cap) expose it here
        post = getattr(self.model, "logit_postprocess", None)
        return post(logits) if post is not None else logits
