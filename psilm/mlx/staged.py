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


def recompute_in_backward(layer, mask):
    """One frozen layer, recomputed in the backward pass rather than taped.

    What mx.checkpoint was meant to do, with the recompute ordered after the
    cotangent. mx.checkpoint recomputes from a Depends node on the primal and
    the forward OUTPUT, so nothing makes a layer's recompute wait for the
    backward pass to reach it: the scheduler may run every window layer's
    recompute up front and hold all their tapes at once. On the GPU that is
    what happened -- the 9B constitution runs (l_rev 24, --checkpoint-from -1)
    peaked exactly as if fully taped (41.4 MiB per batch-token against the
    untaped physics run's 29.2, the difference being two more GatedDeltaNet
    tapes). Tying the recompute's input to the incoming cotangent with
    mx.depends forces the order, so one layer's tape is alive at a time. The
    gradients are the taped ones to the bit (eval/checkpoint_selftest.py).
    """
    def fn(h):
        return layer(h, mask=mask, cache=None)

    @mx.custom_function
    def f(h):
        return fn(h)

    @f.vjp
    def f_vjp(primals, cotangent, output):
        h = primals[0] if isinstance(primals, (tuple, list)) else primals
        c = cotangent[0] if isinstance(cotangent, (tuple, list)) else cotangent
        _, (g,) = mx.vjp(fn, [mx.depends(h, c)], [c])
        return (g,)

    return f


class MlxStream:
    #: layer index at and above which to recompute rather than tape (None: tape
    #: everything, the default). Only worth setting on backbones whose backward
    #: pass is memory-bound -- Qwen3.5, whose GatedDeltaNet layers fall back to a
    #: pure-ops scan that keeps its whole recurrence alive. Checkpointing is
    #: w.r.t. the layer's input, which is all that is needed here: the backbone
    #: is frozen, so the only gradient crossing a layer is the residual-stream
    #: one heading back to the bridges.
    checkpoint_from = None

    def __init__(self, model, input_ids, attn=None):
        self.model = model
        inner = model.model
        self.inner = inner
        self.hidden = inner.embed_tokens(input_ids)
        if attn is None:
            attn = mx.ones(input_ids.shape, dtype=mx.int32)
        self.mask = padded_causal_mask(attn, self.hidden.dtype)

    def run(self, lo: int, hi: int):
        cp = self.checkpoint_from
        for i, layer in enumerate(self.inner.layers[lo:hi], start=lo):
            if cp is None or i < cp:
                self.hidden = layer(self.hidden, mask=self.mask, cache=None)
            else:
                # the mask is captured as a constant; no gradient wanted
                self.hidden = recompute_in_backward(layer, self.mask)(self.hidden)
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
