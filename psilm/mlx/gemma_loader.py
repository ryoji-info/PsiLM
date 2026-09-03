"""Gemma 4 text tower in the MlxStream layout.

mlx_lm's ``gemma4`` Model nests the text model as ``language_model.model``
(a ``Gemma4TextModel``) and its layers return ``(h, shared_kv, offset)``;
the embeddings are scaled by sqrt(hidden) inside the model's forward, and the
tied head is followed by a tanh soft-cap. MlxStream drives
``model.model.layers[i](h, mask=..., cache=None)`` expecting an array, reads
``model.model.embed_tokens(ids)`` raw, and finishes with ``as_linear`` --
so this wrapper (1) unpacks the tuple, (2) scales the embedding on lookup
but not in ``as_linear``, and (3) exposes ``logit_postprocess`` for the
soft-cap. Sliding-window layers are driven with the same causal mask, which
is exact while the sequence is not longer than the window (1024 here).
"""

import mlx.core as mx
import mlx_lm


class _LayerShim:
    def __init__(self, layer, index, window=None):
        self.layer = layer
        self.index = index
        self.window = window        # sliding window for sliding layers, None for full

    def __call__(self, x, mask=None, cache=None):
        if self.window is not None and x.shape[1] > self.window:
            raise ValueError(f"layer {self.index}: sequence {x.shape[1]} exceeds the sliding "
                             f"window {self.window}; the causal mask is no longer exact")
        h, _, _ = self.layer(x, mask, cache)
        return h

    def __getattr__(self, name):
        if name == "layer":
            raise AttributeError(name)
        return getattr(self.layer, name)


class _ScaledEmbed:
    """embed_tokens view: scaled lookup, unscaled tied head."""

    def __init__(self, embed, scale):
        self._embed = embed
        self._scale = scale

    def __call__(self, ids):
        return self._embed(ids) * self._scale

    def as_linear(self, h):
        return self._embed.as_linear(h)

    def __getattr__(self, name):
        if name in ("_embed", "_scale"):
            raise AttributeError(name)
        return getattr(self._embed, name)


class _InnerView:
    def __init__(self, inner, shims):
        self._inner = inner
        self.layers = shims
        self.embed_tokens = _ScaledEmbed(inner.embed_tokens, inner.embed_scale)

    @property
    def norm(self):
        return self._inner.norm

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class GemmaTower:
    """MlxStream-shaped view of an mlx_lm gemma4 (or gemma4_text) model."""

    def __init__(self, model):
        self._model = model
        lm = getattr(model, "language_model", model)      # gemma4 wrapper or bare text model
        self._lm = lm
        inner = lm.model
        cfg = inner.config if hasattr(inner, "config") else lm.args
        self.args = lm.args
        window = getattr(self.args, "sliding_window", None)
        shims = []
        for i, layer in enumerate(inner.layers):
            is_sliding = getattr(layer, "layer_type", "full_attention") == "sliding_attention"
            shims.append(_LayerShim(layer, i, window if is_sliding else None))
        self.model = _InnerView(inner, shims)
        self.tie_word_embeddings = lm.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = lm.lm_head
        cap = lm.final_logit_softcapping
        self.logit_postprocess = (lambda x: mx.tanh(x / cap) * cap) if cap else None
        self.model_type = getattr(model, "model_type", "gemma4")

    def __call__(self, ids, **kw):          # one-shot reference forward = the stock model
        return self._model(ids, **kw)

    def freeze(self):
        self._model.freeze()
        return self

    def parameters(self):
        return self._model.parameters()


def load_gemma_tower(repo_id="mlx-community/gemma-4-12B-it-4bit"):
    model, tok = mlx_lm.load(repo_id)
    return GemmaTower(model), tok
