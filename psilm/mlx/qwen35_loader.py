"""Qwen3.5 (hybrid linear-attention) text tower in the MlxStream layout.

Qwen3.5 is a vision-language checkpoint whose decoder alternates two kinds of
layer: three GatedDeltaNet blocks -- Mamba-style linear attention, carrying a
recurrent state rather than a KV cache -- then one full-attention block, set by
``full_attention_interval``. The 9B is 24 linear and 8 full over 32 layers.

Two things that costs us, both handled here rather than in MlxStream:

  * the two layer kinds want different masks. Full attention takes the additive
    causal mask MlxStream builds; GatedDeltaNet takes a boolean per-position
    padding mask and applies it as ``where(mask, qkv, 0)``. Feeding it the
    additive mask would zero almost everything, silently. The padding mask is
    recoverable from the additive one: its last row is causal-open at every
    position, so ``mask[:, 0, -1, :] == 0`` is exactly "this key is real".
  * mlx-lm's own forward passes ``create_ssm_mask``, which returns None without
    a cache -- so stock generation does not mask padding at all. That is fine
    at batch 1 and wrong for the right-padded batches PsiLM trains on, so the
    shim always supplies the mask. At batch 1 the two agree, which is what
    keeps the parity check meaningful.

The wrapper otherwise mirrors psilm/mlx/gemma_loader.py: ``.model.layers`` of
shims for the staged forward, ``.model.norm`` and ``.lm_head`` for the head,
and the stock model reachable as ``_model`` for mlx_lm.generate.
"""

import mlx.core as mx

QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}


class _LayerShim:
    """One decoder layer, given the mask its kind expects."""

    def __init__(self, layer, index):
        self.layer = layer
        self.index = index
        self.is_linear = bool(getattr(layer, "is_linear", False))

    def __call__(self, x, mask=None, cache=None):
        if self.is_linear and mask is not None and mask.ndim == 4:
            # additive (B,1,L,L) -> boolean (B,L): the last row is causal-open
            # everywhere, so it is the key-validity vector on its own
            mask = mask[:, 0, -1, :] == 0
        return self.layer(x, mask, cache)

    def __getattr__(self, name):
        if name == "layer":
            raise AttributeError(name)
        return getattr(self.layer, name)


class _InnerView:
    def __init__(self, inner, shims):
        self._inner = inner
        self.layers = shims

    @property
    def norm(self):
        return self._inner.norm

    @property
    def embed_tokens(self):
        return self._inner.embed_tokens

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class Qwen35Tower:
    """MlxStream-shaped view of an mlx_lm qwen3_5 model."""

    def __init__(self, model):
        self._model = model
        lm = getattr(model, "language_model", model)
        self._lm = lm
        inner = lm.model
        self.args = lm.args
        self.model = _InnerView(inner, [_LayerShim(l, i) for i, l in enumerate(inner.layers)])
        self.tie_word_embeddings = bool(getattr(lm.args, "tie_word_embeddings", False))
        if not self.tie_word_embeddings and hasattr(lm, "lm_head"):
            self.lm_head = lm.lm_head
        self.logit_postprocess = None
        # GatedDeltaNet runs a Metal kernel for the SSM scan, and that kernel has
        # no VJP: any gradient path crossing a linear-attention layer raises
        # "[Primitive::vjp] Not implemented for CustomKernel". mlx-lm already
        # carries the switch -- the layer calls gated_delta_update(...,
        # use_kernel=not self.training) -- so training mode selects a pure-MLX
        # scan that differentiates. It is the only use of self.training in
        # qwen3_5 and attention_dropout is 0.0, so the flag does nothing else.
        # The two paths agree to 7.2e-3 relative on this checkpoint, identical
        # argmax and top-5; train on the ops path, evaluate on the kernel.
        self.needs_train_mode_for_grad = True

    # -- enough of the stock model for mlx_lm's generation path --------------
    def __call__(self, *a, **kw):
        return self._model(*a, **kw)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self._lm.make_cache() if hasattr(self._lm, "make_cache") else self._model.make_cache()

    def set_grad_window(self, lo: int):
        """Differentiable SSM scan only from layer ``lo`` up.

        The backward pass reaches only the layers above the injection point, so
        the cheap Metal kernel can stay everywhere below it. That matters: the
        pure-ops scan keeps its whole recurrence on the tape, and running it in
        all 32 layers costs 25.2 GB at batch 2 against 12.9 GB when only the
        top twelve use it.
        """
        for i, shim in enumerate(self.model.layers):
            shim.layer.train(i >= lo)
        return self

    def freeze(self, *a, **kw):
        return self._model.freeze(*a, **kw)

    def parameters(self):
        return self._model.parameters()

    def __getattr__(self, name):
        if name in ("_model", "_lm"):
            raise AttributeError(name)
        return getattr(self._model, name)
