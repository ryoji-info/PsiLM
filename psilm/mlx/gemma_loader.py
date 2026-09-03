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

Loading. ``mlx-community/gemma-4-12B-it-4bit`` ships ``config.json`` with
``model_type = "gemma4_unified"`` (text_config ``gemma4_unified_text``) and
11 ``vision_embedder.*`` tensors next to the 1324 ``language_model.model.*``
ones. mlx_lm 0.31.3 has no ``gemma4_unified`` module (``mlx_lm.load`` raises
"Model type gemma4_unified not supported") and ``gemma4.Model.sanitize`` only
drops ``vision_tower/multi_modal_projector/audio_tower/embed_audio/embed_vision``,
so a strict ``load_weights`` would also choke on ``vision_embedder.*``. The
weight names and shapes of the language tower match ``gemma4_text`` exactly
(sliding layers: 16 x 256 q-heads, 8 kv heads; full layers: 16 x 512 q-heads,
1 kv head, k == v), so ``load_stock_gemma`` remaps the model type to
``gemma4`` and keeps only the ``language_model.*`` weights, then goes through
mlx_lm's own ``load_model`` (quantization predicate, strict load) and
``load_tokenizer`` (eos ids from generation_config: <eos>=1, <turn|>=106, 50).
"""

import importlib
from pathlib import Path

import mlx.core as mx
import mlx_lm

GEMMA4_MODEL_TYPES = {"gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"}
_MODEL_TYPE_REMAP = {"gemma4_unified": "gemma4", "gemma4_unified_text": "gemma4_text"}


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
    """MlxStream-shaped view of an mlx_lm gemma4 (or gemma4_text) model.

    Also quacks enough like the stock model for mlx_lm's generation path:
    ``__call__`` is the stock one-shot forward, ``layers`` / ``make_cache``
    let ``mlx_lm.models.cache.make_prompt_cache`` and ``generate_step`` run
    on the tower itself (the stock model stays reachable as ``_model``).
    """

    def __init__(self, model):
        self._model = model
        lm = getattr(model, "language_model", model)      # gemma4 wrapper or bare text model
        self._lm = lm
        inner = lm.model
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

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self._model.make_cache()

    def freeze(self):
        self._model.freeze()
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()


def _gemma_classes(config):
    """mlx_lm ``get_model_classes`` hook: gemma4/gemma4_text classes for the
    ``*_unified`` model types, with a sanitize that keeps only the language
    tower's weights (the unified checkpoint carries vision_embedder.* tensors
    the text model has no parameters for)."""
    mt = _MODEL_TYPE_REMAP.get(config["model_type"], config["model_type"])
    mod = importlib.import_module(f"mlx_lm.models.{mt}")
    base = mod.Model

    class Model(base):
        def sanitize(self, weights):
            weights = base.sanitize(self, weights)
            if hasattr(self, "language_model"):
                weights = {k: v for k, v in weights.items() if k.startswith("language_model.")}
            return weights

    Model.__name__, Model.__qualname__ = base.__name__, base.__qualname__
    return Model, mod.ModelArgs


def _snapshot(repo_id):
    from mlx_lm.utils import _download
    return Path(_download(repo_id))


def load_stock_gemma(repo_id="mlx-community/gemma-4-12B-it-4bit", lazy=False):
    """The stock mlx_lm gemma4 Model + TokenizerWrapper (eos ids {1, 106, 50}
    from generation_config) + merged config, for a gemma4 / gemma4_unified
    checkpoint. Equivalent to ``mlx_lm.load`` once the model type is one
    mlx_lm knows."""
    from mlx_lm.utils import load_config, load_model, load_tokenizer
    path = _snapshot(repo_id)
    cfg = load_config(path)
    mt = cfg.get("model_type")
    if mt not in GEMMA4_MODEL_TYPES:
        raise ValueError(f"{repo_id}: model_type {mt!r} is not a Gemma 4 checkpoint")
    model_config = {"model_type": _MODEL_TYPE_REMAP[mt]} if mt in _MODEL_TYPE_REMAP else None
    model, cfg = load_model(path, lazy=lazy, model_config=model_config,
                            get_model_classes=_gemma_classes)
    tok = load_tokenizer(path, eos_token_ids=cfg.get("eos_token_id", None))
    return model, tok, cfg


def load_gemma_tower(repo_id="mlx-community/gemma-4-12B-it-4bit"):
    model, tok, _ = load_stock_gemma(repo_id)
    return GemmaTower(model), tok


def load_backbone_any(repo_id):
    """(tower_for_MlxStream, stock_model_for_generate, mlx_tokenizer).

    Gemma 4 checkpoints go through ``load_stock_gemma`` and are wrapped in a
    GemmaTower; anything else is ``mlx_lm.load`` with the model returned in
    both slots. The tokenizer is mlx_lm's TokenizerWrapper, whose
    ``eos_token_ids`` come from the checkpoint's generation_config -- the one
    to hand to ``mlx_lm.generate`` (a bare HF tokenizer stops only on
    ``eos_token_id`` = <eos> 1, never on Gemma's turn end <turn|> 106).
    """
    from mlx_lm.utils import load_config
    cfg = load_config(_snapshot(repo_id))
    if cfg.get("model_type") in GEMMA4_MODEL_TYPES:
        stock, tok, _ = load_stock_gemma(repo_id)
        return GemmaTower(stock), stock, tok
    model, tok = mlx_lm.load(repo_id)
    return model, model, tok
