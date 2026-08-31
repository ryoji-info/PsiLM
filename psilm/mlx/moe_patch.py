"""Make MLX MoE blocks differentiable w.r.t. inputs (not routing indices).

mlx-lm's Qwen3 MoE block passes `argpartition` top-k indices straight into
SwitchGLU's gather. Autograd then tries to build a VJP with respect to those
indices and raises. The indices are discrete routing decisions and carry no
gradient by definition, so wrapping them in `stop_gradient` is both the fix
and the mathematically correct statement: gradients flow through the router
*scores* (which are differentiable softmax outputs) and through the expert
computations, but not through the argmax-like selection.

Call `patch_moe_gather()` once before building a coupled model.
"""

import mlx.core as mx


def patch_moe_gather():
    patched = []
    try:
        import mlx_lm.models.qwen3_moe as q
    except ImportError:
        return patched

    cls = getattr(q, "Qwen3MoeSparseMoeBlock", None)
    if cls is None or getattr(cls, "_psilm_patched", False):
        return patched

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(mx.stop_gradient(gates), kth=-k, axis=-1)[..., -k:]
        inds = mx.stop_gradient(inds)
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    cls.__call__ = __call__
    cls._psilm_patched = True
    patched.append("Qwen3MoeSparseMoeBlock")
    return patched


def router_logits(model, stream_hidden, layer_idx):
    """Router distribution at one layer — the MoE-specific diagnostic:
    does coupling change which experts the language model selects?"""
    blk = model.model.layers[layer_idx]
    gates = blk.mlp.gate(stream_hidden)
    return mx.softmax(gates, axis=-1, precise=True)
