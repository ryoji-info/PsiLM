"""Stage 2d: PsiLM in 2D with a pretrained physics foundation model.

Frozen Qwen2.5-0.5B coupled to (briefly fine-tuned, then frozen) DPOT-Tiny —
a 7.5M-parameter AFNO operator pretrained across 12 PDE datasets — on 2D
Fisher-KPP reaction-diffusion QA. Same bridge principles as Stage 2, with
the 2b generalization law applied: every positional quantity (bump center,
query point) is read as a fully-covered 100-bin classification.
"""
