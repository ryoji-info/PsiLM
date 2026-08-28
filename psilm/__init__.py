"""PsiLM (ΨLM): a language model coupled with a physics model.

Stage 0: loop-level coupling — the LLM extracts physical parameters from a
question, a deterministic simulator computes the outcome, and the result is
fed back into the LLM's context before it answers.
"""

__version__ = "0.0.1"
