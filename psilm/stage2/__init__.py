"""Stage 2: the PsiLM configuration proper.

A frozen LLM (language hemisphere) coupled to a frozen Fourier Neural
Operator (physics hemisphere) through trainable latent bridges in both
directions — no text at the interface:

  forward  (language -> physics): a readout over the LLM's layer-10 hidden
      states extracts the initial-condition parameters and constructs the
      operator's input field, differentiably;
  reverse  (physics -> language): learnable queries compress the operator's
      latent field into K soft tokens which every primary position
      cross-attends to at layer 15, through a suppression gate.
"""
