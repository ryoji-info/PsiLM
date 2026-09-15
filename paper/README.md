# This directory holds the ΨLM paper

`psilm.tex` is the manuscript for the work described in it — a frozen language model
coupled to a frozen physics model through trainable latent bridges, Stage 0 through
the Qwen3.5 9B campaign. It is the living copy: **revise it here.**

```
latexmk -pdf psilm.tex
```

## The second paper

The constitution bridge — the same interface with a partner that holds a document
instead of computing a field, together with the dual stack that runs both partners on
one backbone — was drafted as a section of this manuscript and is now a paper of its
own, `paper/psilm2.tex` in the **PsiLM-2** repository. It cites this one as the
companion rather than restating it, and it is where the value-neuron write site, the
write-budget accounting, and the dual-channel results are reported.

Nothing in this manuscript needs to change for that: the constitution material was
never part of the version published here.
