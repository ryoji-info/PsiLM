"""Stage 1: reproduction of the Bicameral Model (arXiv:2605.11167).

Two frozen Qwen2.5-0.5B-Instruct streams generate in lockstep, coupled through
a trainable neural interface on their intermediate hidden states: forward
coupling (primary -> auxiliary) at a lower layer, reverse coupling
(auxiliary -> primary) at a higher layer, each a translation network plus a
learned suppression gate that reads the receiver's state (the paper's "pull"
design). The auxiliary stream drives a calculator tool whose output is forced
back into the auxiliary stream only — the primary receives the result purely
through the hidden-state channel.
"""
