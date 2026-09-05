"""QA builder and batcher for the 2D Fisher-KPP task."""

import torch

from ..stage2.qa import SYSTEM

QUESTION = (
    "A concentration field on the periodic unit square starts as a Gaussian "
    "bump of height {a} centered at ({cx}, {cy}) with width {w}. It evolves "
    "by the Fisher-KPP reaction-diffusion equation (D = 0.001, r = 6) until "
    "t = 0.4. What is the value of u at the point ({x0}, {y0})? Answer with "
    "a number rounded to 2 decimal places."
)


# The six numeric quantities the template states, in prompt order, and the
# text that immediately precedes each one in QUESTION. The MLX readout pools
# deterministically over each quantity's token span (psilm/mlx/bridges2d.py);
# the prefix anchors the sublist match so a value that also appears as a
# constant of the template ("t = 0.4", "D = 0.001") or as another quantity
# (x0 == cx happens) cannot be matched at the wrong position.
QUANTITIES = ("a", "cx", "cy", "w", "x0", "y0")
_SPAN_PREFIX = {"a": " height ", "cx": " (", "cy": ", ", "w": " width ", "x0": " (", "y0": ", "}


class QA2DBuilder:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos_id = tokenizer.eos_token_id

    def spans(self, p_prompt, item):
        """Token span (lo, hi) of every quantity in QUANTITIES, in prompt order,
        widened by one token each side (mirror of QABuilder.x0_span in
        psilm/stage2/qa.py). Each value is located by matching the token
        sublist of prefix+value at or after the previous quantity's match; the
        span covers the value tokens only (before widening). Raises if the
        tokenizer merges across the prefix/value boundary or a match fails --
        silent fallback would pool the whole prompt."""
        enc = lambda t: self.tok.encode(t, add_special_tokens=False)
        out, cursor = [], 0
        for name in QUANTITIES:
            pre, val = _SPAN_PREFIX[name], str(item[name])
            pre_ids, pv_ids = enc(pre), enc(pre + val)
            if pv_ids[:len(pre_ids)] != pre_ids or len(pv_ids) == len(pre_ids):
                raise ValueError(f"span of {name}={val!r}: tokenizer merges across the "
                                 f"prefix boundary ({pre!r}); the value span is ill-defined")
            hit = -1
            for i in range(cursor, len(p_prompt) - len(pv_ids) + 1):
                if p_prompt[i:i + len(pv_ids)] == pv_ids:
                    hit = i
                    break
            if hit < 0:
                raise ValueError(f"span of {name}={val!r} not found in the prompt after token "
                                 f"{cursor}; deterministic span pooling would silently pool "
                                 "the whole prompt")
            lo, hi = hit + len(pre_ids), hit + len(pv_ids)
            out.append((max(0, lo - 1), min(len(p_prompt), hi + 1)))
            cursor = hi
        return out

    def prompt_ids(self, item):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION.format(**{
                k: item[k] for k in ("a", "cx", "cy", "w", "x0", "y0")})},
        ]
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def build(self, item):
        p_prompt = self.prompt_ids(item)
        resp = f"u at ({item['x0']}, {item['y0']}) equals {item['u']:.2f}."
        resp_ids = self.tok.encode(resp, add_special_tokens=False) + [self.eos_id]
        return {
            "p_ids": p_prompt + resp_ids,
            "p_labels": [-100] * len(p_prompt) + resp_ids,
            "prompt_len": len(p_prompt),
            "spans": self.spans(p_prompt, item),
            "meta": item,
        }


def make_batch(builder, items, device, pad_multiple=8):
    exs = [builder.build(it) for it in items]
    L = max(len(e["p_ids"]) for e in exs)
    L = ((L + pad_multiple - 1) // pad_multiple) * pad_multiple
    pad = builder.tok.pad_token_id or builder.eos_id
    B = len(exs)
    ids = torch.full((B, L), pad, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    pmask = torch.zeros((B, L), dtype=torch.bool)
    for i, e in enumerate(exs):
        n = len(e["p_ids"])
        ids[i, :n] = torch.tensor(e["p_ids"])
        lab[i, :n] = torch.tensor(e["p_labels"])
        attn[i, :n] = 1
        pmask[i, :e["prompt_len"]] = True
    reg = torch.tensor([[e["meta"]["a"], e["meta"]["w"]] for e in exs], dtype=torch.float32)
    bins = torch.tensor(
        [[round(e["meta"][k] * 100) for k in ("cx", "cy", "x0", "y0")] for e in exs],
        dtype=torch.long).clamp(0, 99)
    # all six quantities as 100-bin targets on the 0.01 grid (the MLX readout
    # classifies every quantity; positions are periodic, a and w are not)
    qbins = torch.tensor(
        [[round(e["meta"][k] * 100) for k in QUANTITIES] for e in exs],
        dtype=torch.long).clamp(0, 99)
    return {
        "p_ids": ids.to(device), "p_labels": lab.to(device),
        "p_attn": attn.to(device), "prompt_mask": pmask.to(device),
        "reg": reg.to(device), "bins": bins.to(device),
        "u_true": torch.tensor([e["meta"]["u"] for e in exs], dtype=torch.float32).to(device),
        "spans": torch.tensor([e["spans"] for e in exs], dtype=torch.long).to(device),
        "qbins": qbins.to(device),
        "metas": [e["meta"] for e in exs],
    }
