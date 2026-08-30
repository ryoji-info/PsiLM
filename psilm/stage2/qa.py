"""Stage-2 QA: field-value questions over Burgers flow.

Each item: initial condition u(x,0) = a sin(2 pi x + phi), evolved to t = 1
by the spectral solver (ground truth). The question asks for u at a point
x0; the answer is the value rounded to 2 decimals. Ground truth at off-grid
x0 comes from Fourier interpolation of the solver field.

Dataset generation samples trajectories once and asks several x0 questions
per trajectory. The LLM alone cannot answer these; the physics hemisphere
can — if the two bridges learn to carry the problem in and the field out.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

from ..physics.burgers import initial_condition, solve

SYSTEM = "You are a helpful assistant."

QUESTION = (
    "A velocity field on the periodic domain [0,1) starts as "
    "u(x,0) = {a} * sin(2*pi*x + {phi}). It evolves by Burgers' equation "
    "with viscosity 0.02 until t = 0.5. What is the value of u at x = {x0}? "
    "Answer with a number rounded to 2 decimal places."
)


def fourier_interp(field: np.ndarray, x0: float) -> float:
    """Evaluate the periodic grid field at arbitrary x0 via its Fourier series."""
    n = len(field)
    f_hat = np.fft.rfft(field) / n
    k = np.arange(len(f_hat))
    val = f_hat[0].real + 2 * np.sum(
        f_hat[1:].real * np.cos(2 * np.pi * k[1:] * x0)
        - f_hat[1:].imag * np.sin(2 * np.pi * k[1:] * x0)
    )
    return float(val)


def generate_dataset(n_traj: int, x0_per_traj: int, seed: int, out_path: str):
    rng = random.Random(seed)
    items = []
    for _ in range(n_traj):
        a = round(rng.uniform(0.5, 1.5), 2)
        phi = round(rng.uniform(0.0, 6.28), 2)
        field = solve(initial_condition(a, phi))
        for _ in range(x0_per_traj):
            x0 = round(rng.uniform(0.0, 0.99), 2)
            u = fourier_interp(field, x0)
            items.append({"a": a, "phi": phi, "x0": x0, "u": round(u, 4)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(items))
    return items


class QABuilder:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos_id = tokenizer.eos_token_id

    def prompt_ids(self, item):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION.format(
                a=item["a"], phi=item["phi"], x0=item["x0"])},
        ]
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False)
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def build(self, item):
        p_prompt = self.prompt_ids(item)
        answer = f"{item['u']:+.2f}".replace("+", "")
        resp = f"u at x = {item['x0']} equals {answer}."
        resp_ids = self.tok.encode(resp, add_special_tokens=False) + [self.eos_id]
        return {
            "p_ids": p_prompt + resp_ids,
            "p_labels": [-100] * len(p_prompt) + resp_ids,
            "prompt_len": len(p_prompt),
            "params": [item["a"], np.sin(item["phi"]), np.cos(item["phi"])],
            "x0": item["x0"],
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
    params = torch.tensor([e["params"] for e in exs], dtype=torch.float32)
    u_true = torch.tensor([e["meta"]["u"] for e in exs], dtype=torch.float32)
    x0 = torch.tensor([e["x0"] for e in exs], dtype=torch.float32)
    return {
        "p_ids": ids.to(device), "p_labels": lab.to(device),
        "p_attn": attn.to(device), "prompt_mask": pmask.to(device),
        "params": params.to(device), "u_true": u_true.to(device),
        "x0": x0.to(device),
        "param_mask": torch.ones_like(params).to(device),
        "metas": [e["meta"] for e in exs],
    }
