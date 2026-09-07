"""Multi-mode Burgers QA with held-out IC families (Stage 2b/2c).

u(x,0) = sum over active modes m of a_m sin(2 pi m x + phi_m), m in {1,2,3}.

Modes are restricted to {1, 2}: at nu=0.02, t=0.5 the mode-3 component decays
by e^-3.55 ~ 0.03 and its questions degenerate to "about zero". Mode-2
amplitudes are sampled higher to offset its faster decay.

Families:
  train     — single-mode questions only: {1} with a~U(0.3,0.7), or {2} with
              a~U(0.5,1.0)
  val_iid   — same distribution, fresh seed
  val_combo — the held-out combination {1,2}: two-term questions the bridges
              never saw (compositional generalization)
  val_amp   — mode 1 with a~U(0.8,1.0): amplitude extrapolation

The physics FNO is trained separately on a broader distribution than all of
these, so family-transfer failures are attributable to the bridges.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

from ..physics.burgers import initial_condition_multi, solve
from .qa import SYSTEM, fourier_interp

N_MODES = 2
AMP = {1: (0.3, 0.7), 2: (0.5, 1.0)}

QUESTION = (
    "A velocity field on the periodic domain [0,1) starts as u(x,0) = {ic}. "
    "It evolves by Burgers' equation with viscosity 0.02 until t = 0.5. "
    "What is the value of u at x = {x0}? Answer with a number rounded to "
    "2 decimal places."
)

TRAIN_COMBOS = [(1,), (2,)]


def ic_text(modes):
    terms = []
    for m, a, phi in modes:
        arg = f"2*pi*x + {phi}" if m == 1 else f"{2 * m}*pi*x + {phi}"
        terms.append(f"{a}*sin({arg})")
    return " + ".join(terms)


def _sample_modes(rng, family):
    if family == "val_combo":
        combo = (1, 2)
        ranges = {m: AMP[m] for m in combo}
    elif family == "val_amp":
        combo = (1,)
        ranges = {1: (0.8, 1.0)}
    else:  # train / val_iid
        combo = TRAIN_COMBOS[rng.randrange(len(TRAIN_COMBOS))]
        ranges = {m: AMP[m] for m in combo}
    return [(m, round(rng.uniform(*ranges[m]), 2), round(rng.uniform(0.0, 6.28), 2))
            for m in combo]


def generate_dataset(n_traj, x0_per_traj, seed, family, out_path):
    rng = random.Random(seed)
    items = []
    for _ in range(n_traj):
        modes = _sample_modes(rng, family)
        field = solve(initial_condition_multi(modes))
        for _ in range(x0_per_traj):
            x0 = round(rng.uniform(0.0, 0.99), 2)
            items.append({"modes": modes, "x0": x0,
                          "u": round(fourier_interp(field, x0), 4)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(items))
    return items



def _span_of(tok, p_prompt, prefixes, value, cursor, name):
    """(lo, hi) of `value`'s tokens, located by the first candidate prefix that
    both tokenizes cleanly (prefix tokens are a prefix of prefix+value tokens)
    and matches the prompt at or after `cursor`.

    Several spellings are tried because tokenizers bind the surrounding
    punctuation differently -- Gemma reads ' =' as one token, so the prefix has
    to carry the leading space. Raises when none matches: a silent fallback
    would pool the whole prompt."""
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    for prefix in prefixes:
        pre_ids, pv_ids = enc(prefix), enc(prefix + value)
        if pv_ids[:len(pre_ids)] != pre_ids or len(pv_ids) == len(pre_ids):
            continue
        for i in range(cursor, len(p_prompt) - len(pv_ids) + 1):
            if p_prompt[i:i + len(pv_ids)] == pv_ids:
                return i + len(pre_ids), i + len(pv_ids)
    raise ValueError(f"span of {name}={value!r} not found after token {cursor} with any of "
                     f"{prefixes}; deterministic span pooling would pool the whole prompt")


def with_second_term(item, rng, amp_max=0.25):
    """The same question written with the absent mode present at a small
    amplitude drawn from U(0.02, amp_max), the answer recomputed by the solver.

    Structure coverage: a readout trained only on one-term prompts has no
    constraint on what it does with two, and its held-out combination accuracy
    lands wherever the seed puts it (0.30-0.98 over the probes in
    results/readout_transfer/). Adding two-term prompts fixes the structure
    while leaving the combination family's amplitudes (0.3-0.7 with 0.5-1.0)
    outside the second slot's training support, which the mode-shared heads
    cover instead. A distractor pinned at exactly 0.0 does NOT work: it teaches
    that a second term contributes nothing (mean 0.681 against 0.608 without).
    """
    present = {m for m, _, _ in item["modes"]}
    missing = [m for m in range(1, N_MODES + 1) if m not in present]
    if not missing:
        return item
    a = round(rng.uniform(0.02, amp_max), 2)
    modes = sorted([list(m) for m in item["modes"]]
                   + [[missing[0], a, round(rng.uniform(0.0, 6.28), 2)]],
                   key=lambda t: t[0])
    field = solve(initial_condition_multi([tuple(m) for m in modes]))
    return {**item, "modes": modes, "u": round(float(fourier_interp(field, item["x0"])), 4)}


class QA2Builder:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos_id = tokenizer.eos_token_id

    def prompt_ids(self, item):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION.format(
                ic=ic_text(item["modes"]), x0=item["x0"])},
        ]
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False)
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def x0_span(self, p_prompt, item):
        """Token span of x0 in the prompt (mirror of QABuilder.x0_span): x0 is
        the last number mentioned, so the last sublist match is unambiguous.
        Widened by 1 for tokenizer boundary effects. Raises if no match, so
        deterministic span pooling can never silently pool the whole prompt."""
        for cand in (f" {item['x0']}", f"{item['x0']}"):
            sub = self.tok.encode(cand, add_special_tokens=False)
            hit = -1
            for i in range(len(p_prompt) - len(sub) + 1):
                if p_prompt[i:i + len(sub)] == sub:
                    hit = i
            if hit >= 0:
                return max(0, hit - 1), min(len(p_prompt), hit + len(sub) + 1)
        raise ValueError(f"x0 span not found in prompt (x0={item.get('x0')!r}); "
                         "deterministic span pooling would silently pool the whole prompt")


    # Slot order of the span readout: (a, phi) per mode, then x0. Absent modes
    # keep a (0, 0) span and a 0 in the mask; the trainer supervises their
    # amplitude to zero without pooling anything.
    SLOTS = tuple(f"{q}{m}" for m in range(1, N_MODES + 1) for q in ("a", "phi")) + ("x0",)

    def spans(self, p_prompt, item):
        """Token span (lo, hi) of every number in the prompt, in SLOTS order,
        widened by one token each side (mirror of QA2DBuilder.spans).

        Each value is located by matching the token sublist of prefix+value at
        or after the previous match, so repeated prefixes ('x + ' occurs once
        per mode) stay unambiguous. Returns (spans, present) where present[i]
        is 1.0 for a slot whose mode occurs in this item. Raises if a match
        fails or the tokenizer merges across a prefix boundary -- a silent
        fallback would pool the whole prompt."""
        enc = lambda t: self.tok.encode(t, add_special_tokens=False)
        modes = sorted(item["modes"], key=lambda t: t[0])
        spans = [(0, 0)] * len(self.SLOTS)
        present = [0.0] * len(self.SLOTS)
        cursor = 0
        for k, (m, a, phi) in enumerate(modes):
            first = (k == 0)
            for name, prefixes, value in (
                    (f"a{m}", (" = ", "= ") if first else (" + ", "+ "), str(a)),
                    (f"phi{m}", ("*x + ", "x + ", " + "), str(phi))):
                lo, hi = _span_of(self.tok, p_prompt, prefixes, value, cursor, name)
                i = self.SLOTS.index(name)
                spans[i] = (max(0, lo - 1), min(len(p_prompt), hi + 1))
                present[i] = 1.0
                cursor = hi
        spans[-1] = self.x0_span(p_prompt, item)
        present[-1] = 1.0
        return spans, present

    def build(self, item):
        p_prompt = self.prompt_ids(item)
        answer = f"{item['u']:+.2f}".replace("+", "")
        resp = f"u at x = {item['x0']} equals {answer}."
        resp_ids = self.tok.encode(resp, add_special_tokens=False) + [self.eos_id]
        params = [0.0] * (3 * N_MODES)
        mask = [0.0] * (3 * N_MODES)
        for m, a, phi in item["modes"]:
            params[3 * m - 3: 3 * m] = [a, np.sin(phi), np.cos(phi)]
            mask[3 * m - 3: 3 * m] = [1.0, 1.0, 1.0]
        for m in range(1, N_MODES + 1):          # absent modes: supervise a -> 0
            if mask[3 * m - 3] == 0.0:
                mask[3 * m - 3] = 1.0
        return {
            "p_ids": p_prompt + resp_ids,
            "p_labels": [-100] * len(p_prompt) + resp_ids,
            "prompt_len": len(p_prompt),
            "params": params, "param_mask": mask,
            "x0": item["x0"],
            "x0_span": self.x0_span(p_prompt, item),
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
    return {
        "p_ids": ids.to(device), "p_labels": lab.to(device),
        "p_attn": attn.to(device), "prompt_mask": pmask.to(device),
        "params": torch.tensor([e["params"] for e in exs], dtype=torch.float32).to(device),
        "param_mask": torch.tensor([e["param_mask"] for e in exs], dtype=torch.float32).to(device),
        "x0": torch.tensor([e["x0"] for e in exs], dtype=torch.float32).to(device),
        "amp_bins": torch.tensor(
            [[round(e["params"][3 * m] * 100) for m in range(N_MODES)] for e in exs],
            dtype=torch.long).to(device),
        "u_true": torch.tensor([e["meta"]["u"] for e in exs], dtype=torch.float32).to(device),
        "x0_span": torch.tensor([e["x0_span"] for e in exs], dtype=torch.long).to(device),
        "metas": [e["meta"] for e in exs],
    }
