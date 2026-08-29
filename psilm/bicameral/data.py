"""Aligned dual-stream training data for the Bicameral reproduction.

Each example is a pair of token sequences that advance in lockstep:

  primary:  [chat prompt: "What is A * B?"] [response: "A * B equals R."]
  auxiliary:[aux prompt]                    [aligned window, same length as
                                            the whole primary sequence]

The auxiliary's aligned window is: wait tokens, then `calc(A*B)` (trained),
then `=R;` (tool-forced, masked from the loss), then wait tokens (trained).
Causality is enforced by construction: the calc call starts only after both
operands are visible in the primary stream, and the tool result completes
before the first digit of the answer appears in the primary response — if it
would not, the response preamble is extended with filler until it does.

Wait tokens and call/result spans are assembled as id lists (never by
re-tokenizing the concatenated string), so alignment is exact.
"""

import math
import random

import torch

SYSTEM = "You are a helpful assistant."
AUX_PROMPT = (
    "You are a silent calculator process. When a multiplication problem "
    "appears, output calc(a*b). Otherwise output '.' and wait."
)
FILLER = "Let me work it out. "
PROMPT_TAIL_SLACK = 5  # generation suffix tokens after the operands in the prompt


class ExampleBuilder:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.wait_id = tokenizer.encode(".", add_special_tokens=False)[0]
        # The prompt ends with literal wait tokens: the first window token is
        # predicted from the (uncoupled) last prompt position, so the frozen
        # model must already be inclined to continue with waits — otherwise
        # free-running generation derails before the coupling can act.
        self.aux_prompt_ids = (
            tokenizer.encode(AUX_PROMPT, add_special_tokens=False)
            + [self.wait_id] * 4
        )
        self.eos_id = tokenizer.eos_token_id

    def prompt_ids(self, a, b):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"What is {a} * {b}?"},
        ]
        out = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if not isinstance(out, list):  # transformers 5.x returns a BatchEncoding
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def build(self, a, b):
        res = a * b
        p_prompt = self.prompt_ids(a, b)
        call_ids = self.tok.encode(f"calc({a}*{b})", add_special_tokens=False)
        result_ids = self.tok.encode(f"={res};", add_special_tokens=False)

        # calc call starts once the operands are safely visible in the prompt
        call_start = len(p_prompt) - PROMPT_TAIL_SLACK
        result_end = call_start + len(call_ids) + len(result_ids)  # exclusive

        # response preamble must cover positions up to result_end
        filler = ""
        while True:
            preamble = f"{filler}{a} * {b} equals "
            pre_ids = self.tok.encode(preamble, add_special_tokens=False)
            first_digit_pos = len(p_prompt) + len(pre_ids)  # aligned index
            if first_digit_pos > result_end:
                break
            filler += FILLER
        response_text = f"{preamble}{res}."
        resp_ids = self.tok.encode(response_text, add_special_tokens=False) + [self.eos_id]

        p_ids = list(p_prompt) + resp_ids
        Lp = len(p_ids)
        p_labels = [-100] * len(p_prompt) + resp_ids

        # auxiliary aligned window (length Lp)
        window_ids, window_labels = [], []
        i = 0
        while i < Lp:
            if i < call_start:
                window_ids.append(self.wait_id); window_labels.append(self.wait_id)
                i += 1
            elif i < call_start + len(call_ids):
                k = i - call_start
                window_ids.append(call_ids[k]); window_labels.append(call_ids[k])
                i += 1
            elif i < result_end:
                k = i - call_start - len(call_ids)
                window_ids.append(result_ids[k]); window_labels.append(-100)  # forced
                i += 1
            else:
                window_ids.append(self.wait_id); window_labels.append(self.wait_id)
                i += 1

        a_ids = self.aux_prompt_ids + window_ids
        a_labels = [-100] * len(self.aux_prompt_ids) + window_labels
        return {
            "p_ids": p_ids, "p_labels": p_labels,
            "a_ids": a_ids, "a_labels": a_labels,
            "aux_prompt_len": len(self.aux_prompt_ids),
            "prompt_len": len(p_prompt),
            "meta": {"a": a, "b": b, "res": res},
        }


def sample_operands(rng, lo=2, hi=10**6):
    return (
        int(math.exp(rng.uniform(math.log(lo), math.log(hi)))),
        int(math.exp(rng.uniform(math.log(lo), math.log(hi)))),
    )


def make_batch(builder, rng, batch_size, device, lo=2, hi=10**6, pad_multiple=8):
    exs = [builder.build(*sample_operands(rng, lo, hi)) for _ in range(batch_size)]
    Lp = max(len(e["p_ids"]) for e in exs)
    Lp = ((Lp + pad_multiple - 1) // pad_multiple) * pad_multiple
    # the aux stream must cover aux_prompt + the full padded primary length,
    # so the coupled window slice [Pa : Pa+Lp] always exists
    La = exs[0]["aux_prompt_len"] + Lp
    pad = builder.tok.pad_token_id or builder.eos_id

    def pack(key, labels_key, L):
        ids = torch.full((batch_size, L), pad, dtype=torch.long)
        lab = torch.full((batch_size, L), -100, dtype=torch.long)
        attn = torch.zeros((batch_size, L), dtype=torch.long)
        for i, e in enumerate(exs):
            n = len(e[key])
            ids[i, :n] = torch.tensor(e[key])
            lab[i, :n] = torch.tensor(e[labels_key])
            attn[i, :n] = 1
        return ids.to(device), lab.to(device), attn.to(device)

    p_ids, p_labels, p_attn = pack("p_ids", "p_labels", Lp)
    a_ids, a_labels, a_attn = pack("a_ids", "a_labels", La)
    return {
        "p_ids": p_ids, "p_labels": p_labels, "p_attn": p_attn,
        "a_ids": a_ids, "a_labels": a_labels, "a_attn": a_attn,
        "aux_prompt_len": exs[0]["aux_prompt_len"],
        "metas": [e["meta"] for e in exs],
    }
