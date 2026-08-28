"""The two Stage-0 evaluation arms.

Arm 1 (alone): the LLM answers the multiple-choice physics question directly.
Arm 2 (tool loop): the LLM first extracts the scene parameters as JSON, the
simulator computes the outcome, and the numeric result is appended to the
prompt before the LLM answers — the Mind's Eye pattern, run locally.
"""

import json
import re

from .simulator import SCENES, render_result, simulate

ANSWER_SYSTEM = (
    "You are a careful physics assistant. Answer the multiple-choice question. "
    "Think briefly, then give your final answer as a single line exactly of the "
    "form 'Answer: A' or 'Answer: B' or 'Answer: C'."
)

EXTRACT_SYSTEM = (
    "You extract physical parameters from a question into JSON for a physics "
    "simulator. Reply with ONLY a JSON object, no other text."
)


def _options_text(item) -> str:
    return "\n".join(f"({k}) {v}" for k, v in item["options"].items())


def parse_choice(text: str):
    """Take the last A/B/C the model committed to."""
    m = re.findall(r"[Aa]nswer\s*:?\s*\(?([ABC])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"\(([ABC])\)", text)
    if m:
        return m[-1]
    m = re.findall(r"\b([ABC])\b", text)
    return m[-1] if m else None


def answer_alone(llm, item) -> dict:
    user = f"{item['question']}\n\n{_options_text(item)}"
    text = llm.chat(ANSWER_SYSTEM, user)
    return {"choice": parse_choice(text), "raw": text}


def _extract_params(llm, item, retries: int = 2):
    spec = SCENES[item["scene"]]
    fields = "\n".join(f'  "{k}": <{desc}>' for k, desc in spec["params"].items())
    user = (
        f"Question: {item['question']}\n\n"
        f"Fill in this JSON with the numbers from the question "
        f"(numbers only, no units):\n{{\n{fields}\n}}"
    )
    last_err = None
    for _ in range(retries + 1):
        text = llm.chat(EXTRACT_SYSTEM, user, max_tokens=250)
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            last_err = "no JSON object in output"
            continue
        try:
            params = json.loads(m.group(0))
            missing = [k for k in spec["params"] if k not in params]
            if missing:
                last_err = f"missing keys: {missing}"
                continue
            return {k: float(params[k]) for k in spec["params"]}, None
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            last_err = str(e)
    return None, last_err


def answer_with_tool(llm, item) -> dict:
    params, err = _extract_params(llm, item)
    if params is None:
        # Tool call failed: fall back to answering alone, and record it.
        out = answer_alone(llm, item)
        out.update({"tool_ok": False, "tool_error": err})
        return out
    try:
        result = simulate(item["scene"], params)
    except (KeyError, ValueError) as e:
        out = answer_alone(llm, item)
        out.update({"tool_ok": False, "tool_error": f"simulate: {e}"})
        return out
    evidence = render_result(item["scene"], result)
    user = (
        f"{item['question']}\n\n{_options_text(item)}\n\n"
        f"A trusted physics simulator was run on this problem.\n{evidence}\n"
        f"Compare the two numbers in the simulation result carefully. If they "
        f"differ by less than 5%, the answer is (C). Use the simulation result "
        f"to answer."
    )
    text = llm.chat(ANSWER_SYSTEM, user)
    return {
        "choice": parse_choice(text),
        "raw": text,
        "tool_ok": True,
        "extracted": params,
        "sim": {k: round(v, 4) for k, v in result.items()},
    }
