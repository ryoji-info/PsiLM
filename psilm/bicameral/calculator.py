"""The auxiliary stream's calculator tool.

The auxiliary model emits text; when the tail of its generated text completes
a well-formed call `calc(A*B)`, the product is computed and `=RESULT;` is
force-fed back into the auxiliary stream, one token per lockstep step.
"""

import re

CALL_RE = re.compile(r"calc\((\d+)\*(\d+)\)$")


def check_call(generated_text: str):
    """Return '=RESULT;' if the text just completed a calc call, else None."""
    m = CALL_RE.search(generated_text)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return f"={a * b};"
