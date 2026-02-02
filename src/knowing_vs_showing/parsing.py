from __future__ import annotations

import json
import re
from typing import Any


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_ANSWER_FIELD_RE = re.compile(r'(?i)"?\banswer\b"?\s*[:=]\s*("?)([12ab]|yes|no)\1')


def _coerce_answer_value(value: Any) -> int:
    if value is None:
        return -1

    if isinstance(value, int):
        if value == 1:
            return 0
        if value == 2:
            return 1
        return -1

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "a", "yes"}:
            return 0
        if v in {"2", "b", "no"}:
            return 1
        return -1

    return -1


def parse_binary_choice(text: str) -> int:
    """
    Parse a model output to a binary decision.

    Returns 0 for option 1, 1 for option 2, and -1 if parsing fails.
    """
    if not text:
        return -1

    t = str(text).strip()

    m = _JSON_OBJ_RE.search(t)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if str(k).strip().lower() == "answer":
                        parsed = _coerce_answer_value(v)
                        if parsed != -1:
                            return parsed
        except Exception:
            pass

    m = _ANSWER_FIELD_RE.search(t)
    if m:
        return _coerce_answer_value(m.group(2))

    first = t.split(maxsplit=1)[0].strip().strip('",.{}[]()')
    return _coerce_answer_value(first)
