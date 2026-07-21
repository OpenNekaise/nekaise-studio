"""numeric_tolerance — hard-verifiable numeric comparison with explicit tolerance (R7).

meta: {"value": <number>,          gold value
       "atol": <float> = 0.0,      absolute tolerance
       "rtol": <float> = 0.0,      relative tolerance
       "which": "last"|"first"}    which number in the response to grade (default last —
                                   final-answer convention; "first" for cloze-style)

Correct iff |pred - value| <= atol + rtol*|value|. Binary score.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "").rstrip(".") for n in _NUM.findall(str(text))]


def verify(prompt: str, response: str, meta: dict) -> float:
    nums = _numbers(response)
    if not nums:
        return 0.0
    pick = nums[0] if meta.get("which") == "first" else nums[-1]
    try:
        pred, gold = float(pick), float(meta["value"])
    except (ValueError, KeyError, TypeError):
        return 0.0
    tol = float(meta.get("atol", 0.0)) + float(meta.get("rtol", 0.0)) * abs(gold)
    return 1.0 if abs(pred - gold) <= tol + 1e-12 else 0.0
