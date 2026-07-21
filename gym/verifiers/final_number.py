"""final_number — GSM8K-style: the last number in the response vs the gold's.

meta: {"gold": <str>}   gold text; its final number (after '####' if present) is the answer.

Score: 1.0 exact final-number match; 0.1 shaping credit for a well-formed
'#### <number>' final line; else 0.0. Ported verbatim from packs/gsm8k/scorer.py.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final(text: str) -> str | None:
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def verify(prompt: str, response: str, meta: dict) -> float:
    gold = extract_final(str(meta.get("gold", "")))
    pred = extract_final(str(response))
    if pred is not None and gold is not None and pred == gold:
        return 1.0
    well_formed = "####" in str(response) and pred is not None
    return 0.1 if well_formed else 0.0
