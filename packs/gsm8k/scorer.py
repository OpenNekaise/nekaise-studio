"""GSM8K scorer — the FIXED fitness function. Do NOT edit (see skills/run-experiment.md).

Defines the metric the autoresearch loop optimizes: exact match on the final numeric
answer. GSM8K gold answers end with '#### <number>'; predictions are parsed for their
last number.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Return the final numeric answer in `text`, normalized, or None."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def is_correct(prediction: str, gold_answer: str) -> bool:
    """True iff the prediction's final number matches the gold answer's."""
    pred = extract_answer(prediction)
    gold = extract_answer(gold_answer)
    return pred is not None and pred == gold


def load_test(n: int | None = 200) -> list[dict]:
    """Load the GSM8K test split (first `n`, or all if None)."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return list(ds)
