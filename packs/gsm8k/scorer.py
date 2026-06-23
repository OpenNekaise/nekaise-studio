"""GSM8K scorer — the FIXED fitness function / referee. Do NOT edit.

This is the pack's contract. Every method (SFT, RFT, DPO, GRPO, distillation)
uses the SAME referee, so results stay comparable and can't be gamed:

  load_split(split, n) -> list[{"question", "answer"}]   # data + gold
  is_correct(pred, gold) -> bool                         # eval metric / RFT filter
  reward(pred, gold) -> float                            # graded signal for RL (GRPO)
  extract_answer(text) -> str | None                     # shared helper

GSM8K gold answers end with '#### <number>'; predictions are parsed for their last
number. Editing this file would cheat the metric (see skills/run-experiment.md).
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
    """True iff the prediction's final number matches the gold answer's.

    Used both as the eval metric and as the rejection-sampling filter
    (keep a generated solution only if is_correct is True).
    """
    pred = extract_answer(prediction)
    gold = extract_answer(gold_answer)
    return pred is not None and pred == gold


def reward(prediction: str, gold_answer: str) -> float:
    """Graded reward in [0, 1] for RL (GRPO/PPO).

    Dense enough to give the optimizer a gradient before the model is fully
    correct: full credit for the right answer, a small shaping credit for at
    least emitting a well-formed '#### <number>' final answer.
    """
    if is_correct(prediction, gold_answer):
        return 1.0
    well_formed = "####" in prediction and extract_answer(prediction) is not None
    return 0.1 if well_formed else 0.0


def load_split(split: str = "test", n: int | None = 200) -> list[dict]:
    """Load a GSM8K split (first `n`, or all if None).

    Rows follow the pack's schema: each is a dict with at least
    `question` (the task input) and `answer` (the gold, incl. '#### <num>').
    `split` is "train" or "test".
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return list(ds)


def load_test(n: int | None = 200) -> list[dict]:
    """Back-compat alias for load_split('test', n)."""
    return load_split("test", n)
