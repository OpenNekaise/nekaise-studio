"""GSM8K pack — THIN SHIM over gym (R1). The logic lives in gym/, once.

Legacy pack contract preserved; grading is gym.verifiers.final_number — the same function
training rewards, RFT filters, and eval import. Do not add logic here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym.verifiers import final_number as _v  # noqa: E402

extract_answer = _v.extract_final


def is_correct(prediction: str, gold_answer: str) -> bool:
    return _v.verify("", prediction, {"gold": gold_answer}) >= 0.999


def reward(prediction: str, gold_answer: str) -> float:
    return _v.verify("", prediction, {"gold": gold_answer})


def load_split(split: str = "test", n: int | None = 200) -> list[dict]:
    from gym.tasks import gsm8k as _t
    return [{"question": t.prompt, "answer": t.meta["gold"]}
            for t in _t.load(split=split, n=n or 0)]


def load_test(n: int | None = 200) -> list[dict]:
    return load_split("test", n)
