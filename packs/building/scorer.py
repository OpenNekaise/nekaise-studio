"""building pack — THIN SHIM over gym (R1). The logic lives in gym/, once.

Task minting + holdout enforcement live in gym.tasks.building (NEKAISE_HOLDOUT is now
REQUIRED — no silent fallback, R10); grading is gym.verifiers.ontology_qa. Legacy pack
contract preserved. Do not add logic here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym.tasks import building as _t  # noqa: E402
from gym.verifiers import ontology_qa as _v  # noqa: E402


def load_split(split: str = "test", n: int | None = 200) -> list[dict]:
    return [{"question": t.prompt, "answer": t.meta["gold"]}
            for t in _t.load(split=split, n=n or 0)]


def extract_answer(text: str) -> str:
    return text.strip()


def reward(prediction: str, gold_answer: str) -> float:
    return _v.verify("", prediction, {"gold": gold_answer})


def is_correct(prediction: str, gold_answer: str) -> bool:
    return reward(prediction, gold_answer) >= 0.999


def load_test(n: int | None = 200) -> list[dict]:
    return load_split("test", n)
