"""corpus_probes pack — THIN SHIM over gym (R1). The logic lives in gym/, once.

Legacy pack contract (load_split / is_correct / reward / extract_answer) preserved for
existing experiment recipes and lib/pack.py. Grading is gym.verifiers.numeric_cloze — the
same function the TRL reward wrapper and the eval runner import. Do not add logic here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym.tasks import corpus_probes as _t  # noqa: E402
from gym.verifiers import numeric_cloze as _v  # noqa: E402

in_split = _t.in_split


def load_split(split: str = "dev", n: int = 0) -> list[dict]:
    rows = [{"question": t.prompt,
             "answer": json.dumps({"value": t.meta["value"], "answer": t.ref_solution}),
             "id": t.id, "track": t.tags["track"], "topic": t.tags["topic"]}
            for t in _t.load(split=split)]
    return rows[:n] if n else rows


def extract_answer(text: str) -> str | None:
    return _v.first_number(text)


def is_correct(pred: str, gold: str) -> bool:
    return _v.verify("", pred, {"value": json.loads(gold)["value"]}) >= 0.999


def reward(pred: str, gold: str) -> float:
    return _v.verify("", pred, {"value": json.loads(gold)["value"]})
