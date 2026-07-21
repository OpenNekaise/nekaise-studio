"""bench pack — THIN SHIM over gym (R1). The logic lives in gym/, once.

Split rule + version + grading all come from gym.tasks.bench / gym.verifiers.bench_qa
(which itself imports the EXTERNAL benchmark's own grading — never reimplemented).
Legacy contract + the extras tools/eval_bench.py uses (BENCH_DIR, bench_version,
system_prompt). Do not add logic here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym.tasks import bench as _t  # noqa: E402
from gym.verifiers import bench_qa as _v  # noqa: E402

BENCH_DIR = _v.bench_dir()
harness = _v.harness
bench_version = _t.version
in_split = _t.in_split


def system_prompt(track: str) -> str:
    h = _v.harness()
    return h.MCQ_SYSTEM if track == "mcq" else h.OPEN_SYSTEM


def load_split(split: str = "dev", n: int | None = None) -> list[dict]:
    rows = [{"question": t.prompt, "answer": json.dumps(t.meta["gold"], ensure_ascii=False),
             "id": t.id, "track": t.tags["track"], "topic": t.tags["topic"],
             "difficulty": t.tags["difficulty"]}
            for t in _t.load(split=split)]
    return rows[:n] if n else rows


def extract_answer(text: str) -> str:
    return _v.harness().strip_think(text).strip()


def reward(prediction: str, gold_answer: str) -> float:
    return _v.verify("", prediction, {"gold": gold_answer})


def is_correct(prediction: str, gold_answer: str) -> bool:
    return reward(prediction, gold_answer) >= 0.999


def load_test(n: int | None = None) -> list[dict]:
    return load_split("test", n)
