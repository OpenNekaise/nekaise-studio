"""bench task set — nekaise-bench, the EXTERNAL milestone referee. Never a loop metric.

Wraps the independently-authored benchmark; grading is the bench's own (gym.verifiers.
bench_qa imports its harness). The dev/test split rule lives HERE, canonically — NEVER
change it (recorded scores depend on it; tests pin a fingerprint):

    dev  (~75%)  advisory / milestone preview
    test (~25%)  FROZEN — milestone checks only; tools enforce --milestone + audit log

Scores are only comparable within one bench dataset version (version()).
"""
from __future__ import annotations

import hashlib
import json

from gym.tasks import Task
from gym.verifiers import bench_qa


def version() -> str:
    p = bench_qa.bench_dir() / "VERSION"
    return p.read_text().strip() if p.exists() else "unversioned"


def in_split(qid: str, split: str) -> bool:
    if split == "all":
        return True
    frozen = int(hashlib.md5(qid.encode()).hexdigest(), 16) % 4 == 0   # ~25% -> test
    return frozen if split == "test" else not frozen


def load(split: str = "dev", n: int = 0) -> list[Task]:
    h = bench_qa.harness()
    rows: list[Task] = []
    for line in (bench_qa.bench_dir() / "questions.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        q = json.loads(line)
        if not in_split(q["id"], split):
            continue
        if q["track"] == "mcq":
            prompt = h.mcq_prompt(q)
            gold = {"track": "mcq", "answer": q["answer"], "choices": q["choices"]}
            ref, system = str(q["answer"]), h.MCQ_SYSTEM
        else:
            prompt = q["question"]
            gold = {"track": "open", "answer": q["answer"], "aliases": q.get("aliases") or []}
            ref, system = str(q["answer"]), h.OPEN_SYSTEM
        rows.append(Task(id=q["id"], prompt=prompt, verifier="bench_qa", meta={"gold": gold},
                         ref_solution=ref, mode="chat", system=system,
                         tags={"track": q["track"], "topic": q.get("topic", ""),
                               "difficulty": q.get("difficulty", "")}))
    return rows[:n] if n else rows
