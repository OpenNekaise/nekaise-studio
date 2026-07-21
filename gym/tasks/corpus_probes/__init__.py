"""corpus_probes task set — numeric cloze probes minted from the cleaned corpus.

The pure-CPT loop metric. Probes run in base-model COMPLETION mode (no chat template).
Splits (id-hashed, deterministic — NEVER change the rule, tests pin a fingerprint):

    dev    (~80%)  the loop's keep/revert signal
    frozen (~20%)  milestone confirmation only

Tracks: absorption (from CPT train docs — the loop metric) and transfer (held-out docs —
generalization diagnostic). Filter by tags["track"] downstream.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gym.tasks import Task

TASK_DIR = Path(__file__).resolve().parent


def _rows() -> list[dict]:
    path = TASK_DIR / "probes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — mint once: python tools/build_probes.py")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def in_split(probe_id: str, split: str) -> bool:
    frozen = int(hashlib.md5(probe_id.encode()).hexdigest(), 16) % 5 == 0
    return frozen if split == "frozen" else (not frozen) if split == "dev" else True


def load(split: str = "dev", n: int = 0) -> list[Task]:
    rows = [Task(id=p["id"], prompt=p["prompt"], verifier="numeric_cloze",
                 meta={"value": p["value"]}, ref_solution=str(p["value"]),
                 mode="complete", tags={"track": p["kind"], "topic": p["topic"]})
            for p in _rows() if in_split(p["id"], split)]
    return rows[:n] if n else rows
