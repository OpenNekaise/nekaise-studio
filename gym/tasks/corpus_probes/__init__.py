"""Document-disjoint numeric cloze probes minted from nekaise-corpus.

The pure-CPT loop metric. Probes run in base-model COMPLETION mode (no chat template).
Splits are explicit and assigned at source-document level by the versioned builder.  No
source document may occur in both dev and frozen.

Tracks: transfer (held-out documents, macro-averaged across topics as the loop metric)
and absorption (train-eligible documents, diagnostic only).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from gym.tasks import Task

TASK_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _rows() -> tuple[dict, ...]:
    path = TASK_DIR / "probes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — mint once: python tools/build_probes.py")
    return tuple(json.loads(l) for l in path.read_text().splitlines() if l.strip())


def in_split(probe_id: str, split: str) -> bool:
    if split == "all":
        return True
    row = next((probe for probe in _rows() if probe["id"] == probe_id), None)
    if row is None:
        raise KeyError(f"unknown corpus probe: {probe_id}")
    return row["split"] == split


def load(split: str = "dev", n: int = 0) -> list[Task]:
    rows = [Task(id=p["id"], prompt=p["prompt"], verifier="numeric_cloze",
                 meta={"value": p["value"]}, ref_solution=str(p["value"]),
                 mode="complete",
                 tags={"track": p["kind"], "topic": p["topic"], "doc_id": p["doc_id"]})
            for p in _rows() if split == "all" or p["split"] == split]
    return rows[:n] if n else rows
