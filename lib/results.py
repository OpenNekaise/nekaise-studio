"""results — append-only measurement ledger per experiment. FIXED plumbing.

One JSONL row per completed measurement (an eval or a training run's final metric):
what was measured, on what data, and the score — machine-readable, so campaigns,
dashboards, and future sessions ACCUMULATE evidence instead of re-parsing logs.

    experiments/<exp>/results.jsonl      (git-ignored, like the rest of the run state)

LOG.md stays the narrative (why); the ledger is the record (what).
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def log_result(exp_dir, **fields) -> Path:
    """Append one measurement row (adds a timestamp). Returns the ledger path."""
    p = Path(exp_dir) / "results.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"t": time.time(), **fields}, ensure_ascii=False) + "\n")
    return p


def read_results(exp_dir) -> list[dict]:
    p = Path(exp_dir) / "results.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
