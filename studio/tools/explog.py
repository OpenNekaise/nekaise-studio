"""Experiment adjudication for the SQLite run store.

There is no Markdown journal in the agent-first design. Decisions are typed records keyed
by ``run_id`` and can be queried through ``python -m studio.cli show/list``.
"""
from __future__ import annotations

import json
from pathlib import Path

from studio.stages._common import REPO

FIELDS = ("hypothesis", "variable", "expectation", "result",
          "noise_band", "verdict", "confidence")
VERDICTS = {"keep", "revert", "invalid", "pending", "aborted-timebox", "baseline"}
CONFIDENCE = {"low", "medium", "high", "n/a"}


def append(exp_dir: str | Path, **record) -> None:
    """Compatibility API: persist a typed decision, never a text/JSONL side ledger."""
    missing = [field for field in FIELDS if field not in record]
    if missing:
        raise ValueError(f"explog record missing required fields: {missing}")
    if record["verdict"] not in VERDICTS:
        raise ValueError(f"verdict {record['verdict']!r} not in {sorted(VERDICTS)}")
    if record["confidence"] not in CONFIDENCE:
        raise ValueError(f"confidence {record['confidence']!r} not in {sorted(CONFIDENCE)}")
    run_id = record.pop("run_id", None)
    if not run_id:
        raise ValueError("agent-first decisions require run_id")
    from runstore import RunStore
    store = RunStore(REPO)
    metric = record.get("metric") or record.get("variable") or "unknown"
    store.record_decision(
        run_id, metric=metric, value=record.get("value"),
        verdict=record["verdict"], reason=record.get("reason") or record["result"],
        best_before=record.get("best_before"), noise_band=record["noise_band"],
        baseline_run_id=record.get("baseline_run_id"), metadata=record,
    )


def read(exp_dir: str | Path) -> list[dict]:
    from runstore import RunStore
    return RunStore(REPO).list_decisions(experiment=Path(exp_dir).name)


def load_noise_band(exp_dir: str | Path, metric: str) -> float | None:
    path = Path(exp_dir) / "noise_band.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    entry = data.get(metric)
    return entry["band"] if isinstance(entry, dict) else entry


def decide(value: float, best: float, noise_band: float | None) -> dict:
    delta = round(value - best, 6)
    if noise_band is None:
        return {"verdict": "pending", "delta": delta,
                "reason": "no noise band measured — run variance_check first"}
    if delta > noise_band:
        return {"verdict": "keep", "delta": delta,
                "reason": f"effect {delta:+.4f} > band {noise_band:.4f}"}
    return {"verdict": "revert", "delta": delta,
            "reason": f"effect {delta:+.4f} within band {noise_band:.4f} — noise"}


def hit_rate(exp_dir: str | Path) -> dict:
    rows = [row for row in read(exp_dir) if row["verdict"] in ("keep", "revert")]
    kept = [row for row in rows if row["verdict"] == "keep"]
    return {"n_decided": len(rows), "n_kept": len(kept),
            "hit_rate": round(len(kept) / len(rows), 3) if rows else None}
