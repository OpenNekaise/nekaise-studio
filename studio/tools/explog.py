"""explog — schema'd experiment log (R8) + the adjudication rule (R5 + SPEC §3).

Every run writes ONE record, twice: machine-readable `log.jsonl` (fixed schema below) and
a human-readable line appended to `LOG.md`. History is not backfilled; the schema is
mandatory from the first post-refactor record.

    FIELDS   hypothesis / variable / expectation / result / noise_band / verdict / confidence
    verdicts keep | revert | invalid | pending | aborted-timebox | baseline

Adjudication (`decide`): KEEP only when the effect size exceeds the measured noise band
(studio/tools/variance_check.py) — a delta inside the band is noise, logged as revert.
Algorithm changes additionally need the null-hypothesis control (SPEC §3); that control is
its own record with variable="null-control: ...".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

FIELDS = ("hypothesis", "variable", "expectation", "result",
          "noise_band", "verdict", "confidence")
VERDICTS = {"keep", "revert", "invalid", "pending", "aborted-timebox", "baseline"}
CONFIDENCE = {"low", "medium", "high", "n/a"}


def append(exp_dir: str | Path, **rec) -> Path:
    """Validate against the schema and write both formats. Extra keys are kept in JSONL."""
    missing = [f for f in FIELDS if f not in rec]
    if missing:
        raise ValueError(f"explog record missing required fields: {missing}")
    if rec["verdict"] not in VERDICTS:
        raise ValueError(f"verdict {rec['verdict']!r} not in {sorted(VERDICTS)}")
    if rec["confidence"] not in CONFIDENCE:
        raise ValueError(f"confidence {rec['confidence']!r} not in {sorted(CONFIDENCE)}")
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    row = {"t": time.time(), **rec}
    with (exp_dir / "log.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    icon = {"keep": "✅", "revert": "❌", "invalid": "🚫", "baseline": "📌",
            "aborted-timebox": "⏰", "pending": "…"}[rec["verdict"]]
    band = rec.get("noise_band")
    with (exp_dir / "LOG.md").open("a") as f:
        f.write(f"| {time.strftime('%Y-%m-%d')} | {rec['variable']} | {rec['result']} "
                f"| band={band if band is not None else '—'} | {icon} {rec['verdict']} "
                f"| {rec['confidence']} |\n")
    return exp_dir / "log.jsonl"


def read(exp_dir: str | Path) -> list[dict]:
    p = Path(exp_dir) / "log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_noise_band(exp_dir: str | Path, metric: str) -> float | None:
    p = Path(exp_dir) / "noise_band.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    entry = data.get(metric)
    return entry["band"] if isinstance(entry, dict) else entry


def decide(value: float, best: float, noise_band: float | None) -> dict:
    """The keep/revert rule the loop MUST use (run-experiment skill references this).

    keep ⇔ (value − best) > noise_band. With no measured band the verdict degrades to
    'pending' — run variance_check first; deciding on noise accumulates false LOG entries.
    """
    delta = round(value - best, 6)
    if noise_band is None:
        return {"verdict": "pending", "delta": delta,
                "reason": "no noise band measured — run studio.tools.variance_check first"}
    if delta > noise_band:
        return {"verdict": "keep", "delta": delta,
                "reason": f"effect {delta:+.4f} > band {noise_band:.4f}"}
    return {"verdict": "revert", "delta": delta,
            "reason": f"effect {delta:+.4f} within band {noise_band:.4f} — noise"}


def hit_rate(exp_dir: str | Path) -> dict:
    """Aggregate hypothesis success stats from the JSONL (R8 done-when)."""
    rows = [r for r in read(exp_dir) if r.get("verdict") in ("keep", "revert")]
    kept = [r for r in rows if r["verdict"] == "keep"]
    return {"n_decided": len(rows), "n_kept": len(kept),
            "hit_rate": round(len(kept) / len(rows), 3) if rows else None}
