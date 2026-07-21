"""gym.tasks.sampler — difficulty bands + training-pool filtering (R6).

The benchmark's value is DISCRIMINATION, not difficulty: an all-hard pool gives a 1B
student no measurement range and GRPO no gradient (an all-wrong group has advantage 0).
Bands, from measured pass rate of the CURRENT student (gym/tools/calibrate.py):

    frontier      < 0.05   kept as the ceiling (no training signal today)
    hard_reserve  0.05–0.20
    trainable     0.20–0.80 ← the training sampler's window
    easy_reserve  0.80–0.95
    graduated     > 0.95   regression set (forgetting watch)

Calibration is a SIDECAR (gym/tasks/<set>/splits/calibration.json + one id-list file per
band — physical separation), never an in-place edit of task data. Re-calibrate as the
student improves; eval splits stay frozen + git-versioned regardless.
"""
from __future__ import annotations

import json
from pathlib import Path

from gym.tasks import Task

TASKS_DIR = Path(__file__).resolve().parent
BANDS = ("frontier", "hard_reserve", "trainable", "easy_reserve", "graduated")


def band(pass_rate: float, lo: float = 0.2, hi: float = 0.8) -> str:
    if pass_rate < 0.05:
        return "frontier"
    if pass_rate < lo:
        return "hard_reserve"
    if pass_rate <= hi:
        return "trainable"
    if pass_rate <= 0.95:
        return "easy_reserve"
    return "graduated"


def splits_dir(task_set: str, root: Path | None = None) -> Path:
    return (root or TASKS_DIR) / task_set / "splits"


def write_calibration(task_set: str, pass_rates: dict[str, float], meta: dict,
                      root: Path | None = None) -> Path:
    """Write the sidecar: calibration.json + one physical id-list file per band."""
    d = splits_dir(task_set, root)
    d.mkdir(parents=True, exist_ok=True)
    by_band: dict[str, list[str]] = {b: [] for b in BANDS}
    for tid, p in sorted(pass_rates.items()):
        by_band[band(p)].append(tid)
    (d / "calibration.json").write_text(json.dumps(
        {"meta": meta, "pass_rates": pass_rates,
         "bands": {b: len(ids) for b, ids in by_band.items()}}, indent=2))
    for b, ids in by_band.items():
        (d / f"{b}.json").write_text(json.dumps(ids, indent=2))
    return d


def load_calibration(task_set: str, root: Path | None = None) -> dict[str, float] | None:
    p = splits_dir(task_set, root) / "calibration.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["pass_rates"]


def filter_trainable(rows: list[Task], task_set: str, *, lo: float = 0.2, hi: float = 0.8,
                     missing: str = "keep", root: Path | None = None) -> list[Task]:
    """Training-pool filter by measured pass rate. Uncalibrated ids: keep (default) or drop.

    This filters the task POOL before training (data recipe); it is NOT the banned
    dynamic/difficulty sampling DURING training (SPEC §2).
    """
    calib = load_calibration(task_set, root)
    if calib is None:
        return list(rows) if missing == "keep" else []
    out = []
    for t in rows:
        p = calib.get(t.id)
        if p is None:
            if missing == "keep":
                out.append(t)
        elif band(p, lo, hi) == "trainable":
            out.append(t)
    return out
