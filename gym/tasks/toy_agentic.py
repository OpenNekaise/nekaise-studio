"""toy_agentic task set — 5–15-step outcome-verified toy tasks for the agentic stage (R11).

Each task ships a small file WORKSPACE (in meta["workspace"]: relative path → content) and
a question whose answer requires reading those files (e.g. averaging a trend column). The
OUTCOME is verified — numeric_tolerance on the final answer — never the steps: no process
rewards, no dense supervision (SPEC card, agentic row). The episode protocol/rollout lives
in studio/stages/agentic (the harness side); gym only defines task + verifier.

Deterministic by construction (seeded); no wall-clock or RNG at import time.
"""
from __future__ import annotations

import random

from gym.tasks import Task


def _mk_trend(rng: random.Random, n_rows: int) -> tuple[str, float]:
    vals = [round(rng.uniform(15.0, 25.0), 2) for _ in range(n_rows)]
    lines = ["timestamp,supply_temp_c"]
    lines += [f"2026-01-01T{h:02d}:00:00,{v}" for h, v in enumerate(vals)]
    return "\n".join(lines) + "\n", round(sum(vals) / len(vals), 4)


def load(split: str = "dev", n: int = 0) -> list[Task]:
    rng = random.Random(3407 if split == "dev" else 7331)
    rows: list[Task] = []
    for i in range(n or 8):
        csv, avg = _mk_trend(rng, rng.randint(6, 18))
        readme = "Files: trend.csv holds hourly supply temperature readings.\n"
        rows.append(Task(
            id=f"toy-{split}-{i}", verifier="numeric_tolerance",
            prompt=("You are in a workspace with the files listed in README.txt. "
                    "What is the average supply_temp_c in trend.csv? "
                    "Reply with the number only, 4 decimals."),
            meta={"value": avg, "atol": 0.001, "which": "last",
                  "workspace": {"README.txt": readme, "trend.csv": csv}},
            ref_solution=f"{avg}", mode="chat",
            tags={"kind": "toy", "max_steps": 8}))
    return rows
