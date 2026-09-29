"""Read-only, aggregate-only external benchmark surface. Never a teaching input."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ROOT

MMLU_DATASET = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"


CATEGORY_COUNTS = {'biology':717, 'business':789, 'chemistry':1132, 'computer science':410,
    'economics':844, 'engineering':969, 'health':818, 'history':381, 'law':1101,
    'math':1351, 'other':924, 'philosophy':499, 'physics':1299, 'psychology':798}


class Subject(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True, allow_inf_nan=False)
    category: str = Field(max_length=30)
    n: int = Field(ge=1, le=12032)
    correct: int = Field(ge=0, le=12032)
    score: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def coherent(self):
        if (CATEGORY_COUNTS.get(self.category) != self.n or self.correct > self.n or
                abs(self.score - self.correct/self.n) > 1e-9):
            raise ValueError('Inconsistent subject result')
        return self


class Aggregate(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    schema_version: Literal[1]
    benchmark: Literal["mmlu-pro"]
    run_id: str = Field(pattern=r"^[0-9]{8}T[0-9]{12}Z-[a-f0-9]{8}$")
    status: Literal["running", "complete", "failed", "interrupted"]
    started_at: str = Field(max_length=40)
    updated_at: str = Field(max_length=40)
    model_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_label: str = Field(max_length=160)
    round_id: str | None = Field(default=None, max_length=100)
    round_number: int | None = Field(default=None, ge=0)
    campaign_id: str | None = Field(default=None, max_length=100)
    protocol: Literal["mmlu-pro-choice-1"]
    protocol_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    dataset_sha256: Literal[MMLU_DATASET]
    max_input_tokens: int = Field(ge=1, le=32768)
    n: Literal[12032]
    completed: int = Field(ge=0, le=12032)
    correct: int | None = Field(default=None, ge=0, le=12032)
    score: float | None = Field(default=None, ge=0, le=1)
    ci95: list[float] | None = Field(default=None, min_length=2, max_length=2)
    chance_score: float | None = Field(default=None, ge=0, le=1)
    subjects: list[Subject] | None = Field(default=None, min_length=14, max_length=14)
    seal_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def coherent(self):
        start, end = (datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (self.started_at, self.updated_at))
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("Invalid evaluation timestamps")
        scores = (self.correct, self.score, self.ci95, self.chance_score, self.subjects, self.seal_sha256)
        if self.status != "complete":
            if any(v is not None for v in scores):
                raise ValueError("Incomplete run has a provisional score")
        else:
            if (any(v is None for v in scores) or not self.model_id or not self.protocol_id or
                    self.completed != self.n or abs(self.score - self.correct / self.n) > 1e-9):
                raise ValueError("Incomplete or inconsistent MMLU-Pro result")
            if (not 0 <= self.ci95[0] <= self.ci95[1] <= 1 or
                    self.ci95[0] > self.score + 1e-12 or self.ci95[1] < self.score - 1e-12):
                raise ValueError("Inconsistent result diagnostics")
            if (len({row.category for row in self.subjects}) != 14 or
                    sum(row.n for row in self.subjects) != self.n or
                    sum(row.correct for row in self.subjects) != self.correct or
                    not .1 <= self.chance_score <= 1/3):
                raise ValueError('Inconsistent subject aggregates')
        return self


class Catalog(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    schema_version: Literal[1]
    benchmark: Literal["mmlu-pro"]
    runs: list[object] = Field(max_length=40)


def read_mmlu_pro(directory=None):
    root = Path(directory or os.environ.get("NEKAISE_BENCH_MMLU_PRO_PROJECTION_DIR") or
                ROOT.parent / "nekaise-bench/workspace/mmlu-pro/projection").resolve()
    empty = {"schema_version": 1, "benchmark": "mmlu-pro", "status": "unavailable", "runs": [], "stale": False}
    try:
        path = root / "mmlu-pro.json"
        if not path.exists():
            return {**empty, "status": "empty"}
        if path.resolve().parent != root or path.stat().st_size > 256 * 1024:
            return empty
        with path.open("rb") as handle:
            raw = handle.read(256 * 1024 + 1)
        if len(raw) > 256 * 1024:
            return empty
        envelope = Catalog.model_validate(json.loads(raw))
        runs = []
        for item in envelope.runs:
            try:
                row = Aggregate.model_validate(item).model_dump()
            except (ValueError, TypeError):
                continue
            row["stale"] = bool(row["status"] == "running" and
                (datetime.now(timezone.utc) - datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))).total_seconds() > 600)
            runs.append(row)
        ids = [row["run_id"] for row in runs]
        runs = [row for row in runs if ids.count(row["run_id"]) == 1]
        rejected = len(envelope.runs) - len(runs)
        if rejected and not runs:
            return {**empty, "rejected_entries": rejected}
        return {"schema_version": 1, "benchmark": "mmlu-pro", "runs": runs,
                "status": "ok", "stale": bool(runs and runs[0]["stale"]), "rejected_entries": rejected}
    except (OSError, ValueError, TypeError):
        return empty
