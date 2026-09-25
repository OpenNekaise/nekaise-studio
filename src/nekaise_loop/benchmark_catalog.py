"""Read-only, aggregate-only external benchmark surface. Never a teaching input."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ROOT

GPQA_DATASET = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"


class Aggregate(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    schema_version: Literal[1]
    benchmark: Literal["gpqa-diamond"]
    run_id: str = Field(pattern=r"^[0-9]{8}T[0-9]{12}Z-[a-f0-9]{8}$")
    status: Literal["running", "complete", "failed", "interrupted"]
    started_at: str = Field(max_length=40)
    updated_at: str = Field(max_length=40)
    model_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_label: str = Field(max_length=160)
    round_id: str | None = Field(default=None, max_length=100)
    round_number: int | None = Field(default=None, ge=0)
    campaign_id: str | None = Field(default=None, max_length=100)
    protocol: Literal["gpqa-diamond-gen-1"]
    protocol_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    dataset_sha256: Literal[GPQA_DATASET]
    max_new_tokens: int = Field(ge=1, le=8192)
    max_input_tokens: int = Field(ge=1, le=32768)
    n: Literal[198]
    completed: int = Field(ge=0, le=198)
    correct: int | None = Field(default=None, ge=0, le=198)
    score: float | None = Field(default=None, ge=0, le=1)
    ci95: list[float] | None = Field(default=None, min_length=2, max_length=2)
    invalid: int | None = Field(default=None, ge=0, le=198)
    budget_exhausted: int | None = Field(default=None, ge=0, le=198)
    seal_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def coherent(self):
        start, end = (datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (self.started_at, self.updated_at))
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("Invalid evaluation timestamps")
        scores = (self.correct, self.score, self.ci95, self.invalid, self.budget_exhausted, self.seal_sha256)
        if self.status != "complete":
            if any(v is not None for v in scores):
                raise ValueError("Incomplete run has a provisional score")
        else:
            if (any(v is None for v in scores) or not self.model_id or not self.protocol_id or
                    self.completed != self.n or abs(self.score - self.correct / self.n) > 1e-9):
                raise ValueError("Incomplete or inconsistent GPQA result")
            if (self.correct + max(self.invalid, self.budget_exhausted) > self.n or
                    not 0 <= self.ci95[0] <= self.score <= self.ci95[1] <= 1):
                raise ValueError("Inconsistent result diagnostics")
        return self


class Catalog(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    schema_version: Literal[1]
    benchmark: Literal["gpqa-diamond"]
    runs: list[Aggregate] = Field(max_length=40)


def read_gpqa(directory=None):
    root = Path(directory or os.environ.get("NEKAISE_BENCH_GPQA_PROJECTION_DIR") or
                ROOT.parent / "nekaise-bench/workspace/gpqa/projection").resolve()
    empty = {"schema_version": 1, "benchmark": "gpqa-diamond", "status": "unavailable", "runs": [], "stale": False}
    try:
        path = root / "gpqa-diamond.json"
        if not path.exists():
            return {**empty, "status": "empty"}
        if path.resolve().parent != root or path.stat().st_size > 256 * 1024:
            return empty
        with path.open("rb") as handle:
            raw = handle.read(256 * 1024 + 1)
        if len(raw) > 256 * 1024:
            return empty
        value = Catalog.model_validate(json.loads(raw)).model_dump()
        runs = value["runs"]
        if len({r["run_id"] for r in runs}) != len(runs):
            return empty
        stale = bool(runs and runs[0]["status"] == "running" and
                     (datetime.now(timezone.utc) - datetime.fromisoformat(runs[0]["updated_at"].replace("Z", "+00:00"))).total_seconds() > 600)
        return {**value, "status": "ok", "stale": stale}
    except (OSError, ValueError, TypeError):
        return empty
