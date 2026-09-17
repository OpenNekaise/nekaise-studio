"""Read-only dashboard projection. Never imported by teaching or recovery tools."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ROOT


class ProjectionModel(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True, allow_inf_nan=False)


class Point(ProjectionModel):
    model_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    root_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    round_id: str | None = Field(default=None, max_length=100)
    round_number: int = Field(ge=0)
    retained_tokens: int = Field(ge=0)
    checkpoint_at: str = Field(max_length=80)
    evaluated_at: str = Field(max_length=80)
    run_id: str = Field(max_length=100)
    protocol_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    protocol: Literal['chat-1', 'completion-1']
    release_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    release_name: str = Field(max_length=100)
    n: int = Field(gt=0, le=100000)
    correct: int = Field(ge=0)
    score: float = Field(ge=0, le=1)
    invalid: int = Field(ge=0)
    budget_exhausted: int = Field(ge=0)
    numerical: float | None = Field(default=None, ge=0, le=1)
    choice: float | None = Field(default=None, ge=0, le=1)
    delta: float | None = Field(default=None, ge=-1, le=1)
    delta_ci95: list[float] | None = Field(default=None, min_length=2, max_length=2)
    clusters: int | None = Field(default=None, ge=0)
    interval_reliable: bool = False

    @model_validator(mode='after')
    def coherent(self):
        if self.correct > self.n or self.invalid + self.budget_exhausted > self.n:
            raise ValueError('Inconsistent count')
        if abs(self.score - self.correct/self.n) > 1e-9:
            raise ValueError('Inconsistent score')
        if self.delta_ci95 and not -1 <= self.delta_ci95[0] <= self.delta_ci95[1] <= 1:
            raise ValueError('Invalid difference interval')
        for stamp in (self.checkpoint_at, self.evaluated_at):
            if datetime.fromisoformat(stamp.replace('Z', '+00:00')).tzinfo is None:
                raise ValueError('Timestamp requires timezone')
        return self


class Issue(ProjectionModel):
    model_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    status: Literal['failed_infra', 'skipped', 'incompatible']


class Projection(ProjectionModel):
    schema_version: Literal[1]
    campaign_id: str = Field(max_length=80)
    generated_at: str = Field(max_length=80)
    heartbeat_at: str | None = Field(default=None, max_length=80)
    status: Literal['no_baseline', 'pending', 'running', 'ok', 'stale', 'failed_infra', 'incompatible']
    latest: Point | None
    points: list[Point] = Field(max_length=200)
    issues: list[Issue] = Field(max_length=50)
    baseline_ready: bool
    target_model_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    target_round: int | None = Field(default=None, ge=0)
    lag_tokens: int | None = Field(default=None, ge=0)
    stale: bool
    interval_seconds: int = Field(ge=60, le=86400)


def bounded_json(path, root, limit):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError('Projection path escapes configured directory')
    with path.open('rb') as handle:
        raw = handle.read(limit+1)
    if len(raw) > limit:
        raise ValueError('Projection exceeds size limit')
    return json.loads(raw)


def read_benchmark(campaign_id, directory=None):
    """Return only validated aggregate fields; even unknown nested keys are discarded."""
    empty = {'campaign_id': campaign_id, 'status': 'unavailable', 'latest': None, 'points': [], 'issues': []}
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', campaign_id):
        return empty
    root = Path(directory or os.environ.get('NEKAISE_BENCH_PROJECTION_DIR',
        ROOT.parent/'nekaise-bench/workspace/monitor/projection')).expanduser().resolve()
    try:
        value = Projection.model_validate(bounded_json(root/(campaign_id+'.json'), root, 1024*1024))
        if value.campaign_id != campaign_id:
            raise ValueError('Wrong campaign projection')
        data = value.model_dump()
        # Separate freshness from score state: an old valid score stays visibly old.
        heartbeat = bounded_json(root.parent/'heartbeat.json', root.parent, 4096).get('at')
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))).total_seconds()
        data.update(service_stale=age > 180 or age < -60, heartbeat_at=heartbeat)
        return data
    except FileNotFoundError:
        return {**empty, 'status': 'not_ready'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return empty
