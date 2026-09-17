"""Read-only dashboard projection. Never imported by teaching or recovery tools."""
from __future__ import annotations

import json
import hashlib
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
    observation_kind: Literal['baseline', 'milestone', 'sample'] = 'sample'
    numerical_n: int | None = Field(default=None, ge=0)
    numerical_correct: int | None = Field(default=None, ge=0)
    choice_n: int | None = Field(default=None, ge=0)
    choice_correct: int | None = Field(default=None, ge=0)
    gained: int | None = Field(default=None, ge=0)
    lost: int | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def coherent(self):
        if self.correct > self.n or self.invalid + self.budget_exhausted > self.n:
            raise ValueError('Inconsistent count')
        if abs(self.score - self.correct/self.n) > 1e-9:
            raise ValueError('Inconsistent score')
        if self.delta_ci95 and not -1 <= self.delta_ci95[0] <= self.delta_ci95[1] <= 1:
            raise ValueError('Invalid difference interval')
        for name in ('numerical', 'choice'):
            n, correct, score = getattr(self, name + '_n'), getattr(self, name + '_correct'), getattr(self, name)
            if (n is None) != (correct is None):
                raise ValueError('Incomplete component counts')
            if n is not None and (not 0 <= correct <= n <= self.n or
                    (n == 0 and score is not None) or
                    (n > 0 and (score is None or abs(score - correct / n) > 1e-9))):
                raise ValueError('Inconsistent component counts')
        if (self.gained is None) != (self.lost is None):
            raise ValueError('Incomplete transitions')
        if self.gained is not None and (self.gained + self.lost > self.n or self.delta is None or
                abs(self.delta - (self.gained - self.lost) / self.n) > 1e-9):
            raise ValueError('Inconsistent transitions')
        for stamp in (self.checkpoint_at, self.evaluated_at):
            if datetime.fromisoformat(stamp.replace('Z', '+00:00')).tzinfo is None:
                raise ValueError('Timestamp requires timezone')
        return self


class Issue(ProjectionModel):
    model_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    status: Literal['failed_infra', 'skipped', 'incompatible']


class Comparison(ProjectionModel):
    kind: Literal['baseline', 'milestone']
    reference_model_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    reference_round_id: str | None = Field(default=None, max_length=100)
    reference_tokens: int = Field(ge=0)
    reference_score: float = Field(ge=0, le=1)
    n: int = Field(gt=0, le=100000)
    delta: float = Field(ge=-1, le=1)
    delta_ci95: list[float] = Field(min_length=2, max_length=2)
    gained: int = Field(ge=0)
    lost: int = Field(ge=0)
    invalid_to_correct: int = Field(ge=0)
    correct_to_invalid: int = Field(ge=0)
    clusters: int = Field(ge=1)
    interval_reliable: bool
    interpretation: Literal['descriptive']

    @model_validator(mode='after')
    def coherent(self):
        if not -1 <= self.delta_ci95[0] <= self.delta_ci95[1] <= 1:
            raise ValueError('Invalid difference interval')
        if (self.gained + self.lost > self.n or self.invalid_to_correct > self.gained or
                self.correct_to_invalid > self.lost or self.clusters > self.n or
                abs(self.delta - (self.gained - self.lost) / self.n) > 1e-9):
            raise ValueError('Inconsistent comparison')
        return self


class Comparisons(ProjectionModel):
    baseline: Comparison | None = None
    milestone: Comparison | None = None


class ComparisonReasons(ProjectionModel):
    baseline: Literal['no_baseline', 'same_checkpoint', 'incompatible'] | None = None
    milestone: Literal['no_earlier_milestone', 'same_checkpoint', 'incompatible'] | None = None


class History(ProjectionModel):
    id: str = Field(pattern=r'^[a-f0-9]{64}$')
    cohort_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    count: int = Field(gt=0)
    page_size: Literal[100]
    pages: int = Field(gt=0)
    overview_count: int = Field(gt=0, le=200)

    @model_validator(mode='after')
    def coherent(self):
        if self.pages != (self.count + self.page_size - 1) // self.page_size or self.overview_count > self.count:
            raise ValueError('Invalid history size')
        return self


class Projection(ProjectionModel):
    schema_version: Literal[1, 2]
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
    history: History | None = None
    comparisons: Comparisons | None = None
    comparison_reasons: ComparisonReasons | None = None

    @model_validator(mode='after')
    def coherent(self):
        if self.history and self.history.overview_count != len(self.points):
            raise ValueError('Overview count mismatch')
        if self.comparisons:
            for kind in ('baseline', 'milestone'):
                pair = getattr(self.comparisons, kind)
                if pair and (not self.latest or pair.kind != kind or pair.n != self.latest.n or
                             abs(self.latest.score - pair.reference_score - pair.delta) > 1e-9 or
                             pair.reference_tokens > self.latest.retained_tokens):
                    raise ValueError('Comparison does not describe latest observation')
        return self


def bounded_json(path, root, limit):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError('Projection path escapes configured directory')
    with path.open('rb') as handle:
        raw = handle.read(limit+1)
    if len(raw) > limit:
        raise ValueError('Projection exceeds size limit')
    return json.loads(raw)


def projection_root(directory=None):
    return Path(directory or os.environ.get('NEKAISE_BENCH_PROJECTION_DIR',
        ROOT.parent/'nekaise-bench/workspace/monitor/projection')).expanduser().resolve()


def read_benchmark(campaign_id, directory=None):
    """Return only validated aggregate fields; even unknown nested keys are discarded."""
    empty = {'campaign_id': campaign_id, 'status': 'unavailable', 'latest': None, 'points': [], 'issues': []}
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', campaign_id):
        return empty
    root = projection_root(directory)
    try:
        value = Projection.model_validate(bounded_json(root/(campaign_id+'.json'), root, 1024*1024))
        if value.campaign_id != campaign_id:
            raise ValueError('Wrong campaign projection')
        data = value.model_dump()
        # Separate freshness from score state: an old valid score stays visibly old.
        pulse = bounded_json(root.parent/'heartbeat.json', root.parent, 4096)
        heartbeat = pulse.get('at')
        poll = pulse.get('poll_seconds', 60)
        if type(poll) is not int or not 5 <= poll <= 3600:
            raise ValueError('Invalid observer heartbeat interval')
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))).total_seconds()
        data.update(service_stale=age > max(180, poll * 3) or age < -60, heartbeat_at=heartbeat)
        return data
    except FileNotFoundError:
        return {**empty, 'status': 'not_ready'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return empty


def sealed_json(root, category, key):
    if not re.fullmatch(r'[a-f0-9]{64}', key):
        raise ValueError('Invalid aggregate identity')
    path = (root/'history'/category/(key+'.json')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Aggregate path escapes configured directory')
    with path.open('rb') as handle:
        raw = handle.read(1024*1024 + 1)
    if len(raw) > 1024*1024 or hashlib.sha256(raw).hexdigest() != key:
        raise ValueError('Aggregate history seal mismatch')
    return json.loads(raw)


def read_benchmark_history(campaign_id, history_id, page, directory=None):
    """Serve one immutable, sealed aggregate page; never read private runs."""
    empty = {'campaign_id': campaign_id, 'status': 'unavailable', 'points': []}
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', campaign_id) or type(page) is not int or page < 0:
        return empty
    root = projection_root(directory)
    try:
        manifest = sealed_json(root, 'manifests', history_id)
        if manifest['schema_version'] != 2 or manifest['campaign_id'] != campaign_id or manifest['page_size'] != 100:
            raise ValueError('Wrong history manifest')
        count, pages = manifest['count'], manifest['pages']
        if (type(count) is not int or count <= 0 or not isinstance(pages, list) or
                len(pages) != (count + 99)//100 or page >= len(pages)):
            raise ValueError('Wrong page index')
        identity = {k: manifest[k] for k in ('root_id', 'protocol_id', 'release_id')}
        if any(not isinstance(v, str) or not re.fullmatch(r'[a-f0-9]{64}', v) for v in identity.values()):
            raise ValueError('Invalid cohort identity')
        # Bench io.digest uses UTF-8, sorted keys and compact separators, without a newline.
        cohort = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if cohort != manifest['cohort_id']:
            raise ValueError('Cohort identity mismatch')
        data = sealed_json(root, 'pages', pages[page])
        if data['schema_version'] != 2 or data['cohort_id'] != cohort or data['offset'] != page * 100:
            raise ValueError('Wrong history page')
        if not isinstance(data['points'], list) or len(data['points']) != min(100, count-page*100):
            raise ValueError('Incomplete history page')
        points = [Point.model_validate(p).model_dump() for p in data['points']]
        if (any(any(p[k] != v for k, v in identity.items()) for p in points) or
                any(a['retained_tokens'] > b['retained_tokens'] for a, b in zip(points, points[1:]))):
            raise ValueError('History mixes evaluation series or order')
        return {'schema_version': 2, 'campaign_id': campaign_id, 'history_id': history_id,
                'cohort_id': cohort, 'count': count, 'page_size': 100, 'page': page,
                'pages': len(pages), 'points': points}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        return empty
