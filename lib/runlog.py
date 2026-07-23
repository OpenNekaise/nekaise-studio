"""Compatibility facade over the agent-first :mod:`runstore`.

New code should treat ``run_id`` as the primary key.  ``RunLogger`` remains the small
training-stage adapter used by Transformers callbacks; durable state and queries live in
SQLite and immutable run files managed by ``RunStore``.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from runstore import RunStore

REPO = Path(__file__).resolve().parents[1]


class RunLogger:
    def __init__(
        self, exp: str, model: str, pack: str, metric: str,
        baseline: float | None = None, run_id: str | None = None,
        *, stage: str = "unknown", kind: str = "experiment", seed: int | None = None,
        dataset_id: str | None = None, config_sha256: str | None = None,
        code: dict | None = None, environment: dict | None = None,
        metadata: dict | None = None,
    ):
        self.store = RunStore(REPO)
        requested = run_id or os.environ.get("NEKAISE_RUN_ID")
        base_meta = {"pack": pack, "metric": metric, "baseline": baseline,
                     **(metadata or {})}
        self.run_id = self.store.create_run(
            experiment=exp, stage=stage, kind=kind, model=model, seed=seed,
            dataset_id=dataset_id, config_sha256=config_sha256,
            command=sys.argv, code=code, environment=environment, metadata=base_meta,
            run_id=requested, status="running", allow_existing=bool(requested),
        )
        self.dir = self.store.run_dir(exp, self.run_id)
        self.events = self.dir / "events.jsonl"
        self.meta_path = self.dir / "state.json"

    @property
    def meta(self) -> dict:
        return self.store.get_run(self.run_id)

    def log_step(self, step: int, **metrics) -> None:
        self.store.log_event(self.run_id, step, metrics)

    def update(self, **fields) -> None:
        direct = {}
        extra = {}
        for key, value in fields.items():
            if key in self.store.RUN_COLUMNS or key in self.store.JSON_COLUMNS:
                direct[key] = value
            else:
                extra[key] = value
        if extra:
            current = self.store.get_run(self.run_id).get("metadata") or {}
            direct["metadata"] = {**current, **extra}
        if direct:
            self.store.update_run(self.run_id, **direct)

    def trained(self, *, checkpoint: dict, minutes: float, timeboxed: bool) -> None:
        self.update(checkpoint_digest=checkpoint["digest"], checkpoint_path=checkpoint["path"],
                    minutes=minutes, timeboxed=timeboxed)
        scratch = self.dir / "scratch"
        if scratch.is_dir() and scratch.parent == self.dir:
            shutil.rmtree(scratch)
        self.store.transition(self.run_id, "aborted" if timeboxed else "trained")

    def evaluating(self) -> None:
        self.store.transition(self.run_id, "evaluating")

    def finish(self, after: float | None = None) -> None:
        current = self.store.get_run(self.run_id).get("metadata") or {}
        baseline = current.get("baseline")
        delta = round(after - baseline, 6) if after is not None and baseline is not None else None
        self.update(after=after, delta=delta)
        self.store.transition(self.run_id, "succeeded")

    def fail(self, error: BaseException | str, *, exit_code: int | None = None) -> None:
        self.store.fail(self.run_id, error, exit_code=exit_code)


def trainer_callback(logger: "RunLogger"):
    """Stream bounded scalar telemetry into the run event log."""
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):
        KEYS = ("loss", "reward", "learning_rate", "grad_norm", "kl", "entropy",
                "completion_length")

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            keep = {k: logs[k] for k in self.KEYS if logs.get(k) is not None}
            if keep:
                logger.log_step(state.global_step, **keep)

    return _Cb()
