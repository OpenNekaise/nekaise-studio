"""Tiny run-logger the training writes to, and the dashboard reads from.

Each run gets a directory under experiments/<exp>/runs/<run_id>/ with:
  - meta.json     : model, pack, metric, status, before/after, baseline, timings
  - events.jsonl  : one line per logging step {step, t, loss, lr, ...}

Zero dependencies. The dashboard (dashboard/server.py) scans these files live.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _runs_dir(exp: str) -> Path:
    return REPO / "experiments" / exp / "runs"


class RunLogger:
    def __init__(self, exp: str, model: str, pack: str, metric: str,
                 baseline: float | None = None, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = _runs_dir(exp) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = self.dir / "events.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.meta = dict(exp=exp, run_id=self.run_id, model=model, pack=pack,
                         metric=metric, baseline=baseline, status="running",
                         started=time.time(), before=None, after=None, delta=None)
        self._flush()

    def _flush(self) -> None:
        self.meta_path.write_text(json.dumps(self.meta, indent=2))

    def log_step(self, step: int, **metrics) -> None:
        with self.events.open("a") as f:
            f.write(json.dumps(dict(step=step, t=time.time(), **metrics)) + "\n")

    def update(self, **kw) -> None:
        self.meta.update(kw)
        self._flush()

    def finish(self, after: float | None = None) -> None:
        if after is not None and self.meta.get("baseline") is not None:
            self.meta["delta"] = round(after - self.meta["baseline"], 4)
        self.meta.update(status="done", after=after, ended=time.time())
        self._flush()


def trainer_callback(logger: "RunLogger"):
    """Return a transformers TrainerCallback that streams loss to `logger`."""
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):
        # Stream whichever of these the trainer reports — SFT logs loss/lr,
        # GRPO logs reward/kl, etc. The dashboard plots whatever shows up.
        KEYS = ("loss", "reward", "learning_rate", "grad_norm", "kl")

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            keep = {k: logs[k] for k in self.KEYS if logs.get(k) is not None}
            if keep:
                logger.log_step(state.global_step, **keep)
    return _Cb()
