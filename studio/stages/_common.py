"""_common — shared stage machinery. FIXED plumbing (bootloader).

What every stage entry point gets from here:

    load_config(path)             YAML config → dict
    assert_frozen(cfg, FROZEN)    the R3 contract: config's `frozen:` must EQUAL the
                                  stage's card constants; drift refuses to run (spec
                                  changes are human review, not loop moves)
    wsd_kwargs(...)               WSD schedule geometry from frozen ratios
    make_reward(tasks)            thin TRL adapter over gym verifiers (R1: logic in gym)
    finish_stage(...)             immutable content-addressed checkpoint commit
    Budget                        wall-clock hard cap (R9): stop callback + abort record
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))            # gym.*
sys.path.insert(0, str(REPO / "lib"))    # trainkit/datakit/runlog/runstore


# ------------------------------- workspace ------------------------------------
def active_workspace():
    from workspace import Workspace
    return Workspace.resolve(REPO).apply_environment()


def experiment_dir(name: str) -> Path:
    """Runtime experiment directory in the active user workspace."""
    return active_workspace().experiment_dir(name)


# ------------------------------- config ---------------------------------------
def load_config(path: str | Path) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    for section in ("frozen", "data", "run"):
        if section not in cfg:
            raise SystemExit(f"config {path} is missing the '{section}:' section")
    return cfg


def assert_frozen(cfg: dict, frozen: dict, stage: str) -> None:
    """Refuse to run when the config's frozen section drifts from the algorithm card."""
    got = cfg.get("frozen") or {}
    if got != frozen:
        drift = {k: (got.get(k, "<missing>"), frozen.get(k, "<not in card>"))
                 for k in sorted(set(got) | set(frozen)) if got.get(k) != frozen.get(k)}
        lines = "\n".join(f"  {k}: config={a!r}  card={b!r}" for k, (a, b) in drift.items())
        raise SystemExit(
            f"[{stage}] frozen-section drift vs the algorithm card (SPEC.md §1):\n{lines}\n"
            f"The frozen section is spec. Changing it is a human-reviewed spec change — "
            f"edit studio/stages/{stage}.py AND SPEC.md together, never the yaml alone. "
            f"The agent-movable knob is the 'data:' section only.")


def wsd_kwargs(total_steps: int, warmup_ratio: float, decay_ratio: float) -> dict:
    """Warmup–stable–decay geometry (CPT card row). Returns TrainingArguments kwargs.

    ``total_steps`` is an estimate made before TRL tokenizes and packs the dataset. Do
    not pin ``num_stable_steps`` from it: Transformers receives the actual step count
    from Trainer and derives the stable phase so decay always finishes at the real end.
    """
    warmup = max(1, math.ceil(total_steps * warmup_ratio))
    decay = max(1, math.ceil(total_steps * decay_ratio))
    return {"lr_scheduler_type": "warmup_stable_decay", "warmup_steps": warmup,
            "lr_scheduler_kwargs": {"num_decay_steps": decay}}


# ------------------------------- rewards --------------------------------------
def make_reward(rows):
    """TRL reward_funcs adapter over gym verifiers — thin by construction (R1).

    Dataset rows must carry a `task_id` column; TRL passes it back per-completion.
    """
    from gym import tasks as gym_tasks
    by_id = {t.id: t for t in rows}

    def verifier_reward(completions, task_id=None, **kwargs):
        texts = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
        return [gym_tasks.score(by_id[tid], text) for tid, text in zip(task_id, texts)]

    verifier_reward.__name__ = "gym_verifier"
    return verifier_reward


# ------------------------------- budget (R9) -----------------------------------
class Budget:
    """Wall-clock hard cap: cooperative stop for the trainer + abort verdict for the log."""

    def __init__(self, max_minutes: float):
        self.max_minutes = float(max_minutes)
        self.t0 = time.time()

    def callback(self):
        import trainkit
        return trainkit.time_budget_callback(self.max_minutes)

    @property
    def exceeded(self) -> bool:
        return (time.time() - self.t0) / 60 > self.max_minutes

    @property
    def minutes(self) -> float:
        return round((time.time() - self.t0) / 60, 1)


# ------------------------------- finish ---------------------------------------
def finish_stage(*, exp_dir: Path, stage: str, metric: str, value: float | None,
                 cfg: dict, budget: Budget, run_logger,
                 model=None, tok=None, log_fields: dict | None = None) -> dict | None:
    """Commit one immutable checkpoint and advance only mutable pointer aliases.

    Training never chooses ``best``: independent evaluation records the metric and an
    explicit decision moves a best pointer later.
    """
    import trainkit

    if model is not None:
        artifact = trainkit.save_checkpoint(
            model, tok, store=run_logger.store, run_id=run_logger.run_id,
            provenance={"stage": stage, "config": cfg, "metric": metric,
                        "value": value, "minutes": budget.minutes,
                        "timeboxed": budget.exceeded, **(log_fields or {})},
        )
        run_logger.store.set_pointer(exp_dir.name, "latest", run_logger.run_id,
                                     artifact["digest"])
        return artifact
    return None
