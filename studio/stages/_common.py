"""_common — shared stage machinery. FIXED plumbing (bootloader).

What every stage entry point gets from here:

    load_config(path)             YAML config → dict
    assert_frozen(cfg, FROZEN)    the R3 contract: config's `frozen:` must EQUAL the
                                  stage's card constants; drift refuses to run (spec
                                  changes are human review, not loop moves)
    wsd_kwargs(...)               WSD schedule geometry from frozen ratios
    make_reward(tasks)            thin TRL adapter over gym verifiers (R1: logic in gym)
    finish_stage(...)             provenance save + retention (best+last only, R9) +
                                  schema'd log row (R8) + METRIC line
    Budget                        wall-clock hard cap (R9): stop callback + abort record
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))            # gym.*
sys.path.insert(0, str(REPO / "lib"))    # trainkit/datakit/runlog/results (legacy plumbing)


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
    """Warmup–stable–decay geometry (CPT card row). Returns TrainingArguments kwargs."""
    warmup = max(1, math.ceil(total_steps * warmup_ratio))
    decay = max(1, math.ceil(total_steps * decay_ratio))
    stable = max(1, total_steps - warmup - decay)
    return {"lr_scheduler_type": "warmup_stable_decay", "warmup_steps": warmup,
            "lr_scheduler_kwargs": {"num_stable_steps": stable, "num_decay_steps": decay}}


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


# ------------------------------- retention (R9) --------------------------------
def retain_best_and_last(out_dir: Path, last_stage: str) -> list[str]:
    """Keep outputs/<best> + outputs/<last_stage>; delete every other stage dir."""
    out_dir = Path(out_dir)
    best_path = out_dir / "best.json"
    keep = {last_stage}
    if best_path.exists():
        keep.add(Path(json.loads(best_path.read_text()).get("path", "")).name)
    removed = []
    for d in out_dir.iterdir() if out_dir.exists() else []:
        if d.is_dir() and d.name not in keep and (d / "meta.json").exists():
            shutil.rmtree(d)
            removed.append(d.name)
    return removed


# ------------------------------- finish ---------------------------------------
def finish_stage(*, exp_dir: Path, stage: str, metric: str, value: float | None,
                 cfg: dict, budget: Budget, model=None, tok=None,
                 log_fields: dict | None = None) -> None:
    """Checkpoint provenance + retention + schema'd log + METRIC line (one exit path)."""
    import trainkit
    from studio.tools import explog

    out_dir = exp_dir / "outputs"
    if model is not None:
        trainkit.save_checkpoint(model, tok, out_dir / stage,
                                 {"stage": stage, "config": cfg, "metric": metric,
                                  "value": value, "minutes": budget.minutes,
                                  "timeboxed": budget.exceeded})
        if value is not None:
            trainkit.update_best(out_dir, stage, metric, value)
        removed = retain_best_and_last(out_dir, stage)
        if removed:
            print(f"[retention] removed {removed} (best+last policy)")
    explog.append(exp_dir, **{
        "hypothesis": (log_fields or {}).get("hypothesis", ""),
        "variable": (log_fields or {}).get("variable", f"{stage}: config run"),
        "expectation": (log_fields or {}).get("expectation", ""),
        "result": f"{metric}={value}" if value is not None else "no metric",
        "noise_band": explog.load_noise_band(exp_dir, metric),
        "verdict": "aborted-timebox" if budget.exceeded and value is None
                   else (log_fields or {}).get("verdict", "pending"),
        "confidence": (log_fields or {}).get("confidence", "n/a"),
        "stage": stage, "metric": metric, "value": value, "minutes": budget.minutes})
    if value is not None:
        print(f"METRIC {metric}={value:.4f}")
