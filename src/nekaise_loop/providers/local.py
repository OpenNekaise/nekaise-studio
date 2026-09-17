"""JSON file / JSONL subprocess adapter for local student inference and training."""
from __future__ import annotations

import os
import time
from pathlib import Path

from ..artifacts import atomic_write, canonical
from ..config import ROOT


class LocalModel:
    def __init__(self, config, settings, runner, directory):
        self.config, self.settings, self.runner, self.directory = config, settings, runner, directory
        self.deadline = None

    def _run(self, task, payload, on_message, *, timeout=None):
        timeout = self.config.max_stage_seconds if timeout is None else timeout
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("Model operations exhausted their shared stage budget")
        path = self.directory / f"{task}.input.json"
        atomic_write(path, canonical(payload))
        result = []
        def receive(message):
            if message["type"] == "result":
                result.append(message["data"])
            else:
                on_message(message)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
        entrypoint = {"score": "scoring.py", "generate": "generation.py"}.get(task, "model.py")
        self.runner.run([self.settings.model_python, "-u", str(ROOT / "src/nekaise_loop/workers" / entrypoint), task, str(path)], cwd=self.directory, log=self.directory / f"{task}.log", timeout=self.config.max_stage_seconds if timeout is None else timeout, on_message=receive, env=env)
        if len(result) != 1:
            raise RuntimeError("Model worker did not return exactly one result")
        return result[0]

    def generate(self, checkpoint, rows, on_answer=lambda _: None):
        return self._run("generate", {"checkpoint": checkpoint, "rows": rows, "config": self.config.model_dump()}, lambda m: on_answer(m["data"]) if m["type"] == "answer" else None)

    def compare(self, checkpoint, reference, rows, on_answer=lambda _: None, *, reference_format=None, reference_rows=None):
        # Sequential owned processes release the current model before loading the
        # reference. Both loads, generations and audits share one stage budget.
        deadline = self.deadline if self.deadline is not None else time.monotonic() + self.config.max_stage_seconds
        results = {}
        # Keep paired questions in identical batch contexts on both checkpoints.
        # Other current-only questions share the process, but never these batches.
        paired = [r["id"] for r in (reference_rows if reference_rows is not None else rows)]
        current_ids = [r["id"] for r in rows]
        if len(set(current_ids)) != len(rows) or len(set(paired)) != len(paired) or not set(paired) <= set(current_ids):
            raise ValueError("Comparison requires unique IDs and reference prompts from the current set")
        groups = [ids for ids in (paired, [i for i in current_ids if i not in set(paired)]) if ids]
        for name, path in (("current", checkpoint), ("reference", reference)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Checkpoint comparison exhausted its stage budget")
            config = (self.config.model_copy(update={"student_format": reference_format})
                      if name == "reference" and reference_format is not None else self.config)
            model = LocalModel(config, self.settings, self.runner, self.directory / name)
            callback = on_answer if name == "current" else lambda _: None
            selected = reference_rows if name == "reference" and reference_rows is not None else rows
            results[name] = model._run("generate", {"checkpoint": path, "rows": selected, "config": config.model_dump(),
                                                   "batch_groups": groups if name == "current" else ([paired] if paired else [])},
                                       lambda m: callback(m["data"]) if m["type"] == "answer" else None, timeout=remaining)
        return results

    def prepare(self, checkpoint, rows):
        return self._run("prepare", {"checkpoint": checkpoint, "rows": rows, "config": self.config.model_dump()}, lambda _: None)

    def score_history(self, pairs):
        deadline = time.monotonic() + self.config.max_stage_seconds
        self.deadline = deadline
        results = []
        for index, pair in enumerate(pairs):
            observation = {"reference": pair}
            for side in ("before", "after"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Historical scoring exhausted its stage budget")
                local = LocalModel(self.config, self.settings, self.runner, self.directory / f"score-{index}-{side}")
                observation[side] = local._run("score", {"checkpoint": pair[side], "samples": pair["samples"],
                    "dataset_hash": pair["dataset_hash"], "config": self.config.model_dump()}, lambda _: None, timeout=remaining)
            results.append(observation)
        return results

    def train(self, checkpoint, dataset, dataset_hash, on_metric=lambda _: None):
        return self._run("train", {"checkpoint": checkpoint, "dataset": dataset, "dataset_hash": dataset_hash, "output": str(self.directory / "checkpoint"), "config": self.config.model_dump()}, lambda m: on_metric(m["data"]) if m["type"] == "metric" else None)
