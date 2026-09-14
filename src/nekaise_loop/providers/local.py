"""JSON file / JSONL subprocess adapter for local student inference and training."""
from __future__ import annotations

import os
from pathlib import Path

from ..artifacts import atomic_write, canonical
from ..config import ROOT


class LocalModel:
    def __init__(self, config, settings, runner, directory):
        self.config, self.settings, self.runner, self.directory = config, settings, runner, directory

    def _run(self, task, payload, on_message):
        path = self.directory / f"{task}.input.json"
        atomic_write(path, canonical(payload))
        result = []
        def receive(message):
            if message["type"] == "result":
                result.append(message["data"])
            else:
                on_message(message)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
        self.runner.run([self.settings.model_python, "-u", str(ROOT / "src/nekaise_loop/workers/model.py"), task, str(path)], cwd=self.directory, log=self.directory / f"{task}.log", timeout=self.config.max_stage_seconds, on_message=receive, env=env)
        if len(result) != 1:
            raise RuntimeError("Model worker did not return exactly one result")
        return result[0]

    def generate(self, checkpoint, rows, on_answer=lambda _: None):
        return self._run("generate", {"checkpoint": checkpoint, "rows": rows, "config": self.config.model_dump()}, lambda m: on_answer(m["data"]) if m["type"] == "answer" else None)

    def prepare(self, checkpoint, rows):
        return self._run("prepare", {"checkpoint": checkpoint, "rows": rows, "config": self.config.model_dump()}, lambda _: None)

    def train(self, checkpoint, dataset, dataset_hash, on_metric=lambda _: None):
        return self._run("train", {"checkpoint": checkpoint, "dataset": dataset, "dataset_hash": dataset_hash, "output": str(self.directory / "checkpoint"), "config": self.config.model_dump()}, lambda m: on_metric(m["data"]) if m["type"] == "metric" else None)
