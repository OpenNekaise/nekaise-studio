"""Execution fixtures; no generated answer here is student learning evidence."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from nekaise_loop.config import ROOT, Settings
from nekaise_loop.workers.generation import plan_batches


def test_batch_limits_count_padding_and_reserved_completions():
    lengths = {"a": 10, "b": 20, "c": 5, "d": 20, "e": 4}
    groups = [["b", "a", "c"], ["d", "e"]]
    assert plan_batches(lengths, groups, max_rows=4, max_tokens=60, max_new_tokens=10) == [["b", "a"], ["c"], ["d", "e"]]
    assert plan_batches(lengths, None, max_rows=1, max_tokens=60, max_new_tokens=10) == [[x] for x in lengths]
    assert plan_batches({}, [], max_rows=4, max_tokens=60, max_new_tokens=10) == []


@pytest.mark.parametrize("groups", [[["a", "a"]], [["a"]], [["a", "c"]], [["a", "b"], []]])
def test_batch_groups_must_be_an_exact_partition(groups):
    with pytest.raises(ValueError, match="partition"):
        plan_batches({"a": 1, "b": 2}, groups, max_rows=4, max_tokens=60, max_new_tokens=10)


def test_single_prompt_cannot_silently_overrun_position_budget():
    with pytest.raises(ValueError, match="budget"):
        plan_batches({"long": 51}, None, max_rows=4, max_tokens=60, max_new_tokens=10)


def test_actual_batched_inference_on_tiny_cpu_model(tmp_path):
    runtime = Settings(tmp_path).model_python
    if not Path(runtime).is_file():
        pytest.skip("ML environment needed for the tiny CPU batching probe")
    result = subprocess.run([runtime, str(ROOT/"tests/ml_batch_probe.py"), str(tmp_path)], cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    observation = json.loads(result.stdout.strip().splitlines()[-1])
    assert observation == {"device": "cpu", "mixed_lengths": True, "native_chat": True,
                           "early_eos": True, "preflight": True, "result_order": True}
