import json
import os
from pathlib import Path
import subprocess

import pytest

from nekaise_loop.config import ROOT, Settings


def test_actual_trainer_preserves_optimizer_and_fp32_weights_across_rounds(tmp_path):
    runtime = Settings(tmp_path).model_python
    if not Path(runtime).is_file():
        pytest.skip("Set NEKAISE_MODEL_PYTHON to run the tiny CPU integration probe")
    result = subprocess.run([runtime, str(ROOT/"tests/ml_continuity_probe.py"), str(tmp_path)], cwd=ROOT, env={**os.environ,"CUDA_VISIBLE_DEVICES":"","TOKENIZERS_PARALLELISM":"false","HF_HUB_OFFLINE":"1"}, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    observation = json.loads(result.stdout.strip().splitlines()[-1])
    assert observation["device"] == "cpu"
    assert observation["global_steps"] == 4
    assert observation["weighted_loss_verified"]
    assert observation["prompt_observations_verified"]
    assert observation["generation_stopping_verified"]
    assert observation["generation_audit_verified"]
    assert observation["native_chat_verified"]
