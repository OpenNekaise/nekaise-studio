"""CPT/SFT are checkpoint producers; independent gym evaluation runs afterward."""
from __future__ import annotations

import ast
from pathlib import Path

from studio.stages import _common

REPO = Path(__file__).resolve().parents[1]


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_cpt_and_sft_do_not_import_or_launch_gym():
    for stage in ("cpt", "sft"):
        path = REPO / "studio" / "stages" / f"{stage}.py"
        assert "gym" not in imported_roots(path)
        assert "subprocess" not in imported_roots(path)


def test_old_training_tuning_environment_switches_are_gone():
    example = (REPO / ".env.example").read_text()
    forbidden = (
        "NEKAISE_BASE_MODEL",
        "NEKAISE_METHOD",
        "NEKAISE_CPT_LR",
        "NEKAISE_CPT_FULL",
        "NEKAISE_CPT_4BIT",
    )
    assert all(name not in example for name in forbidden)


def test_active_cpt_model_is_minicpm5_1b_base():
    config = (REPO / "configs" / "cpt.yaml").read_text()
    assert "base_model: openbmb/MiniCPM5-1B-Base" in config
    assert "finetuning: full" in config
    assert "precision: bf16" in config
    assert "kind: lora" not in config


def test_wsd_stable_phase_uses_trainers_actual_step_count():
    kwargs = _common.wsd_kwargs(total_steps=100, warmup_ratio=0.03, decay_ratio=0.10)
    assert kwargs["warmup_steps"] == 3
    assert kwargs["lr_scheduler_kwargs"] == {"num_decay_steps": 10}
