"""R3 done-when: each stage's FROZEN constants == its config's `frozen:` section, and the
RLVR values == the JustRL numbers in SPEC Appendix A. Drift in either direction refuses."""
from __future__ import annotations

from pathlib import Path

import yaml

from studio.stages import cpt, opd, rlvr, sft
from studio.stages import agentic

REPO = Path(__file__).resolve().parents[1]

STAGES = {"cpt": cpt, "sft": sft, "rlvr": rlvr, "opd": opd, "agentic": agentic}


def test_configs_match_card():
    for name, mod in STAGES.items():
        cfg = yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())
        assert cfg["frozen"] == mod.FROZEN, f"configs/{name}.yaml frozen drifts from card"


def test_rlvr_is_justrl_verbatim():
    f = rlvr.FROZEN
    assert f["learning_rate"] == 1.0e-6 and f["schedule"] == "constant"
    assert f["num_generations"] == 8
    assert f["temperature"] == 1.0
    assert (f["epsilon"], f["epsilon_high"]) == (0.2, 0.28)   # clip [0.8, 1.28]
    assert f["beta"] == 0.0                                   # no KL
    assert f["entropy_regularization"] is False
    assert f["effective_prompts_per_step"] == 256
    assert f["stabilizer"] == "clip_higher_only"


def test_spec_appendix_pins_the_same_numbers():
    spec = (REPO / "SPEC.md").read_text()
    for needle in ("1e-6", "num_generations=8", "epsilon=0.2", "epsilon_high=0.28",
                   "beta=0.0", "temperature=1.0", "[0.8, 1.28]"):
        assert needle in spec, f"SPEC Appendix A lost {needle!r}"


def test_agentic_caps():
    assert agentic.FROZEN["max_steps"] == 15
    assert agentic.FROZEN["process_rewards"] is False
    assert agentic.FROZEN["reward"] == "outcome_only"


def test_stage_dry_run_validates(capsys):
    for name, mod in STAGES.items():
        if name == "agentic":
            continue
        mod.main(["--config", str(REPO / "configs" / f"{name}.yaml"), "--dry-run"])
        assert "OK" in capsys.readouterr().out
