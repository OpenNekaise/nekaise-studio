"""R3 done-when: each stage's FROZEN constants == its config's `frozen:` section. The
CoAPT round config trains through the CPT stage and must match the same card row.
Drift in either direction refuses."""
from __future__ import annotations

from pathlib import Path

import yaml

from studio.stages import cpt, sft

REPO = Path(__file__).resolve().parents[1]

# config name -> stage module whose FROZEN it must match
CONFIGS = {"cpt": cpt, "sft": sft, "coapt": cpt}


def test_configs_match_card():
    for name, mod in CONFIGS.items():
        cfg = yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())
        assert cfg["frozen"] == mod.FROZEN, f"configs/{name}.yaml frozen drifts from card"


def test_coapt_mix_shares_sum_to_one():
    cfg = yaml.safe_load((REPO / "configs" / "coapt.yaml").read_text())
    mix = cfg["data"]["mix"]
    assert set(mix) == {"raw", "teacher_cpt", "qa_text", "anchor"}
    assert abs(sum(float(v) for v in mix.values()) - 1.0) < 1e-9


def test_stage_dry_run_validates(capsys):
    for name, mod in CONFIGS.items():
        mod.main(["--config", str(REPO / "configs" / f"{name}.yaml"), "--dry-run"])
        assert "OK" in capsys.readouterr().out
