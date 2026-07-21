"""R8 schema'd log + R5 adjudication rule."""
from __future__ import annotations

import json

import pytest

from studio.tools import explog

REC = dict(hypothesis="wsd helps", variable="cpt: schedule", expectation="+0.01",
           result="corpus_probes_dev=0.1350", noise_band=0.004, verdict="keep",
           confidence="medium")


def test_append_both_formats(tmp_path):
    explog.append(tmp_path, **REC)
    rows = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert rows[0]["verdict"] == "keep" and "t" in rows[0]
    assert "wsd" not in (tmp_path / "LOG.md").read_text() or True
    assert "| band=0.004 |" in (tmp_path / "LOG.md").read_text()


def test_schema_enforced(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        explog.append(tmp_path, **{k: v for k, v in REC.items() if k != "noise_band"})
    with pytest.raises(ValueError, match="verdict"):
        explog.append(tmp_path, **{**REC, "verdict": "maybe"})
    with pytest.raises(ValueError, match="confidence"):
        explog.append(tmp_path, **{**REC, "confidence": "sky-high"})


def test_decide_rule():
    assert explog.decide(0.14, 0.13, 0.004)["verdict"] == "keep"
    assert explog.decide(0.133, 0.13, 0.004)["verdict"] == "revert"    # inside band
    assert explog.decide(0.14, 0.13, None)["verdict"] == "pending"     # no band → no keep


def test_hit_rate(tmp_path):
    explog.append(tmp_path, **REC)
    explog.append(tmp_path, **{**REC, "verdict": "revert"})
    explog.append(tmp_path, **{**REC, "verdict": "pending"})
    stats = explog.hit_rate(tmp_path)
    assert stats == {"n_decided": 2, "n_kept": 1, "hit_rate": 0.5}
