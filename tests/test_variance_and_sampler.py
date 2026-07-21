"""R5 noise band math + R6 difficulty bands / physical splits / training filter."""
from __future__ import annotations

import json

import pytest

from gym.tasks import Task, sampler
from studio.tools.variance_check import band as noise_band


def test_noise_band_math():
    b = noise_band([0.130, 0.134, 0.126])
    assert b["half_range"] == 0.004
    assert b["band"] >= b["std"] and b["band"] >= b["half_range"]
    with pytest.raises(ValueError):
        noise_band([0.13])


def test_difficulty_bands():
    assert sampler.band(0.01) == "frontier"
    assert sampler.band(0.10) == "hard_reserve"
    assert sampler.band(0.20) == "trainable"
    assert sampler.band(0.80) == "trainable"
    assert sampler.band(0.90) == "easy_reserve"
    assert sampler.band(0.99) == "graduated"


def _task(tid):
    return Task(id=tid, prompt="p", verifier="numeric_cloze", meta={"value": 1},
                ref_solution="1", mode="complete")


def test_physical_splits_and_filter(tmp_path):
    rates = {"a": 0.01, "b": 0.5, "c": 0.99, "d": 0.10}
    d = sampler.write_calibration("fake_set", rates, {"note": "test"}, root=tmp_path)
    assert (d / "calibration.json").exists()
    assert json.loads((d / "frontier.json").read_text()) == ["a"]
    assert json.loads((d / "graduated.json").read_text()) == ["c"]
    assert json.loads((d / "trainable.json").read_text()) == ["b"]

    rows = [_task(t) for t in ("a", "b", "c", "d", "unknown")]
    kept = sampler.filter_trainable(rows, "fake_set", root=tmp_path)
    assert [t.id for t in kept] == ["b", "unknown"]          # unknown kept by default
    kept = sampler.filter_trainable(rows, "fake_set", root=tmp_path, missing="drop")
    assert [t.id for t in kept] == ["b"]
    # no calibration sidecar → pool unchanged (filter is opt-in by measurement)
    assert len(sampler.filter_trainable(rows, "no_such_set", root=tmp_path)) == 5
