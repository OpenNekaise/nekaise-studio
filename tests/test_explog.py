"""Typed SQLite decisions + the noise-band rule."""
from __future__ import annotations

import sys

import pytest

from studio.tools import explog


def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(explog, "REPO", tmp_path)
    sys.path.insert(0, str(tmp_path / "lib"))
    from runstore import RunStore
    store = RunStore(tmp_path)
    store.create_run(experiment="exp1", stage="cpt", kind="experiment", run_id="r1")
    return store


REC = dict(hypothesis="data mix helps", variable="cpt:data", expectation="+0.01",
           result="corpus_probes_dev=0.135", noise_band=0.004, verdict="keep",
           confidence="medium", metric="corpus_probes_dev", value=0.135, run_id="r1")


def test_append_persists_typed_decision_without_markdown(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    explog.append(tmp_path / "experiments" / "exp1", **REC)
    rows = explog.read(tmp_path / "experiments" / "exp1")
    assert rows[0]["verdict"] == "keep"
    assert rows[0]["metadata"]["hypothesis"] == "data mix helps"
    assert not (tmp_path / "experiments" / "exp1" / "LOG.md").exists()


def test_schema_enforced(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="missing"):
        explog.append(tmp_path, **{k: v for k, v in REC.items() if k != "noise_band"})
    with pytest.raises(ValueError, match="verdict"):
        explog.append(tmp_path, **{**REC, "verdict": "maybe"})
    with pytest.raises(ValueError, match="confidence"):
        explog.append(tmp_path, **{**REC, "confidence": "sky-high"})


def test_decide_rule():
    assert explog.decide(0.14, 0.13, 0.004)["verdict"] == "keep"
    assert explog.decide(0.133, 0.13, 0.004)["verdict"] == "revert"
    assert explog.decide(0.14, 0.13, None)["verdict"] == "pending"


def test_hit_rate(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    explog.append(tmp_path / "experiments" / "exp1", **REC)
    store.create_run(experiment="exp1", stage="cpt", kind="experiment", run_id="r2")
    explog.append(tmp_path / "experiments" / "exp1", **{**REC, "run_id": "r2",
                                                           "verdict": "revert"})
    assert explog.hit_rate(tmp_path / "experiments" / "exp1") == {
        "n_decided": 2, "n_kept": 1, "hit_rate": 0.5}
