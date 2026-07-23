"""A skill needs keep decisions from two independently named experiments."""
from __future__ import annotations

from lib.runstore import RunStore
from studio.tools import crystallize_gate as gate


def _decision(store, exp, run_id, finding, verdict):
    store.create_run(experiment=exp, stage="cpt", kind="experiment", run_id=run_id)
    store.record_decision(
        run_id, metric="acc", value=0.5, verdict=verdict, reason="test",
        metadata={"finding": finding},
    )


def test_gate_requires_two_experiments(tmp_path):
    store = RunStore(tmp_path)
    _decision(store, "exp-a", "r1", "data-wins", "keep")
    ok, msg = gate.check("data-wins", root=tmp_path)
    assert not ok and "need ≥2" in msg

    _decision(store, "exp-b", "r2", "data-wins", "keep")
    ok, msg = gate.check("data-wins", root=tmp_path)
    assert ok and "PASS" in msg


def test_reverts_do_not_count(tmp_path):
    store = RunStore(tmp_path)
    _decision(store, "exp-a", "r1", "x", "keep")
    _decision(store, "exp-b", "r2", "x", "revert")
    ok, _ = gate.check("x", root=tmp_path)
    assert not ok


def test_mark_unverified(tmp_path):
    md = tmp_path / "skill.md"
    md.write_text("---\nfinding: x\n---\n# Skill\n")
    gate.mark_unverified(md)
    assert "status: unverified" in md.read_text()
    gate.mark_unverified(md)
    assert md.read_text().count("status: unverified") == 1
