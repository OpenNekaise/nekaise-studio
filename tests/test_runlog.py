"""RunLogger is the trainer callback facade over the durable RunStore."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import runlog  # noqa: E402


def test_runlogger_writes_state_and_events(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "REPO", tmp_path)
    logger = runlog.RunLogger(
        "exp1", model="m", pack="p", metric="acc", baseline=0.5,
        run_id="r1", stage="cpt",
    )
    logger.log_step(1, loss=1.0)
    logger.log_step(2, loss=0.5, reward=0.1)
    logger.finish(after=0.62)

    directory = tmp_path / "experiments" / "exp1" / "runs" / "r1"
    state = json.loads((directory / "state.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert state["status"] == "succeeded"
    assert state["metadata"]["after"] == 0.62
    assert abs(state["metadata"]["delta"] - 0.12) < 1e-9
    assert manifest["run_id"] == "r1" and manifest["stage"] == "cpt"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert [event["step"] for event in events] == [1, 2]
    assert events[1]["reward"] == 0.1


def _trained_logger(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(runlog, "REPO", tmp_path)
    logger = runlog.RunLogger("exp1", model="m", pack="p", metric="acc", run_id=run_id,
                              stage="cpt")
    trainer = logger.dir / "scratch" / "_trainer" / "checkpoint-10"
    trainer.mkdir(parents=True)
    (trainer / "trainer_state.json").write_text("{}")
    return logger


def test_timeboxed_run_keeps_recovery_state_and_records_it(tmp_path, monkeypatch):
    logger = _trained_logger(tmp_path, monkeypatch, "r-timeboxed")
    logger.trained(checkpoint={"digest": "d", "path": "p"}, minutes=1.0, timeboxed=True)
    state = json.loads((logger.dir / "state.json").read_text())
    assert state["status"] == "aborted"
    assert (logger.dir / "scratch" / "_trainer" / "checkpoint-10").is_dir()
    assert state["metadata"]["recovery_dir"] == str(logger.dir / "scratch" / "_trainer")


def test_completed_run_persists_status_then_cleans_scratch(tmp_path, monkeypatch):
    logger = _trained_logger(tmp_path, monkeypatch, "r-done")
    logger.trained(checkpoint={"digest": "d", "path": "p"}, minutes=1.0, timeboxed=False)
    state = json.loads((logger.dir / "state.json").read_text())
    assert state["status"] == "trained"
    assert not (logger.dir / "scratch").exists()
    assert "recovery_dir" not in (state.get("metadata") or {})


def test_cleanup_failure_never_invalidates_a_completed_run(tmp_path, monkeypatch):
    import shutil

    def boom(path):
        raise OSError("disk hiccup")
    logger = _trained_logger(tmp_path, monkeypatch, "r-cleanup")
    monkeypatch.setattr(shutil, "rmtree", boom)
    logger.trained(checkpoint={"digest": "d", "path": "p"}, minutes=1.0, timeboxed=False)
    state = json.loads((logger.dir / "state.json").read_text())
    assert state["status"] == "trained"
    assert "disk hiccup" in state["metadata"]["cleanup_error"]
