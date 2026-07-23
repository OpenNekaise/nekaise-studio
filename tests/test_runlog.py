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
