"""runlog — the telemetry the dashboard reads; meta/events files and the delta math."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import runlog  # noqa: E402


def test_runlogger_writes_meta_and_events(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "REPO", tmp_path)
    lg = runlog.RunLogger("exp1", model="m", pack="p", metric="acc", baseline=0.5, run_id="r1")
    lg.log_step(1, loss=1.0)
    lg.log_step(2, loss=0.5, reward=0.1)
    lg.finish(after=0.62)

    d = tmp_path / "experiments" / "exp1" / "runs" / "r1"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "done" and meta["after"] == 0.62
    assert abs(meta["delta"] - 0.12) < 1e-9
    events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events] == [1, 2] and events[1]["reward"] == 0.1
