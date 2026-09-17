import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeTeacher, FakeModel
from nekaise_loop.api import create_app
from nekaise_loop.config import CampaignConfig
from nekaise_loop.engine import Engine
from nekaise_loop.history import apply_history, file_hash, inventory, restore_log, review_due, trash_path
from nekaise_loop.recovery import apply_recovery, handle_recovery
from nekaise_loop.storage import Store, now
from nekaise_loop.teacher_tools import query
from test_recovery import decision


@pytest.fixture
def history_fixture(setup_loop):
    settings, service, old, engine = setup_loop
    engine.run(old["id"])
    r = service.store.one("SELECT id FROM rounds WHERE campaign_id=? ORDER BY number LIMIT 1", (old["id"],))
    path = settings.workspace/"runs"/r["id"]/"draft-1/generate.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Fixture generation log; the durable student answer is in the stage artifact.\n")
    current = service.create("Current fixture", CampaignConfig.model_validate({**old["config"], "auto_recover": True}))
    recovery_id = service.store.recover(current["id"], "history_review", "Fixture history review")
    choice = {"campaign_id": old["id"], "disposition": "archive", "label": "Fixture baseline", "reason": "Keep learning evidence accessible outside the main list."}
    removal = {"path": path.relative_to(settings.workspace).as_posix(), "sha256": file_hash(path), "summary": "The fixture student generated an answer, preserved in its completed stage artifact.", "reason": "Raw transcript duplicates durable fixture evidence."}
    proposal = {**decision(), "run_retention": [choice], "log_removals": [removal], "history_review_after_rounds": 3}
    return settings, service, old, current, recovery_id, path, proposal


def test_history_review_applies_agent_choices_once_and_preserves_learning(history_fixture, monkeypatch):
    settings, service, old, current, recovery_id, path, proposal = history_fixture
    original = path.read_bytes()
    stages = service.store.query("SELECT * FROM stage_runs")
    records = service.store.query("SELECT * FROM records")
    checkpoints = service.store.query("SELECT checkpoint FROM rounds")
    handle_recovery(settings, recovery_id, agent=lambda *args: proposal)
    apply_recovery(settings, recovery_id)
    apply_recovery(settings, recovery_id)
    assert not path.exists()
    assert trash_path(settings.workspace, recovery_id, proposal["log_removals"][0]["path"]).read_bytes() == original
    assert len(service.store.query("SELECT * FROM log_cleanup")) == 1
    assert len(service.store.query("SELECT * FROM history_reviews")) == 1
    assert service.store.query("SELECT * FROM stage_runs") == stages
    assert service.store.query("SELECT * FROM records") == records
    assert service.store.query("SELECT checkpoint FROM rounds") == checkpoints
    assert service.store.campaign(current["id"])["status"] == "queued"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "resume"}]
    archived = next(c for c in service.list_campaigns() if c["id"] == old["id"])
    assert archived["retention"] == "archive" and archived["display_name"] == "Fixture baseline"
    assert service.snapshot(old["id"])["round"]["lessons"]
    context = {"workspace": str(settings.workspace), "campaign_id": current["id"], "corpus_path": old["config"]["corpus_path"]}
    assert any(c["id"] == old["id"] for c in query(context, {"op": "campaigns"})["rows"])
    assert query(context, {"op": "log_summaries"})["rows"][0]["summary"] == proposal["log_removals"][0]["summary"]
    app = create_app(settings)
    monkeypatch.setattr(app.state.service, "ensure_worker", lambda: None)
    with TestClient(app) as client:
        report = client.get("/api/history-reviews").json()[0]
        assert report["result"]["report"] == proposal["report"]
        assert report["result"]["log_removals"][0]["summary"]
    restore_log(service, 1)
    restore_log(service, 1)
    assert path.read_bytes() == original
    assert service.store.one("SELECT status FROM log_cleanup")["status"] == "restored"


@pytest.mark.parametrize("target", ["../outside.log", "/tmp/outside.log", "runs/x/checkpoint/model.safetensors", "artifacts/example.log"])
def test_cleanup_rejects_paths_outside_raw_log_contract_without_partial_changes(history_fixture, target):
    settings, service, old, current, recovery_id, path, proposal = history_fixture
    proposal["log_removals"].append({**proposal["log_removals"][0], "path": target})
    with pytest.raises(ValueError):
        apply_history(service, recovery_id, proposal)
    assert path.exists()
    assert not service.store.query("SELECT * FROM run_retention")
    assert not service.store.query("SELECT * FROM log_cleanup")


def test_cleanup_rejects_stale_hash_and_symlink(history_fixture, tmp_path):
    settings, service, old, current, recovery_id, path, proposal = history_fixture
    path.write_text("Changed since inventory")
    with pytest.raises(ValueError, match="changed"):
        apply_history(service, recovery_id, proposal)
    path.unlink()
    outside = tmp_path/"outside.log"
    outside.write_text("Outside evidence")
    path.symlink_to(outside)
    assert not any(x["path"] == proposal["log_removals"][0]["path"] for x in inventory(service)["logs"])
    with pytest.raises(ValueError, match="Symlink"):
        apply_history(service, recovery_id, proposal)
    assert outside.read_text() == "Outside evidence"


def test_current_campaign_and_its_logs_are_protected(history_fixture):
    settings, service, old, current, recovery_id, path, proposal = history_fixture
    service.store.set_status(old["id"], "running")
    with pytest.raises(ValueError, match="active campaign"):
        apply_history(service, recovery_id, proposal)
    proposal["run_retention"] = []
    assert inventory(service)["logs"][0]["eligible"] is False
    with pytest.raises(ValueError, match="closed cleanup candidate"):
        apply_history(service, recovery_id, proposal)
    assert path.exists()


def test_interrupted_log_move_finishes_from_journal(history_fixture):
    settings, service, old, current, recovery_id, path, proposal = history_fixture
    r = proposal["log_removals"][0]
    service.store.execute("INSERT INTO log_cleanup(recovery_id,path,sha256,summary,reason,bytes,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'planned',?,?)", (recovery_id, r["path"], r["sha256"], r["summary"], r["reason"], path.stat().st_size, now(), now()))
    saved = trash_path(settings.workspace, recovery_id, r["path"])
    saved.parent.mkdir(parents=True)
    path.rename(saved)
    apply_history(service, recovery_id, proposal)
    assert service.store.one("SELECT status FROM log_cleanup")["status"] == "removed"
    # A restart after restoring the file but before recording it is also recoverable.
    saved.rename(path)
    restore_log(service, 1)
    assert service.store.one("SELECT status FROM log_cleanup")["status"] == "restored"


def test_restore_does_not_overwrite_new_log(history_fixture):
    settings, service, old, current, recovery_id, path, proposal = history_fixture
    apply_history(service, recovery_id, proposal)
    path.write_text("New log from a later operation")
    with pytest.raises(ValueError, match="empty destination"):
        restore_log(service, 1)
    assert path.read_text() == "New log from a later operation"


def test_automatic_review_yields_at_round_boundary_and_uses_agent_interval(setup_loop):
    settings, service, old, _ = setup_loop
    c = service.create("Automatic history fixture", CampaignConfig.model_validate({**old["config"], "auto_recover": True, "manage_history": True, "rounds": 4}))
    engine = Engine(settings, FakeTeacher, FakeModel)
    engine.run(c["id"])
    assert service.store.campaign(c["id"])["status"] == "recovering"
    assert service.store.query("SELECT status FROM rounds WHERE campaign_id=?", (c["id"],)) == [{"status": "complete"}]
    first = service.store.one("SELECT * FROM recoveries")
    assert first["kind"] == "history_review"
    prior = service.store.query("SELECT id,artifact,input_hash FROM stage_runs")
    handle_recovery(settings, first["id"], agent=lambda *args: {**decision(), "history_review_after_rounds": 2})
    apply_recovery(settings, first["id"])
    assert not review_due(service.store)
    service.store.execute("UPDATE actions SET handled_at=?", (now(),))
    engine.run(c["id"])
    assert service.store.campaign(c["id"])["status"] == "recovering"
    assert len(service.store.query("SELECT id FROM rounds WHERE campaign_id=? AND status='complete'", (c["id"],))) == 3
    assert service.store.query("SELECT id,artifact,input_hash FROM stage_runs WHERE id<=?", (prior[-1]["id"],)) == prior
    assert not service.store.query("SELECT id FROM campaigns WHERE parent_campaign_id=?", (c["id"],))
    assert service.store.one("SELECT kind FROM recoveries ORDER BY id DESC")["kind"] == "history_review"


def test_schema_v2_migration_keeps_existing_history(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    stages = service.store.query("SELECT * FROM stage_runs")
    with service.store.connect() as db:
        for table in ("history_reviews", "log_cleanup", "run_retention"):
            db.execute(f"DROP TABLE {table}")
        db.execute("ALTER TABLE campaigns DROP COLUMN operator_hold")
        db.execute("ALTER TABLE actions DROP COLUMN actor")
        db.execute("ALTER TABLE actions DROP COLUMN reason")
        db.execute("UPDATE schema_version SET version=2")
    migrated = Store(settings.workspace)
    assert migrated.one("SELECT version FROM schema_version")["version"] == 4
    assert migrated.query("SELECT * FROM stage_runs") == stages
    assert migrated.campaign(campaign["id"])["status"] == "complete"
