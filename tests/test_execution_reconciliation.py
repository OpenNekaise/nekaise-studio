"""Real queue/scheduler/recovery persistence with fixture-only model execution."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import FakeTeacher, FakeModel
from nekaise_loop.engine import Engine
from nekaise_loop.recovery import handle_recovery, apply_recovery
from nekaise_loop.reports import current_status
from nekaise_loop.service import Service, Conflict
from nekaise_loop.storage import Store, now
from nekaise_loop.supervisor import tick
from nekaise_loop.worker import run_worker
from test_recovery import new_campaign, decision


def fixture_worker(monkeypatch, settings, teacher=FakeTeacher):
    engine = Engine(settings, teacher, FakeModel)
    monkeypatch.setattr("nekaise_loop.engine.Engine", lambda settings: engine)
    monkeypatch.setattr("nekaise_loop.worker.signal.signal", lambda *args: None)
    run_worker(settings)


def test_teacher_pause_wakes_review_then_orchestrator_decides(setup_loop, monkeypatch):
    settings, service, campaign = new_campaign(setup_loop)
    class Teacher(FakeTeacher):
        def reflect(self, observations):
            return {**super().reflect(observations), "action": "pause", "reason": "Inspect fixture generation behavior"}
    service.action(campaign["id"], "start", spawn=False)
    fixture_worker(monkeypatch, settings, Teacher)
    before = service.store.query("SELECT * FROM stage_runs")
    status = current_status(service)
    assert status["campaign"]["status"] == "paused"
    assert status["campaign"]["operator_hold"] is None
    assert status["last_action"]["actor"] == "teacher"
    assert "orchestrator" in status["next_action"]
    assert len(FakeModel.datasets) == 1
    # A fresh service models a scheduler restart after the pause was consumed.
    restarted = Service(settings)
    command = tick(restarted)
    incident = restarted.store.one("SELECT * FROM recoveries")
    assert command == ["recover", str(incident["id"])]
    assert "Inspect fixture generation behavior" in incident["error"]
    assert tick(restarted) == command
    assert len(restarted.store.query("SELECT * FROM recoveries")) == 1
    assert not restarted.store.query("SELECT * FROM actions WHERE handled_at IS NULL")
    assert service.store.query("SELECT * FROM stage_runs") == before
    handle_recovery(settings, incident["id"], agent=lambda *args: decision("pause"))
    apply_recovery(settings, incident["id"])
    assert tick(restarted) is None  # preserve the orchestrator's review time
    restarted.store.execute("UPDATE recoveries SET retry_at='2000-01-01'")
    assert tick(restarted) == command
    handle_recovery(settings, incident["id"], agent=lambda *args: decision("retry"))
    apply_recovery(settings, incident["id"])
    assert restarted.store.one("SELECT actor FROM actions ORDER BY id DESC")["actor"] == "orchestrator"
    assert tick(restarted) == ["worker"]
    # Restore the constructor before building a second fixture worker.
    monkeypatch.setattr("nekaise_loop.engine.Engine", Engine)
    fixture_worker(monkeypatch, settings)
    assert restarted.store.campaign(campaign["id"])["status"] == "complete"
    assert len(FakeModel.datasets) == 2


@pytest.mark.parametrize("status", ["paused", "stopped", "failed", "interrupted", "waiting", "recovering"])
def test_missing_handoff_is_reconciled_without_direct_training(setup_loop, status):
    settings, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], status, "Fixture interrupted without an incident")
    restarted = Service(settings)
    assert tick(restarted)[0] == "recover"
    assert not restarted.store.query("SELECT * FROM actions")
    assert not FakeModel.datasets
    assert restarted.store.one("SELECT kind FROM recoveries")["kind"] == "status_review"


@pytest.mark.parametrize("kind", ["pause", "stop"])
def test_operator_hold_survives_status_overwrite_and_consumed_command(setup_loop, monkeypatch, kind):
    settings, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], "running")
    service.action(campaign["id"], kind, spawn=False)
    service.store.execute("UPDATE actions SET handled_at=?", (now(),))
    # Crash after consuming the command; startup must finish the hold, not repair.
    fixture_worker(monkeypatch, settings)
    assert service.store.campaign(campaign["id"])["status"] == ("paused" if kind == "pause" else "stopped")
    service.store.set_status(campaign["id"], "interrupted", "Late worker status write")
    assert tick(Service(settings)) is None
    assert service.store.recover(campaign["id"], "failure", "Late failure") is None
    with pytest.raises(Conflict, match="operator hold"):
        service.action(campaign["id"], "resume", spawn=False, actor="supervisor")
    assert not service.store.query("SELECT * FROM recoveries")
    assert not FakeModel.datasets
    assert "explicit" in current_status(service)["next_action"]


def test_user_pause_after_teacher_pause_cancels_pending_review(setup_loop, monkeypatch):
    settings, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], "running")
    service.action(campaign["id"], "pause", spawn=False, actor="teacher", reason="Fixture teacher concern")
    fixture_worker(monkeypatch, settings)
    assert tick(service)[0] == "recover"
    service.action(campaign["id"], "pause", spawn=False)
    monkeypatch.setattr("nekaise_loop.engine.Engine", Engine)
    fixture_worker(monkeypatch, settings)
    assert tick(service) is None
    assert service.store.one("SELECT status FROM recoveries")["status"] == "cancelled"


def test_reconciliation_and_user_pause_are_serialized(setup_loop):
    _, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], "paused")
    with ThreadPoolExecutor(max_workers=2) as pool:
        review = pool.submit(service.reconcile_execution)
        pause = pool.submit(service.action, campaign["id"], "pause", False)
        review.result()
        pause.result()
    assert service.store.campaign(campaign["id"])["operator_hold"] == "pause"
    assert not service.store.query("SELECT * FROM recoveries WHERE status IN ('pending','running','waiting','decided')")


def test_concurrent_reconciliation_creates_one_incident(setup_loop):
    _, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], "interrupted")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: service.reconcile_execution(), range(8)))
    assert len(service.store.query("SELECT * FROM recoveries")) == 1


def test_reconciliation_preserves_existing_wait_and_execution_budget(setup_loop):
    _, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "budget", "Fixture allowance exhausted")
    assert tick(service) is None
    assert tick(service) is None
    assert service.store.one("SELECT * FROM recoveries")["id"] == rid
    assert not service.store.query("SELECT * FROM actions")


def test_reconciliation_excludes_unstarted_completed_disabled_and_parent_runs(setup_loop):
    _, service, campaign = new_campaign(setup_loop)
    assert tick(service) is None  # unstarted
    service.store.set_status(campaign["id"], "complete")
    assert tick(service) is None
    service.store.set_status(campaign["id"], "paused")
    service.continue_campaign(campaign["id"])
    assert tick(service) is None  # superseded parent, even with no user hold
    disabled = setup_loop[2]
    service.store.set_status(disabled["id"], "paused")
    assert tick(service) is None


def test_live_worker_prevents_review(setup_loop, monkeypatch):
    _, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], "paused")
    monkeypatch.setattr(service, "worker_alive", lambda: True)
    assert tick(service) is None
    assert not service.store.query("SELECT * FROM recoveries")


def test_queued_work_cannot_override_a_durable_hold(setup_loop, monkeypatch):
    settings, service, campaign = new_campaign(setup_loop)
    service.action(campaign["id"], "start", spawn=False)
    service.action(campaign["id"], "stop", spawn=False)
    service.store.execute("UPDATE actions SET handled_at=? WHERE kind='stop'", (now(),))
    service.store.set_status(campaign["id"], "queued", "Late queued-state write")
    fixture_worker(monkeypatch, settings)
    assert service.store.campaign(campaign["id"])["status"] == "stopped"
    assert tick(service) is None
    assert not FakeModel.datasets


def test_explicit_review_releases_legacy_hold_through_queue(setup_loop, monkeypatch):
    settings, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], "paused")
    with service.store.connect() as db:
        db.execute("ALTER TABLE campaigns DROP COLUMN operator_hold")
        db.execute("ALTER TABLE actions DROP COLUMN actor")
        db.execute("ALTER TABLE actions DROP COLUMN reason")
        db.execute("UPDATE schema_version SET version=3")
    migrated = Store(settings.workspace)
    assert migrated.campaign(campaign["id"])["operator_hold"] == "pause"
    assert tick(Service(settings)) is None  # ambiguous historical pauses stay held
    budget_since = migrated.campaign(campaign["id"])["teacher_budget_since"]
    service.action(campaign["id"], "review", spawn=False, reason="User requested investigation of the stuck teacher pause")
    assert service.store.campaign(campaign["id"])["operator_hold"] is None
    assert service.store.campaign(campaign["id"])["teacher_budget_since"] == budget_since
    assert tick(service) == ["worker"]
    fixture_worker(monkeypatch, settings)
    assert tick(service)[0] == "recover"
    assert "User requested investigation" in service.store.one("SELECT error FROM recoveries")["error"]
    assert not FakeModel.datasets


@pytest.mark.parametrize("result", ["retry", "continue", "wait", "pause"])
def test_new_review_supersedes_running_recovery(setup_loop, monkeypatch, result):
    settings, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "failure", "Older investigation")

    def agent(*args):
        service.action(campaign["id"], "review", spawn=False, reason="Newer investigation")
        return decision(result)

    handle_recovery(settings, rid, agent=agent)
    apply_recovery(settings, rid)
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (rid,))["status"] == "cancelled"
    assert service.store.campaign(campaign["id"])["status"] == "recovering"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "review"}]
    assert not service.store.query("SELECT id FROM campaigns WHERE parent_campaign_id=?", (campaign["id"],))
    fixture_worker(monkeypatch, settings)
    newer = service.store.one("SELECT * FROM recoveries ORDER BY id DESC")
    assert newer["id"] != rid and newer["error"] == "Newer investigation"
    assert tick(service) == ["recover", str(newer["id"])]
    assert not FakeModel.datasets


@pytest.mark.parametrize("result", ["retry", "continue", "wait", "pause"])
def test_review_during_decision_application_wins(setup_loop, monkeypatch, result):
    settings, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "failure", "Older investigation")
    handle_recovery(settings, rid, agent=lambda *args: decision(result))
    monkeypatch.setattr("nekaise_loop.recovery.apply_history", lambda *args: service.action(
        campaign["id"], "review", spawn=False, reason="Review during application"))
    apply_recovery(settings, rid)
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (rid,))["status"] == "cancelled"
    assert service.store.campaign(campaign["id"])["status"] == "recovering"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "review"}]
    assert not service.store.query("SELECT id FROM campaigns WHERE parent_campaign_id=?", (campaign["id"],))


@pytest.mark.parametrize("review_first", [True, False])
def test_stale_resume_cannot_run_past_review(setup_loop, monkeypatch, review_first):
    settings, service, campaign = new_campaign(setup_loop)
    service.store.set_status(campaign["id"], "recovering")
    commands = ["review", "resume"] if review_first else ["resume", "review"]
    for kind in commands:
        service.store.execute("INSERT INTO actions(campaign_id,kind,created_at,actor,reason) VALUES(?,?,?,?,?)",
                              (campaign["id"], kind, now(), "operator" if kind == "review" else "orchestrator", "Newer investigation"))
    calls = []
    monkeypatch.setattr(Engine, "run", lambda *args, **kwargs: calls.append(args))
    fixture_worker(monkeypatch, settings)
    assert calls == []
    assert service.store.campaign(campaign["id"])["status"] == "recovering"
    incident = service.store.one("SELECT * FROM recoveries")
    assert incident["status"] == "pending" and incident["error"] == "Newer investigation"
    assert not service.store.query("SELECT id FROM actions WHERE handled_at IS NULL")


def test_review_during_continuation_preparation_prevents_child(setup_loop, monkeypatch):
    settings, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "failure", "Older investigation")
    handle_recovery(settings, rid, agent=lambda *args: decision("continue"))
    from nekaise_loop.artifacts import Artifacts
    original = Artifacts.put

    def put(artifacts, data):
        service.action(campaign["id"], "review", spawn=False, reason="Review during continuation preparation")
        return original(artifacts, data)

    monkeypatch.setattr(Artifacts, "put", put)
    apply_recovery(settings, rid)
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (rid,))["status"] == "cancelled"
    assert service.store.campaign(campaign["id"])["status"] == "recovering"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "review"}]
    assert not service.store.query("SELECT id FROM campaigns WHERE parent_campaign_id=?", (campaign["id"],))


def test_failed_superseded_agent_cannot_replace_review_with_cooldown(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "failure", "Older investigation")

    def agent(*args):
        service.action(campaign["id"], "review", spawn=False, reason="Newer investigation")
        raise RuntimeError("Fixture transport failure after newer review")

    handle_recovery(settings, rid, agent=agent)
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (rid,))["status"] == "cancelled"
    assert service.store.campaign(campaign["id"])["status"] == "recovering"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "review"}]


def test_review_during_recovery_preparation_prevents_old_agent(setup_loop, monkeypatch):
    import shutil
    settings, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "failure", "Older investigation")
    original = shutil.copyfile
    reviewed = []

    def copyfile(*args, **kwargs):
        if not reviewed:
            reviewed.append(service.action(campaign["id"], "review", spawn=False, reason="Newer investigation"))
        return original(*args, **kwargs)

    monkeypatch.setattr("nekaise_loop.recovery.shutil.copyfile", copyfile)
    calls = []
    handle_recovery(settings, rid, agent=lambda *args: calls.append(args))
    assert reviewed and not calls
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (rid,))["status"] == "cancelled"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "review"}]
