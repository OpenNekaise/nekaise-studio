import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import FakeTeacher, FakeModel
from nekaise_loop.config import CampaignConfig
from nekaise_loop.engine import Engine
from nekaise_loop.failures import TeacherUnavailable, quota_kind, retry_seconds
from nekaise_loop.ownership import source_lock
from nekaise_loop.recovery import handle_recovery, apply_recovery
from nekaise_loop.service import Service, Conflict
from nekaise_loop.storage import Store, now
from nekaise_loop.supervisor import tick


def new_campaign(setup_loop, **overrides):
    settings, service, old, _ = setup_loop
    config = CampaignConfig.model_validate({**old["config"], "auto_recover": True, **overrides})
    campaign = service.create("Recovery fixture", config)
    return settings, service, campaign


def decision(action="retry", updates=None):
    return {"action": action, "reason": "Fixture recovery decision", "retry_seconds": 60, "config_updates": updates or []}


def test_infinite_loop_stops_only_when_requested(setup_loop):
    settings, service, campaign = new_campaign(setup_loop, rounds=-1)
    def stop():
        return service.store.one("SELECT COUNT(*) AS n FROM rounds WHERE campaign_id=? AND status='complete'", (campaign["id"],))["n"] == 3
    Engine(settings, FakeTeacher, FakeModel).run(campaign["id"], controls=stop)
    assert service.store.campaign(campaign["id"])["status"] == "stopped"
    assert len(service.snapshot(campaign["id"])["rounds"]) == 3


def test_quota_wait_resume_reuses_completed_stages_and_backs_off(setup_loop):
    settings, service, campaign = new_campaign(setup_loop, rounds=1, teacher_retry_seconds=30)
    class QuotaTeacher(FakeTeacher):
        def revise(self, lessons):
            raise TeacherUnavailable("usage limit reached")
    engine = Engine(settings, QuotaTeacher, FakeModel)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "waiting"
    assert tick(service) is None
    incident = service.store.one("SELECT * FROM recoveries")
    assert incident["status"] == "waiting"
    assert incident["attempts"] == 0
    stages = service.store.query("SELECT id,artifact FROM stage_runs WHERE status='complete'")
    service.store.execute("UPDATE recoveries SET retry_at='2000-01-01' WHERE id=?", (incident["id"],))
    assert tick(service) == ["worker"]
    service.store.execute("UPDATE actions SET handled_at=?", (now(),))
    engine.run(campaign["id"])
    second = service.store.one("SELECT * FROM recoveries ORDER BY id DESC")
    from datetime import datetime
    assert (datetime.fromisoformat(second["retry_at"])-datetime.fromisoformat(second["created_at"])).total_seconds() >= 59
    service.action(campaign["id"], "resume", spawn=False)
    Engine(settings, FakeTeacher, FakeModel).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert service.store.query("SELECT id,artifact FROM stage_runs WHERE id<=? AND status='complete'", (stages[-1]["id"],)) == stages
    assert len(FakeModel.datasets) == 1


@pytest.mark.parametrize("action", ["pause", "stop"])
def test_operator_control_cancels_wait_and_prevents_automatic_resume(setup_loop, action):
    _, service, campaign = new_campaign(setup_loop)
    recovery_id = service.store.recover(campaign["id"], "quota", "fixture unavailable", retry_at="2000-01-01")
    service.action(campaign["id"], action, spawn=False)
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (recovery_id,))["status"] == "cancelled"
    assert tick(service) == ["worker"]  # consume the operator command only
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": action}]
    service.store.execute("UPDATE actions SET handled_at=?", (now(),))
    service.store.set_status(campaign["id"], "paused" if action == "pause" else "stopped")
    assert tick(service) is None


def test_fault_wakes_distinct_orchestrator_and_decision_is_applied_once(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    FakeTeacher.fail_gate_once = True
    Engine(settings, FakeTeacher, FakeModel).run(campaign["id"])
    incident = service.store.one("SELECT * FROM recoveries")
    assert service.store.campaign(campaign["id"])["status"] == "recovering"
    assert tick(service) == ["recover", str(incident["id"])]
    def agent(settings, row, campaign, directory, runner):
        assert campaign["config"]["orchestrator_model"] == "gpt-6-astra"
        assert campaign["config"]["teacher_model"] == "gpt-5.6-terra"
        assert runner.table == "recoveries"
        with pytest.raises(BlockingIOError):
            with source_lock():
                pass
        return decision()
    handle_recovery(settings, incident["id"], agent=agent)
    assert tick(service) == ["apply-recovery", str(incident["id"])]
    apply_recovery(settings, incident["id"])
    apply_recovery(settings, incident["id"])
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "resume"}]
    assert service.store.campaign(campaign["id"])["status"] == "queued"


def test_unavailable_orchestrator_does_not_use_repair_allowance(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    recovery_id = service.store.recover(campaign["id"], "failure", "fixture failure")
    def agent(*args):
        raise RuntimeError("You've hit your usage limit")
    handle_recovery(settings, recovery_id, agent=agent)
    row = service.store.one("SELECT * FROM recoveries")
    assert row["status"] == "waiting"
    assert row["attempts"] == 0
    assert row["retry_at"]
    handle_recovery(settings, recovery_id, agent=agent)
    assert len(list((settings.workspace/"recoveries"/str(recovery_id)).glob("attempt-*"))) == 2


def test_repair_budget_is_bounded_and_stop_cancels_running_agent(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    recovery_id = service.store.recover(campaign["id"], "failure", "fixture failure")
    service.store.execute("UPDATE recoveries SET attempts=3 WHERE id=?", (recovery_id,))
    def should_not_run(*args):
        pytest.fail("exhausted repair allowance launched an agent")
    handle_recovery(settings, recovery_id, agent=should_not_run)
    assert service.store.one("SELECT retry_at FROM recoveries")["retry_at"] is None
    service.store.execute("UPDATE recoveries SET attempts=0 WHERE id=?", (recovery_id,))
    def stop_agent(*args):
        from nekaise_loop.processes import Cancelled
        service.action(campaign["id"], "stop", spawn=False)
        raise Cancelled("operator stopped recovery")
    handle_recovery(settings, recovery_id, agent=stop_agent)
    assert service.store.one("SELECT status FROM recoveries")["status"] == "cancelled"
    assert service.store.campaign(campaign["id"])["status"] == "stopping"


def test_failed_source_patch_is_preserved_and_restored(tmp_path):
    from types import SimpleNamespace
    from nekaise_loop.recovery import restore_failed_source
    root, attempt = tmp_path/"repo", tmp_path/"attempt"
    source = root/"src/nekaise_loop/example.py"
    original = attempt/"source-before/src/nekaise_loop/example.py"
    source.parent.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    source.write_text("broken patch")
    original.write_text("original source")
    added = source.with_name("new.py")
    added.write_text("new broken module")
    restore_failed_source(SimpleNamespace(root=root), attempt)
    assert source.read_text() == "original source"
    assert not added.exists()
    assert (attempt/"source-failed/src/nekaise_loop/example.py").read_text() == "broken patch"


def test_config_repair_creates_atomic_continuation_with_parent_checkpoint(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    parent = service.snapshot(campaign["id"])
    recovery_id = service.store.recover(campaign["id"], "failure", "fixture recipe repair")
    handle_recovery(settings, recovery_id, agent=lambda *args: decision("continue", [{"field": "learning_rate", "value": "0.000005"}]))
    apply_recovery(settings, recovery_id)
    apply_recovery(settings, recovery_id)
    children = service.store.query("SELECT id FROM campaigns WHERE parent_campaign_id=?", (campaign["id"],))
    assert len(children) == 1
    child = service.store.campaign(children[0]["id"])
    assert child["status"] == "queued"
    assert child["config"]["student_model"] == parent["round"]["checkpoint"]
    assert child["config"]["learning_rate"] == .000005
    assert service.artifacts.get(child["context_artifact"])["gaps"]
    assert len(service.store.query("SELECT * FROM actions WHERE campaign_id=?", (child["id"],))) == 1
    assert service.snapshot(campaign["id"])["round"]["stages"] == parent["round"]["stages"]


def test_source_change_resume_creates_fresh_campaign(setup_loop, monkeypatch):
    _, service, campaign, engine = setup_loop
    FakeTeacher.fail_gate_once = True
    engine.run(campaign["id"])
    before = service.store.query("SELECT * FROM stage_runs")
    monkeypatch.setattr("nekaise_loop.service.source_fingerprint", lambda: "fixture-source-change")
    result = service.action(campaign["id"], "resume", spawn=False)
    assert result["campaign_id"] != campaign["id"]
    assert result["continued_from"] == campaign["id"]
    assert service.store.query("SELECT * FROM stage_runs") == before
    with pytest.raises(Conflict):
        service.action(campaign["id"], "resume", spawn=False)


def test_pause_wins_over_failure_during_current_stage(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    requested = [False]
    class Teacher(FakeTeacher):
        def revise(self, lessons):
            requested[0] = True
            raise TeacherUnavailable("usage limit")
    Engine(settings, Teacher, FakeModel).run(campaign["id"], pause=lambda: requested[0])
    assert service.store.campaign(campaign["id"])["status"] == "paused"
    assert not service.store.query("SELECT * FROM recoveries")


def test_concurrent_database_initialization_is_safe(tmp_path):
    with ThreadPoolExecutor(max_workers=6) as pool:
        stores = list(pool.map(lambda _: Store(tmp_path), range(12)))
    assert stores[0].one("SELECT version FROM schema_version")["version"] == 2


def test_provider_unavailability_classification():
    assert quota_kind("You've hit your usage limit") == "quota"
    assert quota_kind("HTTP 429: too many requests") == "rate_limit"
    assert quota_kind("CUDA out of memory") is None
    assert retry_seconds("retry-after: 120 seconds") == 120
