import hashlib
import json
import sys
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
    return {"action": action, "reason": "Fixture recovery decision",
            "report": "Inspected the fixture failure; chose this action based on fixture evidence.",
            "retry_seconds": 60, "config_updates": updates or [],
            "run_retention": [], "log_removals": [], "history_review_after_rounds": 10}


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


def test_orchestrator_can_read_model_protocol_documentation_without_decoding_it(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    recovery_id = service.store.recover(campaign["id"], "failure", "Fixture curriculum failure")
    cli = settings.workspace/"fixture-codex"
    protocol_example = 'LOOP {"type": "metric" | "answer" | "result", "data": ...}'
    cli.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                   f"print({protocol_example!r}, flush=True)\n"
                   f"Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text({json.dumps(decision())!r})\n")
    cli.chmod(0o700)
    settings.codex = str(cli)
    handle_recovery(settings, recovery_id)
    row = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    assert row["status"] == "decided" and row["attempts"] == 1
    assert json.loads(row["decision"])["action"] == "retry"
    log = settings.workspace/"recoveries"/str(recovery_id)/"attempt-1/turn-1/agent.log"
    assert protocol_example in log.read_text()
    apply_recovery(settings, recovery_id)
    assert service.store.campaign(campaign["id"])["status"] == "queued"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind":"resume"}]


@pytest.fixture
def repair_runtime(setup_loop, tmp_path, monkeypatch):
    """Real recovery persistence/processes, with code edits and checks in a test repo."""
    from nekaise_loop.processes import ProcessRunner
    settings, service, campaign = new_campaign(setup_loop)
    settings.root = tmp_path/"repair-repo"
    source = settings.root/"src/nekaise_loop/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("original source")
    handbook = settings.root/"docs/COAPT.md"
    handbook.parent.mkdir()
    handbook.write_text("Fixture handbook")
    prompts = settings.root/"prompts"
    prompts.mkdir()
    prompts.joinpath("orchestrator.txt").write_text("Fixture operations instructions")
    fingerprint = lambda: hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr("nekaise_loop.recovery.source_fingerprint", fingerprint)
    monkeypatch.setattr("nekaise_loop.service.source_fingerprint", fingerprint)
    service.store.execute("UPDATE campaigns SET implementation_hash=? WHERE id=?", (fingerprint(), campaign["id"]))
    run = ProcessRunner.run
    checks = []
    scripts = []
    def run_fixture_checks(self, command, **kwargs):
        if command == [sys.executable, "-m", "pytest", "-q"]:
            checks.append(command)
            script = scripts.pop(0) if scripts else "print('Fixture host checks passed')"
            return run(self, [sys.executable, "-c", script], **kwargs)
        return run(self, command, **kwargs)
    monkeypatch.setattr(ProcessRunner, "run", run_fixture_checks)
    recovery_id = service.store.recover(campaign["id"], "failure", "Fixture execution failure")
    directory = settings.workspace/"recoveries"/str(recovery_id)/"attempt-1"
    return settings, service, campaign, recovery_id, directory, source, checks, scripts


@pytest.mark.parametrize("final_action", ["continue", "wait", "pause"])
def test_orchestrator_decides_after_host_checks_and_keeps_every_report(repair_runtime, final_action):
    settings, service, campaign, recovery_id, directory, source, checks, _ = repair_runtime
    def agent(settings, row, campaign, turn_directory, runner):
        feedback = row["feedback"]
        if feedback is None:
            source.write_text("fixture repair")
            result = decision("pause")
            result["report"] = "Fixed fixture defect. Sandbox cannot validate; awaiting host evidence."
        else:
            assert feedback["decision"]["action"] == "pause"
            assert feedback["checks"]["status"] == "passed"
            assert feedback["checks"]["source_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert json.loads((directory/"turn-1/checks.json").read_text()) == feedback["checks"]
            assert not service.store.query("SELECT id FROM actions WHERE handled_at IS NULL")
            assert source.read_text() == "fixture repair"
            result = decision(final_action)
            result["report"] = f"Reviewed host checks; chose {final_action} for fixture operational reasons."
        return result
    handle_recovery(settings, recovery_id, agent=agent)
    assert len(checks) == 1
    record = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    saved = json.loads(record["decision"])
    assert record["status"] == "decided"
    assert saved["action"] == final_action
    assert "Sandbox cannot validate" in (directory/"turn-1/report.md").read_text()
    assert "Reviewed host checks" in (directory/"turn-2/report.md").read_text()
    assert saved == json.loads((directory/"decision.json").read_text())
    assert saved["report"] in (directory/"report.md").read_text()
    apply_recovery(settings, recovery_id)
    apply_recovery(settings, recovery_id)
    record = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    if final_action == "continue":
        child = service.store.campaign(record["continuation_id"])
        assert child["status"] == "queued" and record["status"] == "resolved"
        assert child["parent_campaign_id"] == campaign["id"]
        assert child["config"]["inherit_optimizer"] is True
        assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "start"}]
    else:
        assert record["status"] == "waiting"
        assert record["retry_at"]  # an agent pause also schedules its own review
        assert not service.store.query("SELECT id FROM actions WHERE handled_at IS NULL")


@pytest.mark.parametrize("final_action", ["continue", "pause"])
def test_unsuccessful_host_checks_are_evidence_for_the_agent(repair_runtime, final_action):
    settings, service, campaign, recovery_id, directory, source, checks, scripts = repair_runtime
    scripts.append("raise SystemExit('Fixture unavailable dependency')")
    def agent(settings, row, campaign, turn_directory, runner):
        if row["feedback"] is None:
            source.write_text("fixture repair")
            return decision("continue")
        assert row["feedback"]["checks"]["status"] == "unsuccessful"
        assert "Fixture unavailable dependency" in row["feedback"]["checks"]["error"]
        assert source.read_text() == "fixture repair"  # no automatic rollback on check failure
        return {**decision(final_action), "report": "Reviewed fixture dependency failure and chose the final action."}
    handle_recovery(settings, recovery_id, agent=agent)
    assert len(checks) == 1
    assert not (directory/"source-failed").exists()
    assert json.loads(service.store.one("SELECT decision FROM recoveries WHERE id=?", (recovery_id,))["decision"])["action"] == final_action
    apply_recovery(settings, recovery_id)
    actions = service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL")
    assert actions == ([{"kind": "start"}] if final_action == "continue" else [])


def test_agent_can_repair_again_after_failed_checks(repair_runtime):
    settings, service, campaign, recovery_id, directory, source, checks, scripts = repair_runtime
    scripts.append("raise SystemExit('Fixture regression failed')")
    def agent(settings, row, campaign, turn_directory, runner):
        feedback = row["feedback"]
        if feedback is None:
            source.write_text("first fixture repair")
        elif feedback["checks"]["status"] == "unsuccessful":
            source.write_text("corrected fixture repair")
        else:
            assert source.read_text() == "corrected fixture repair"
        return decision("continue")
    handle_recovery(settings, recovery_id, agent=agent)
    assert len(checks) == 2
    assert (directory/"turn-3/report.md").is_file()
    assert json.loads((directory/"turn-2/checks.json").read_text())["status"] == "passed"
    assert source.read_text() == "corrected fixture repair"


def test_check_request_reaches_host_and_returns_evidence_through_cli(repair_runtime):
    settings, service, campaign, recovery_id, directory, source, checks, _ = repair_runtime
    cli = settings.workspace/"fixture-codex"
    cli.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
                   "context = json.loads(sys.stdin.read().rsplit('\\n\\n', 1)[1])\n"
                   "payload = json.loads(Path(context['incident_file']).read_text())\n"
                   "reports = json.loads(Path(payload['reports']).read_text())\n"
                   "assert reports['reports'] and reports['current']['campaign']\n"
                   "incident = payload['incident']\n"
                   f"result = {decision('check')!r}\n"
                   "if incident['feedback'] is not None:\n"
                   "    assert incident['feedback']['checks']['status'] == 'passed'\n"
                   "    result['action'] = 'retry'\n"
                   "    result['report'] = 'Read actual fixture host results; retry unchanged implementation.'\n"
                   "Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(result))\n")
    cli.chmod(0o700)
    settings.codex = str(cli)
    handle_recovery(settings, recovery_id)
    assert len(checks) == 1 and source.read_text() == "original source"
    assert json.loads((directory/"decision.json").read_text())["action"] == "retry"
    apply_recovery(settings, recovery_id)
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "resume"}]


def test_operator_pause_during_review_wins_over_agent_continuation(repair_runtime):
    settings, service, campaign, recovery_id, directory, source, checks, _ = repair_runtime
    def agent(settings, row, campaign, turn_directory, runner):
        if row["feedback"] is None:
            source.write_text("fixture repair")
        else:
            service.action(campaign["id"], "pause", spawn=False)
        return decision("continue")
    handle_recovery(settings, recovery_id, agent=agent)
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (recovery_id,))["status"] == "cancelled"
    assert source.read_text() == "original source"
    assert (directory/"source-failed/src/nekaise_loop/example.py").read_text() == "fixture repair"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "pause"}]


def test_operator_stop_during_host_check_is_recorded_without_another_agent_turn(repair_runtime, monkeypatch):
    from nekaise_loop.processes import Cancelled, ProcessRunner
    settings, service, campaign, recovery_id, directory, source, checks, _ = repair_runtime
    def stop_check(*args, **kwargs):
        service.action(campaign["id"], "stop", spawn=False)
        raise Cancelled("Fixture operator stopped host check")
    monkeypatch.setattr(ProcessRunner, "run", stop_check)
    handle_recovery(settings, recovery_id, agent=lambda *args: decision("check"))
    assert json.loads((directory/"turn-1/checks.json").read_text())["status"] == "cancelled"
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (recovery_id,))["status"] == "cancelled"
    assert not (directory/"turn-2").exists()
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind": "stop"}]


def test_repeated_check_requests_remain_bounded_and_keep_reports(repair_runtime):
    settings, service, campaign, recovery_id, directory, source, checks, _ = repair_runtime
    handle_recovery(settings, recovery_id, agent=lambda *args: decision("check"))
    assert len(checks) == campaign["config"]["max_repair_attempts"]
    assert len(list(directory.glob("turn-*/report.md"))) == len(checks) + 1
    record = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    assert record["status"] == "waiting" and "exhausted host check requests" in record["error"]
    assert not service.store.query("SELECT id FROM actions WHERE handled_at IS NULL")


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


def test_repair_burst_cools_down_then_reconsiders_and_stop_cancels_agent(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    recovery_id = service.store.recover(campaign["id"], "failure", "fixture failure")
    service.store.execute("UPDATE recoveries SET attempts=3 WHERE id=?", (recovery_id,))
    def should_not_run(*args):
        pytest.fail("exhausted repair allowance launched an agent")
    handle_recovery(settings, recovery_id, agent=should_not_run)
    assert service.store.one("SELECT retry_at FROM recoveries")["retry_at"]
    assert tick(service) is None
    service.store.execute("UPDATE recoveries SET retry_at='2000-01-01' WHERE id=?", (recovery_id,))
    assert tick(service) == ["recover", str(recovery_id)]
    def stop_agent(*args):
        from nekaise_loop.processes import Cancelled
        service.action(campaign["id"], "stop", spawn=False)
        raise Cancelled("operator stopped recovery")
    handle_recovery(settings, recovery_id, agent=stop_agent)
    assert service.store.one("SELECT status FROM recoveries")["status"] == "cancelled"
    assert service.store.campaign(campaign["id"])["status"] == "stopping"


def test_agent_pause_reschedules_then_resumes_without_operator_action(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    recovery_id = service.store.recover(campaign["id"], "failure", "fixture transient problem")
    handle_recovery(settings, recovery_id, agent=lambda *args: decision("pause"))
    apply_recovery(settings, recovery_id)
    assert tick(service) is None
    service.store.execute("UPDATE recoveries SET retry_at='2000-01-01' WHERE id=?", (recovery_id,))
    assert tick(service) == ["recover", str(recovery_id)]
    handle_recovery(settings, recovery_id, agent=lambda *args: decision("retry"))
    apply_recovery(settings, recovery_id)
    assert tick(service) == ["worker"]
    assert service.store.query("SELECT kind FROM actions") == [{"kind": "resume"}]
    service.store.execute("UPDATE actions SET handled_at=?", (now(),))
    Engine(settings, FakeTeacher, FakeModel).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"


def test_legacy_untimed_agent_pause_is_revisited(setup_loop):
    _, service, campaign = new_campaign(setup_loop)
    recovery_id = service.store.recover(campaign["id"], "failure", "legacy automatic pause")
    service.store.execute("UPDATE recoveries SET status='waiting',attempts=3 WHERE id=?", (recovery_id,))
    service.store.set_status(campaign["id"], "waiting")
    assert tick(service) == ["recover", str(recovery_id)]


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
    assert stores[0].one("SELECT version FROM schema_version")["version"] == 4


def test_provider_unavailability_classification():
    assert quota_kind("You've hit your usage limit") == "quota"
    assert quota_kind("HTTP 429: too many requests") == "rate_limit"
    assert quota_kind("CUDA out of memory") is None
    assert retry_seconds("retry-after: 120 seconds") == 120
