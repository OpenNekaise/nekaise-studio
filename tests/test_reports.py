import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nekaise_loop.api import create_app
from nekaise_loop.recovery import handle_recovery, apply_recovery, run_agent
from nekaise_loop.reports import agent_context, catalog, current_status, detail
from test_recovery import decision, new_campaign


def test_reports_include_unresolved_and_cancelled_history_with_pagination(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    first = service.store.recover(campaign["id"], "failure", "earlier failure")
    handle_recovery(settings, first, agent=lambda *args: decision("pause"))
    apply_recovery(settings, first)
    service.store.execute("UPDATE recoveries SET status='cancelled' WHERE id=?", (first,))
    second = service.store.recover(campaign["id"], "history_review", "latest review")
    page = catalog(service, limit=1)
    assert [r["id"] for r in page["items"]] == [second]
    assert page["next_before"] == second
    old = catalog(service, before=second, limit=1)
    assert old["items"][0]["status"] == "cancelled" and old["next_before"] is None
    context = agent_context(service)
    assert [r["id"] for r in context["reports"]] == [second, first]
    assert detail(service, first)["turns"][0]["decision"]["report"] == decision()["report"]


def test_report_separates_agent_proposal_from_actual_execution(setup_loop):
    settings, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "failure", "fixture fault")
    handle_recovery(settings, rid, agent=lambda *args: decision("continue"))
    assert detail(service, rid)["recovery"]["status"] == "decided"
    assert "apply" in current_status(service)["next_action"]
    apply_recovery(settings, rid)
    report = detail(service, rid)
    assert report["recovery"]["status"] == "resolved"
    child = report["recovery"]["continuation_id"]
    assert current_status(service)["campaign"]["id"] == child
    assert any(e["kind"] == "recovery_applied" and e["data"]["start_queued"] for e in report["events"])
    assert report["recovery"]["decision"]["report"] == decision()["report"]


@pytest.mark.parametrize("hold", ["pause", "stop"])
def test_recovery_context_follows_incident_not_unrelated_active_campaign(setup_loop, hold):
    settings, service, parent, engine = setup_loop
    engine.run(parent["id"])
    service.store.execute("UPDATE campaigns SET config=?,status='paused' WHERE id=?",
                          (json.dumps({**parent["config"], "auto_recover": True}), parent["id"]))
    service.action(parent["id"], "review", spawn=False, reason="Ancestor investigation")
    child = service.continue_campaign(parent["id"], {"rounds": 1}, start=False)
    service.store.set_status(child["id"], "paused")
    service.action(child["id"], "review", spawn=False, reason="Incident investigation")
    incident_id = service.store.recover(child["id"], "status_review", "Incident investigation")
    service.action(child["id"], hold, spawn=False, reason="Explicit operator hold")
    service.store.set_status(child["id"], "paused" if hold == "pause" else "stopped")
    # A separate active run wins the dashboard selection, but must never replace
    # this incident's ancestry or hide its operator hold in recovery context.
    _, _, unrelated = new_campaign(setup_loop)
    service.store.set_status(unrelated["id"], "paused")
    service.action(unrelated["id"], "review", spawn=False, reason="Unrelated investigation")
    service.store.set_status(unrelated["id"], "running")
    before_actions = service.store.query("SELECT * FROM actions ORDER BY id")
    before_recoveries = service.store.query("SELECT * FROM recoveries ORDER BY id")
    before_campaigns = service.store.query("SELECT * FROM campaigns ORDER BY id")
    default = agent_context(service)
    assert current_status(service)["campaign"]["id"] == unrelated["id"]
    assert [r["reason"] for r in default["operator_review_requests"]["requests"]] == ["Unrelated investigation"]
    scoped = agent_context(service, campaign_id=child["id"])
    assert scoped["current"]["campaign"]["id"] == child["id"]
    assert scoped["current"]["campaign"]["operator_hold"] == hold
    assert "explicit" in scoped["current"]["next_action"]
    assert [r["reason"] for r in scoped["operator_review_requests"]["requests"]] == ["Ancestor investigation", "Incident investigation"]
    assert scoped["reports"] == default["reports"]
    missing = agent_context(service, campaign_id="missing-campaign")
    assert missing["current"]["campaign"] is None
    assert missing["operator_review_requests"]["requests"] == []

    class CaptureRunner:
        def run(self, command, **kwargs):
            # Exercise incident serialization only; never launch an agent.
            Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(decision()))

    directory = settings.workspace / "scope-test"
    directory.mkdir()
    row = service.store.one("SELECT * FROM recoveries WHERE id=?", (incident_id,))
    campaign = service.store.campaign(child["id"])
    campaign["config"]["orchestrator_provider"] = "codex"
    run_agent(settings, row, campaign, directory, CaptureRunner())
    saved = json.loads((directory / "reports.json").read_text())
    assert saved["current"]["campaign"] == scoped["current"]["campaign"]
    assert saved["operator_review_requests"] == scoped["operator_review_requests"]
    assert saved["reports"] == scoped["reports"]
    assert service.store.query("SELECT * FROM actions ORDER BY id") == before_actions
    assert service.store.query("SELECT * FROM recoveries ORDER BY id") == before_recoveries
    assert service.store.query("SELECT * FROM campaigns ORDER BY id") == before_campaigns


def test_report_api_reads_never_enqueue_work_or_serve_raw_provider_logs(setup_loop, monkeypatch):
    settings, service, campaign = new_campaign(setup_loop)
    rid = service.store.recover(campaign["id"], "quota", "fixture quota wait", retry_at="2099-01-01")
    directory = settings.workspace/"recoveries"/str(rid)/"attempt-1"
    directory.mkdir(parents=True)
    (directory/"agent.log").write_text("raw-provider-log-content")
    # A legacy report has no turn subdirectories. Corrupt check evidence should
    # be reported without hiding the rest of the historical incident.
    (directory/"decision.json").write_text(json.dumps(decision()))
    (directory/"checks.json").write_text("incomplete old JSON")
    monkeypatch.setattr("nekaise_loop.service.Service.ensure_worker", lambda *args: None)
    with TestClient(create_app(settings)) as client:
        page = client.get("/api/reports?limit=1")
        report = client.get(f"/api/reports/{rid}")
        assert page.status_code == report.status_code == 200
        assert "raw-provider-log-content" not in report.text
        assert report.json()["turns"][0]["checks_error"]
        assert client.get("/api/reports/999999").status_code == 404
        assert client.get("/api/reports?limit=0").status_code == 422
    assert not service.store.query("SELECT * FROM actions")
    assert service.store.one("SELECT attempts FROM recoveries")["attempts"] == 0
