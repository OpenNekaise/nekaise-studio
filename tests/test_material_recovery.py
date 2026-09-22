"""Operational recovery of a rejected author batch without manual budget resets."""
import json

import pytest

from test_material_authors import AuthorTeacher, author, configured, response
from test_recovery import decision
from nekaise_loop.author_config import AuthorPool
from nekaise_loop.material_allowance import allowance_status
from nekaise_loop.recovery import apply_recovery, handle_recovery
from nekaise_loop.storage import now
from nekaise_loop.supervisor import tick


@pytest.fixture
def interrupted_material(setup_loop):
    class ThreeJobs(AuthorTeacher):
        jobs = [("one", "a"), ("two", "a"), ("three", "a")]
    requests = []
    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        requests.append(payload)
        result = response(request).json()
        if payload["task"]["job"]["id"] == "two" and "retry_validation" not in payload:
            batch = json.loads(result["choices"][0]["message"]["content"])
            batch["rows"][0]["territory"] = "PRIVATE_REJECTED_VALUE"
            result["choices"][0]["message"]["content"] = json.dumps(batch)
        import httpx
        return httpx.Response(200, json=result)
    pool = AuthorPool(authors=[author()], concurrency=1, max_calls_per_round=3, max_output_tokens_per_round=1536)
    settings, service, campaign, engine = configured(setup_loop, handler, teacher=ThreeJobs, pool=pool)
    config = {**campaign["config"], "auto_recover": True, "manage_history": False}
    service.store.execute("UPDATE campaigns SET config=?,teacher_budget_since='2000-01-01' WHERE id=?",
                          (json.dumps(config), campaign["id"]))
    engine.run(campaign["id"])
    assert len(requests) == 2
    recovery = service.store.one("SELECT * FROM recoveries ORDER BY id DESC")
    return settings, service, campaign, engine, requests, recovery


def funded_decision(service, campaign):
    state = allowance_status(service.store, service.artifacts, campaign["id"])
    result = decision()
    result["material_allowance"] = {
        "round_id": state["round_id"], "budget_since": state["budget_since"],
        "expected_calls": state["used_calls"], "expected_reserved_tokens": state["reserved_tokens"],
        "additional_calls": state["shortfall_calls"], "additional_output_tokens": state["shortfall_output_tokens"],
        "reason": "Inspected forbidden field; retry carries schema diagnostics and preserves prior reservations."}
    return result


def test_orchestrator_funds_retry_once_and_training_consumes_valid_expansion(interrupted_material):
    settings, service, campaign, engine, requests, recovery = interrupted_material
    original_calls = service.store.query("SELECT * FROM material_calls ORDER BY id")
    completed = service.store.one("SELECT id,artifact FROM material_jobs WHERE status='complete'")
    assert "territory" in original_calls[1]["error"]
    assert "PRIVATE_REJECTED_VALUE" not in original_calls[1]["error"]
    proposal = funded_decision(service, campaign)
    assert proposal["material_allowance"]["additional_output_tokens"] == 512
    assert proposal["material_allowance"]["additional_calls"] == 1
    handle_recovery(settings, recovery["id"], agent=lambda *args: proposal)
    apply_recovery(settings, recovery["id"])
    apply_recovery(settings, recovery["id"])
    assert service.store.campaign(campaign["id"])["teacher_budget_since"] == "2000-01-01"
    assert service.store.one("SELECT COUNT(*) AS n FROM material_allowances")["n"] == 1
    assert service.store.one("SELECT COUNT(*) AS n FROM actions WHERE actor='orchestrator' AND kind='resume'")["n"] == 1
    assert service.store.query("SELECT * FROM material_calls ORDER BY id") == original_calls
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert [r["task"]["job"]["id"] for r in requests] == ["one", "two", "two", "three"]
    assert requests[2]["retry_validation"] == [{"path": ["rows", 0, "territory"], "type": "extra_forbidden"}]
    assert "PRIVATE_REJECTED_VALUE" not in json.dumps(requests[2])
    assert service.store.one("SELECT id,artifact FROM material_jobs WHERE id=?", (completed["id"],)) == completed
    calls = service.store.query("SELECT * FROM material_calls ORDER BY id")
    assert calls[1]["job_id"] == calls[2]["job_id"]
    saved = service.artifacts.get(calls[2]["artifact"])
    exact = service.artifacts.get(saved["request_artifact"])
    assert json.loads(exact["request"]["messages"][1]["content"]) == requests[2]
    assert service.store.one("SELECT SUM(reserved_tokens) AS n FROM material_calls")["n"] == 2048
    assert len(service.snapshot(campaign["id"])["round"]["materials"]) == 3
    work = service.snapshot(campaign["id"])["round"]["learning_work"]
    retried_job = next(job for job in work["author_yield"]["by_job"] if job["plan_id"] == "two")
    assert retried_job["call_id"] == calls[2]["id"]
    assert retried_job["reserved_output_tokens"] == 512
    assert work["material_author_work"]["reserved_output_tokens_all_attempts"] == 2048


def test_unfunded_internal_budget_wakes_orchestrator_and_does_not_dispatch(interrupted_material):
    settings, service, campaign, engine, requests, previous = interrupted_material
    handle_recovery(settings, previous["id"], agent=lambda *args: decision())
    apply_recovery(settings, previous["id"])
    service.store.execute("UPDATE actions SET handled_at=?", (now(),))
    engine.run(campaign["id"])
    recovery = service.store.one("SELECT * FROM recoveries ORDER BY id DESC")
    assert recovery["kind"] == "material_budget" and recovery["retry_at"] is None
    assert service.store.campaign(campaign["id"])["status"] == "recovering"
    assert tick(service) == ["recover", str(recovery["id"])]
    assert len(requests) == 2


@pytest.mark.parametrize("fault", ["epoch", "round", "calls", "tokens", "overfund", "hold", "live_call", "continue", "wait"])
def test_invalid_grants_never_queue_retry_or_change_budget(interrupted_material, fault):
    settings, service, campaign, _, _, recovery = interrupted_material
    proposal = funded_decision(service, campaign)
    grant = proposal["material_allowance"]
    if fault == "epoch": grant["budget_since"] = "1999-01-01"
    if fault == "round": grant["round_id"] = "unrelated-round"
    if fault == "calls": grant["expected_calls"] += 1
    if fault == "tokens": grant["expected_reserved_tokens"] += 1
    if fault == "overfund": grant["additional_output_tokens"] += 1
    if fault in {"continue", "wait"}: proposal["action"] = fault
    handle_recovery(settings, recovery["id"], agent=lambda *args: proposal)
    if fault == "hold": service.action(campaign["id"], "pause", spawn=False)
    if fault == "live_call": service.store.execute("UPDATE material_calls SET status='running' WHERE id=(SELECT MAX(id) FROM material_calls)")
    apply_recovery(settings, recovery["id"])
    assert service.store.one("SELECT COUNT(*) AS n FROM material_allowances")["n"] == 0
    assert not service.store.one("SELECT id FROM actions WHERE actor='orchestrator' AND kind='resume'")
    assert service.store.campaign(campaign["id"])["teacher_budget_since"] == "2000-01-01"


def test_supplemental_envelope_is_cumulative_across_recoveries(interrupted_material):
    settings, service, campaign, _, _, recovery = interrupted_material
    stage = recovery["stage_id"]
    rid = recovery["round_id"]
    prior = service.store.execute("""INSERT INTO recoveries(campaign_id,round_id,stage_id,kind,status,error,source_hash,created_at,updated_at)
        VALUES(?,?,?,'failure','resolved','Prior fixture retry','fixture',?,?)""", (campaign["id"], rid, stage, now(), now()))
    service.store.execute("""INSERT INTO material_allowances VALUES(?,?,?,?,?,?,?,?)""",
        (prior, campaign["id"], rid, "2000-01-01", 3, 1536, '{}', now()))
    call = service.store.one("SELECT * FROM material_calls ORDER BY id DESC LIMIT 1")
    for _ in range(3):
        service.store.execute("""INSERT INTO material_calls(job_id,stage_id,status,reserved_tokens,input_chars,created_at)
            VALUES(?,?,'failed',512,100,?)""", (call["job_id"], stage, now()))
    proposal = funded_decision(service, campaign)
    handle_recovery(settings, recovery["id"], agent=lambda *args: proposal)
    apply_recovery(settings, recovery["id"])
    assert "Cumulative material repair allowance" in service.store.one("SELECT error FROM recoveries WHERE id=?", (recovery["id"],))["error"]
    assert service.store.one("SELECT COUNT(*) AS n FROM material_allowances")["n"] == 1
    assert not service.store.one("SELECT id FROM actions WHERE actor='orchestrator'")


def test_grant_and_resume_roll_back_together_after_a_host_failure(interrupted_material, monkeypatch):
    settings, service, campaign, _, _, recovery = interrupted_material
    proposal = funded_decision(service, campaign)
    handle_recovery(settings, recovery["id"], agent=lambda *args: proposal)
    from nekaise_loop.storage import Store
    original = Store.event
    def interrupted(self, campaign_id, round_id, kind, *args, **kwargs):
        if kind == "material_allowance":
            raise RuntimeError("Fixture interrupted before queue commit")
        return original(self, campaign_id, round_id, kind, *args, **kwargs)
    monkeypatch.setattr(Store, "event", interrupted)
    apply_recovery(settings, recovery["id"])
    assert service.store.one("SELECT COUNT(*) AS n FROM material_allowances")["n"] == 0
    assert not service.store.one("SELECT id FROM actions WHERE actor='orchestrator'")
    monkeypatch.setattr(Store, "event", original)
    handle_recovery(settings, recovery["id"], agent=lambda *args: proposal)
    apply_recovery(settings, recovery["id"])
    assert service.store.one("SELECT COUNT(*) AS n FROM material_allowances")["n"] == 1
    assert service.store.one("SELECT COUNT(*) AS n FROM actions WHERE actor='orchestrator'")["n"] == 1


def test_missing_request_artifact_remains_visible_to_recovery(interrupted_material):
    _, service, campaign, _, _, _ = interrupted_material
    service.store.execute("UPDATE material_jobs SET input_artifact=? WHERE status='failed'", ('0' * 64,))
    state = allowance_status(service.store, service.artifacts, campaign["id"])
    assert state["needed_output_tokens"] is None
    assert "Cannot read" in state["requirements_error"]


def test_retry_request_is_preserved_even_if_transport_never_returns(interrupted_material):
    settings, service, campaign, engine, _, recovery = interrupted_material
    proposal = funded_decision(service, campaign)
    handle_recovery(settings, recovery["id"], agent=lambda *args: proposal)
    apply_recovery(settings, recovery["id"])
    import httpx
    def disconnected(request):
        raise RuntimeError("Fixture transport vanished before returning evidence")
    engine.material_client_factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(disconnected), **kwargs)
    engine.run(campaign["id"])
    call = service.store.one("SELECT * FROM material_calls ORDER BY id DESC LIMIT 1")
    saved = service.artifacts.get(call["artifact"])
    assert saved["response"] is None and json.loads(call["usage"]) == {}
    exact = service.artifacts.get(saved["request_artifact"])
    assert json.loads(exact["request"]["messages"][1]["content"])["retry_validation"][0]["type"] == "extra_forbidden"
