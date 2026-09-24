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


def test_provenance_diagnostics_are_bounded_without_echoing_citations():
    from types import SimpleNamespace
    from nekaise_loop.material_jobs import CandidateValidationError, candidates

    row = {"id": "fixture", "kind": "sft", "concept": "fixture",
           "student_prompt": "fixture", "training_text": "fixture",
           "training_tokenization": "full_text", "source_keys": [],
           "seed_ids": ["seed"], "rationale": "fixture"}
    spec = {"sources": {}, "job": {"seed_ids": ["seed"]}}
    result = SimpleNamespace(content=json.dumps({"rows": [row]}), complete=True)
    assert candidates(result, spec)[0]["source_keys"] == []
    row["source_keys"] = ["PRIVATE_REJECTED_VALUE"] * 20
    result.content = json.dumps({"rows": [row]})
    with pytest.raises(CandidateValidationError) as caught:
        candidates(result, spec)
    assert caught.value.diagnostics == [
        {"path": ["rows", 0, "source_keys", i], "type": "unprovided_source_key"}
        for i in range(8)]
    assert "PRIVATE_REJECTED_VALUE" not in str(caught.value)


def test_author_citation_schema_is_job_scoped_and_host_still_checks_provenance():
    from types import SimpleNamespace
    from nekaise_loop.material_jobs import CandidateValidationError, candidates, request_body
    from nekaise_loop.material_types import CandidateBatch
    from nekaise_loop.providers.codex_material import strict_schema

    schema = CandidateBatch.model_json_schema()
    original = json.dumps(schema, sort_keys=True)
    key = "0123456789abcdef" * 4
    spec = {"sources": {key: {}}, "job": {"seed_ids": ["seed"], "max_output_tokens": 512}}
    body = request_body(author(), spec, schema)
    payload = json.loads(body["messages"][1]["content"])
    assert strict_schema(payload["output_schema"])["properties"]["rows"]["minItems"] == 1
    properties = strict_schema(payload["output_schema"])["$defs"]["Candidate"]["properties"]
    assert properties["source_keys"]["items"]["enum"] == [key]
    assert properties["seed_ids"]["items"]["enum"] == ["seed"]
    assert payload["task"] == spec
    empty = {"sources": {}, "job": {"seed_ids": [], "max_output_tokens": 512}}
    other = json.loads(request_body(author(), empty, schema)["messages"][1]["content"])
    empty_properties = strict_schema(other["output_schema"])["$defs"]["Candidate"]["properties"]
    for field in ("source_keys", "seed_ids"):
        assert empty_properties[field]["maxItems"] == 0
        assert "enum" not in empty_properties[field]["items"]
    assert json.dumps(schema, sort_keys=True) == original

    row = {"id": "fixture", "kind": "sft", "concept": "fixture",
           "training_text": "fixture", "training_tokenization": "full_text",
           "source_keys": [key], "seed_ids": ["seed"], "rationale": "fixture"}
    result = SimpleNamespace(content=json.dumps({"rows": [row]}), complete=True)
    assert candidates(result, spec) == [CandidateBatch.model_validate_json(result.content).rows[0].model_dump()]
    # A transport may ignore the schema. The independent host check must still
    # reject a shortened hash without rewriting it or pruning the candidate.
    row["source_keys"] = [key[:-3]]
    result.content = json.dumps({"rows": [row]})
    with pytest.raises(CandidateValidationError, match="unprovided_source_key"):
        candidates(result, spec)
    row["source_keys"], row["seed_ids"] = [], []
    result.content = json.dumps({"rows": [row]})
    assert candidates(result, empty)[0]["source_keys"] == []


@pytest.fixture
def interrupted_material(setup_loop, request):
    class ThreeJobs(AuthorTeacher):
        jobs = [("one", "a"), ("two", "a"), ("three", "a")]
    requests = []
    fault = getattr(request, "param", "extra_field")
    def handler(http_request):
        body = json.loads(http_request.content)
        payload = json.loads(body["messages"][1]["content"])
        requests.append(payload)
        result = response(http_request).json()
        if fault == "output_budget" and payload["task"]["job"]["id"] == "two" and "retry_budget" not in payload:
            from nekaise_loop.providers.material import AuthorHTTPError
            error = AuthorHTTPError("Fixture output exceeded reservation", {"events": []})
            error.usage = {"prompt_tokens": 100, "completion_tokens": 544, "total_tokens": 644}
            raise error
        if fault != "output_budget" and payload["task"]["job"]["id"] == "two" and "retry_validation" not in payload:
            batch = json.loads(result["choices"][0]["message"]["content"])
            if fault == "empty_batch":
                batch["rows"] = []
            elif fault == "empty_source":
                batch["rows"][0]["source_keys"] = [""]
            elif fault == "unknown_seed":
                batch["rows"][0]["seed_ids"] = ["PRIVATE_REJECTED_VALUE"]
            else:
                batch["rows"][0]["territory"] = "PRIVATE_REJECTED_VALUE"
            result["choices"][0]["message"]["content"] = json.dumps(batch)
        import httpx
        return httpx.Response(200, json=result)
    pool = AuthorPool(authors=[author()], concurrency=1, max_calls_per_round=3, max_output_tokens_per_round=1536)
    settings, service, campaign, engine = configured(setup_loop, handler, teacher=ThreeJobs, pool=pool,
        review_policy="trusted_author_v1" if fault == "empty_batch" else "teacher_review_v1")
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


@pytest.mark.parametrize("interrupted_material", ["extra_field", "empty_source", "unknown_seed", "empty_batch"], indirect=True)
def test_orchestrator_funds_retry_once_and_training_consumes_valid_expansion(interrupted_material):
    settings, service, campaign, engine, requests, recovery = interrupted_material
    original_calls = service.store.query("SELECT * FROM material_calls ORDER BY id")
    completed = service.store.one("SELECT id,artifact FROM material_jobs WHERE status='complete'")
    rejected = service.artifacts.get(original_calls[1]["artifact"])
    invalid_rows = json.loads(rejected["response"]["choices"][0]["message"]["content"])["rows"]
    invalid_row = invalid_rows[0] if invalid_rows else {}
    expected = ([{"path": ["rows"], "type": "empty_batch"}] if not invalid_rows else
                [{"path": ["rows", 0, "source_keys", 0], "type": "unprovided_source_key"}]
                if invalid_row["source_keys"] == [""] else
                [{"path": ["rows", 0, "seed_ids", 0], "type": "unprovided_seed_id"}]
                if invalid_row["seed_ids"] == ["PRIVATE_REJECTED_VALUE"] else
                [{"path": ["rows", 0, "territory"], "type": "extra_forbidden"}])
    assert rejected["validation_errors"] == expected
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
    assert requests[2]["retry_validation"] == expected
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


@pytest.mark.parametrize("with_evidence", [False, True])
def test_validation_feedback_survives_intervening_transport_failure(interrupted_material, with_evidence):
    settings, service, campaign, engine, requests, recovery = interrupted_material
    import httpx
    from nekaise_loop.providers.material import AuthorHTTPError

    working_factory = engine.material_client_factory
    completed = service.store.one("SELECT id,artifact FROM material_jobs WHERE status='complete'")
    rejected_call = service.store.one("SELECT * FROM material_calls WHERE status='failed'")
    rejected = service.artifacts.get(rejected_call["artifact"])
    handle_recovery(settings, recovery["id"], agent=lambda *args: funded_decision(service, campaign))
    apply_recovery(settings, recovery["id"])

    def disconnected(request):
        if with_evidence:
            raise AuthorHTTPError("Fixture interrupted turn", {"events": []})
        raise RuntimeError("Fixture transport vanished")

    engine.material_client_factory = lambda **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(disconnected), **kwargs)
    engine.run(campaign["id"])
    failed_calls = service.store.query("SELECT * FROM material_calls ORDER BY id")
    assert failed_calls[-1]["status"] == "failed"
    assert json.loads(failed_calls[-1]["usage"]) == {}

    next_recovery = service.store.one("SELECT * FROM recoveries ORDER BY id DESC")
    handle_recovery(settings, next_recovery["id"], agent=lambda *args: funded_decision(service, campaign))
    apply_recovery(settings, next_recovery["id"])
    engine.material_client_factory = working_factory
    engine.run(campaign["id"])

    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert requests[2]["retry_validation"] == rejected["validation_errors"]
    assert service.store.one("SELECT id,artifact FROM material_jobs WHERE id=?", (completed["id"],)) == completed
    assert service.store.query("SELECT * FROM material_calls ORDER BY id LIMIT 3") == failed_calls
    assert service.artifacts.get(rejected_call["artifact"]) == rejected
    assert service.store.one("SELECT SUM(reserved_tokens) AS n FROM material_calls")["n"] == 2560
    assert service.store.campaign(campaign["id"])["teacher_budget_since"] == "2000-01-01"


@pytest.mark.parametrize("interrupted_material", ["output_budget"], indirect=True)
@pytest.mark.parametrize("transport_failure", [False, True])
def test_output_overrun_feedback_reaches_retry_without_changing_teacher_job(interrupted_material, transport_failure):
    settings, service, campaign, engine, requests, recovery = interrupted_material
    import httpx

    working_factory = engine.material_client_factory
    completed = service.store.one("SELECT id,artifact FROM material_jobs WHERE status='complete'")
    rejected = service.store.one("SELECT * FROM material_calls WHERE status='failed'")
    original_evidence = service.artifacts.get(rejected["artifact"])
    proposal = funded_decision(service, campaign)
    assert proposal["material_allowance"]["additional_output_tokens"] == 544
    handle_recovery(settings, recovery["id"], agent=lambda *args: proposal)
    apply_recovery(settings, recovery["id"])
    if transport_failure:
        def disconnected(request):
            raise RuntimeError("Fixture transport vanished with unknown usage")
        engine.material_client_factory = lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(disconnected), **kwargs)
        engine.run(campaign["id"])
        failed = service.store.one("SELECT * FROM material_calls ORDER BY id DESC LIMIT 1")
        assert json.loads(failed["usage"]) == {}
        next_recovery = service.store.one("SELECT * FROM recoveries ORDER BY id DESC")
        handle_recovery(settings, next_recovery["id"], agent=lambda *args: funded_decision(service, campaign))
        apply_recovery(settings, next_recovery["id"])
        engine.material_client_factory = working_factory

    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    retried = requests[2]
    assert retried["retry_budget"] == {
        "reserved_output_tokens": 512, "reported_output_tokens": 544, "overrun_tokens": 32}
    assert retried["task"] == requests[1]["task"]
    assert retried["output_schema"] == requests[1]["output_schema"]
    assert "retry_validation" not in retried
    assert [r["task"]["job"]["id"] for r in requests] == ["one", "two", "two", "three"]
    calls = service.store.query("SELECT * FROM material_calls ORDER BY id")
    saved = service.artifacts.get(calls[-2]["artifact"])
    exact = service.artifacts.get(saved["request_artifact"])
    assert json.loads(exact["request"]["messages"][1]["content"]) == retried
    assert exact["request"]["max_tokens"] == 512
    assert service.store.one("SELECT * FROM material_calls WHERE id=?", (rejected["id"],)) == rejected
    assert service.artifacts.get(rejected["artifact"]) == original_evidence
    assert service.store.one("SELECT id,artifact FROM material_jobs WHERE id=?", (completed["id"],)) == completed
    assert sum(c["reserved_tokens"] for c in calls) == (2560 if transport_failure else 2048)
    assert service.store.campaign(campaign["id"])["teacher_budget_since"] == "2000-01-01"
    assert len(service.snapshot(campaign["id"])["round"]["materials"]) == 3
