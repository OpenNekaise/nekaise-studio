import asyncio
import json
from pathlib import Path

import httpx
import pytest

from conftest import FakeModel, FakeTeacher
from nekaise_loop.author_config import AuthorPool, AuthorSpec, credential
from nekaise_loop.config import CampaignConfig
from nekaise_loop.engine import Engine
from nekaise_loop.material_accounting import job_work
from nekaise_loop.teacher_tools import query, replay_lesson


def author(name="a", **changes):
    return AuthorSpec(id=name, label=f"Fixture author {name}", base_url="https://fixture.invalid/v1", model="fixture-model", **changes)


def response(request):
    body = json.loads(request.content)
    task = json.loads(body["messages"][1]["content"])["task"]
    candidate = {"id": "variant", "kind": "sft", "concept": "Resistance and flow", "student_prompt": "What changes when R doubles?",
                 "training_text": "At fixed temperature difference, doubling R halves heat flow.", "training_tokenization": "full_text",
                 "source_keys": list(task["sources"])[:1], "seed_ids": task["job"]["seed_ids"], "rationale": "A varied relationship example"}
    return httpx.Response(200, json={"model": "fixture-model-revision", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"rows": [candidate]})}}],
                                    "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}})


class AuthorTeacher(FakeTeacher):
    jobs = [("one", "a"), ("two", "a")]
    selection_seen = None
    evaluation_materials = None
    def curriculum(self, brief):
        plan = super().curriculum(brief)
        plan["expansion_jobs"] = [{"id": jid, "author_id": aid, "seed_ids": ["l1"], "instructions": "Vary physical quantities", "expected_items": 2, "max_output_tokens": 512} for jid, aid in self.jobs]
        return plan

    def select_materials(self, manifest):
        type(self).selection_seen = manifest
        return {"manifest_hash": manifest["manifest_hash"], "accepted_ids": [], "accepted_jobs": [j["plan_id"] for j in manifest["jobs"]],
                "edits": [], "seed_exclusions": [], "token_mix": {"teacher": 1, "corpus": 0, "replay": 0}, "train_epochs": 2,
                "review_scope": "Fixture chooses all explicit immutable candidate batches", "reason": "Fixture teacher choice"}

    def evaluate(self, curriculum, lessons):
        type(self).evaluation_materials = lessons
        return super().evaluate(curriculum, lessons)


def configured(setup_loop, handler=response, *, teacher=AuthorTeacher, pool=None, review_policy="teacher_review_v1"):
    settings, service, original, _ = setup_loop
    config = CampaignConfig.model_validate({**original["config"], "rounds": 1, "expansion_policy": "required_v1", "material_review_policy": review_policy, "material_authors": (pool or AuthorPool(authors=[author()])).model_dump()})
    campaign = service.create("Material integration", config)
    factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    engine = Engine(settings, teacher, FakeModel, material_client_factory=factory)
    return settings, service, campaign, engine


def test_complete_package_preserves_teacher_choice_and_no_fake_attempts(setup_loop):
    settings, service, campaign, engine = configured(setup_loop)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    detail = service.snapshot(campaign["id"])["round"]
    assert len(detail["materials"]) == 2 and len(detail["lessons"]) == 2
    assert all(r["student"] is None and r["student_observation"] == "not_requested" for r in detail["materials"])
    assert detail["curriculum"]["train_epochs"] == 2
    assert len(AuthorTeacher.evaluation_materials) == 4
    rows = FakeModel.datasets[-1]
    aux = [r for r in rows if r.get("material_origin")]
    assert len(aux) == 2 and aux[0]["material_origin"]["model"] == "fixture-model-revision"
    assert all(r["stream"] == "sft" for r in aux)
    assert all("variant" not in r["id"] for r in FakeModel.prompts)
    replay = replay_lesson(settings.workspace, detail["id"], detail["materials"][0]["id"])
    assert replay["material_origin"] == detail["materials"][0]["material_origin"] and replay["student"] is None
    archived = query({"workspace": str(settings.workspace), "campaign_id": campaign["id"]}, {"op": "material_candidates", "round_id": detail["id"], "limit": 1})
    assert len(archived["rows"]) == 1 and archived["next_offset"] == 1 and archived["total"] == 2
    work = detail["learning_work"]["material_author_work"]
    assert work["calls"] == 2 and work["reported_output_tokens"] == 60
    assert detail["learning_work"]["teacher_efficiency"]["teacher_tokens"] is None  # Author tokens never enter the denominator.
    assert detail["learning_work"]["material_sources"]["targets_by_origin"]["a"] > 0
    # Fewer candidates than the teacher's requested count are preserved as an
    # observed shortfall, never silently topped up or treated as learning failure.
    assert AuthorTeacher.selection_seen["jobs"][0]["expected_items"] == 2
    assert AuthorTeacher.selection_seen["jobs"][0]["candidate_count"] == 1
    sizing = detail["learning_work"]["author_yield"]["by_author"]["a"]
    assert sizing["produced_candidates"] == 2 and sizing["selected_candidates"] == 2
    assert sizing["selected_exact_content_variants"] == 1
    assert sizing["reported_output_tokens_per_candidate"] == 30
    assert sizing["prepared_targets_per_pass"] == detail["learning_work"]["material_sources"]["targets_by_origin"]["a"]


def test_candidate_teaching_pages_retrieve_exact_source_without_repeating_it(setup_loop):
    from nekaise_loop.teacher_context import EVIDENCE_KEY
    settings, service, campaign, engine = configured(setup_loop)
    engine.run(campaign["id"])
    rid = service.store.one("SELECT id FROM rounds WHERE campaign_id=?", (campaign["id"],))["id"]
    context = {"workspace": str(settings.workspace), "campaign_id": campaign["id"]}
    full = query(context, {"op": "material_candidates", "round_id": rid})
    # Exercise a large exact source even when this corpus fixture is short.
    stage = service.store.one("SELECT id,artifact FROM stage_runs WHERE round_id=? AND stage='expand' AND status='complete' ORDER BY id DESC LIMIT 1", (rid,))
    manifest = service.artifacts.get(stage["artifact"])
    for row in manifest["candidates"]:
        for source in row["sources"].values():
            source["text"] = "Synthetic fixture source " * 100
    key = service.artifacts.put(manifest)
    service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (key, stage["id"]))
    last = manifest["candidates"][-1]
    compact = query(context, {"op": "material_candidates", "round_id": rid, "candidate_id": last["id"], "view": "teaching", "limit": 1})
    row = compact["rows"][0]
    assert row["candidate"] == full["rows"][-1]["candidate"]
    for source_key, source in row["sources"].items():
        ref = source["text"][EVIDENCE_KEY]
        assert ref["pointer"].startswith("/candidates/1/")
        assert query(context, ref)["value"] == last["sources"][source_key]["text"]
    assert compact["next_offset"] is None


@pytest.mark.parametrize("missing_usage", [{"prompt_tokens": 7}, None])
def test_author_sizing_keeps_missing_output_usage_distinct_from_zero(setup_loop, missing_usage):
    from nekaise_loop.material_accounting import author_yield
    _, service, campaign, engine = configured(setup_loop)
    engine.run(campaign["id"])
    rid = service.store.one("SELECT id FROM rounds WHERE campaign_id=?", (campaign["id"],))["id"]
    for job in service.store.query("SELECT id,artifact FROM material_jobs WHERE round_id=?", (rid,)):
        result = service.artifacts.get(job["artifact"])
        result["usage"] = missing_usage
        service.store.execute("UPDATE material_jobs SET artifact=? WHERE id=?", (service.artifacts.put(result), job["id"]))
    sizing = author_yield(service.store, service.artifacts, rid, {})["by_author"]["a"]
    assert sizing["produced_candidates"] == 2 and sizing["jobs_without_output_usage"] == 2
    assert sizing["reported_output_tokens"] is None and sizing["reported_output_tokens_per_candidate"] is None


@pytest.mark.parametrize("passes", [0, 2])
def test_trusted_jobs_preauthorize_all_content_without_a_selection_call(setup_loop, passes):
    class TrustedTeacher(AuthorTeacher):
        def curriculum(self, brief):
            plan = super().curriculum(brief)
            plan.update(train_epochs=passes, token_mix={"teacher": 1, "corpus": 0, "replay": 0})
            return plan

        def select_materials(self, manifest):
            raise AssertionError("Trusted author batches must not invoke a Teacher reviewer")

        def reflect(self, observations):
            assert observations["curriculum"]["train_epochs"] == passes
            return super().reflect(observations)

    def incorrect_fixture(req):
        envelope = response(req).json()
        batch = json.loads(envelope["choices"][0]["message"]["content"])
        batch["rows"][0]["training_text"] = "Deliberately incorrect test fixture: 2+2=5. Not a live teaching result."
        envelope["choices"][0]["message"]["content"] = json.dumps(batch)
        return httpx.Response(200, json=envelope)

    _, service, campaign, engine = configured(setup_loop, incorrect_fixture, teacher=TrustedTeacher, review_policy="trusted_author_v1")
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    detail = service.snapshot(campaign["id"])["round"]
    assert len(detail["materials"]) == 2 and all(r["use_for_training"] for r in detail["materials"])
    assert all("2+2=5" in r["training_text"] for r in detail["materials"])  # No hidden semantic gate or dedup.
    assert all(r["material_origin"]["review_policy"] == "trusted_author_v1" for r in detail["materials"])
    selected = service.artifacts.get(service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='material_select'", (detail["id"],))["artifact"])
    assert selected["selection"]["accepted_jobs"] == ["one", "two"]
    assert selected["selection"]["train_epochs"] == passes
    assert selected["selection"]["token_mix"] == {"teacher": 1, "corpus": 0, "replay": 0}
    assert "not individually reviewed" in selected["selection"]["review_scope"]
    assert selected["selection"]["edits"] == selected["selection"]["seed_exclusions"] == []
    work = detail["learning_work"]
    assert work["teacher_stage_seconds"] == pytest.approx(sum(work["stage_seconds"][s] for s in ("select", "revise", "evaluate", "grade", "adapt")))
    assert work["material_author_work"]["calls"] == 2  # Requested diagnostic jobs still execute.
    if passes:
        assert work["material_expansion"]["review_policy"] == "trusted_author_v1"
        assert work["material_expansion"]["selected_candidates"] == 2
        assert work["material_expansion"]["prepared_expanded_targets_per_pass"] > 0
    else:
        assert work["retained_training"]["trained"] is False


def test_trusted_empty_batch_fails_required_targets_without_silent_recipe_change(setup_loop):
    def empty_batch(req):
        envelope = response(req).json()
        envelope["choices"][0]["message"]["content"] = '{"rows":[]}'
        return httpx.Response(200, json=envelope)
    _, service, campaign, engine = configured(setup_loop, empty_batch, review_policy="trusted_author_v1")
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    row = service.store.one("SELECT id FROM rounds WHERE campaign_id=?", (campaign["id"],))
    selected = service.artifacts.get(service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='material_select'", (row["id"],))["artifact"])
    assert selected["curriculum"]["train_epochs"] > 0
    assert not service.store.query("SELECT id FROM stage_runs WHERE round_id=? AND stage='train'", (row["id"],))


def test_trusted_author_still_rejects_invalid_source_provenance(setup_loop):
    def invalid_source(req):
        envelope = response(req).json()
        batch = json.loads(envelope["choices"][0]["message"]["content"])
        batch["rows"][0]["source_keys"] = ["not-provided"]
        envelope["choices"][0]["message"]["content"] = json.dumps(batch)
        return httpx.Response(200, json=envelope)
    _, service, campaign, engine = configured(setup_loop, invalid_source, review_policy="trusted_author_v1")
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert service.store.one("SELECT COUNT(*) AS n FROM material_jobs WHERE status='failed'")["n"] > 0
    assert not service.store.query("SELECT id FROM stage_runs WHERE stage='material_select'")


def test_completed_jobs_reused_after_partial_failure(setup_loop):
    calls = []
    def handler(req):
        task = json.loads(json.loads(req.content)["messages"][1]["content"])["task"]
        name = task["job"]["id"]
        calls.append(name)
        if name == "two" and calls.count("two") == 1:
            return httpx.Response(503, json={"error": "fixture"})
        return response(req)
    settings, service, campaign, engine = configured(setup_loop, handler, pool=AuthorPool(authors=[author()], concurrency=1))
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert not service.store.query("SELECT id FROM stage_runs WHERE stage='train' AND round_id IN (SELECT id FROM rounds WHERE campaign_id=?)", (campaign["id"],))
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert calls == ["one", "two", "two"]
    rid = service.snapshot(campaign["id"])["round"]["id"]
    assert job_work(service.store, rid)["calls"] == 3


def test_author_seed_examples_separate_feedback_from_output_fields(setup_loop):
    from nekaise_loop.material_types import Candidate
    seen = []

    def handler(req):
        payload = json.loads(json.loads(req.content)["messages"][1]["content"])
        task = payload["task"]
        for seed in task["seeds"]:
            assert set(seed) <= Candidate.model_fields.keys()
            assert "teacher" not in seed
        seen.append(task)
        return response(req)

    _, service, campaign, engine = configured(setup_loop, handler)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert len(seen) == 2
    for task in seen:
        revised = service.artifacts.get(task["seed_artifact"])
        originals = {row["id"]: row for row in revised["lessons"]}
        assert task["seed_feedback"] == {
            seed_id: originals[seed_id]["teacher"] for seed_id in task["job"]["seed_ids"]
        }
        for seed in task["seeds"]:
            for field in ("student_prompt", "training_text", "training_response", "training_tokenization"):
                assert seed[field] == originals[seed["id"]].get(field)


def test_concurrency_respects_global_author_and_shared_pools_without_head_blocking(setup_loop):
    class ParallelTeacher(AuthorTeacher):
        jobs = [("one", "a"), ("two", "b"), ("three", "a"), ("four", "c")]
    running = set()
    peak, pool_peak, seen_independent = 0, 0, False
    async def handler(req):
        nonlocal peak, pool_peak, seen_independent
        task = json.loads(json.loads(req.content)["messages"][1]["content"])["task"]
        jid, aid = task["job"]["id"], task["job"]["author_id"]
        running.add((jid, aid))
        peak = max(peak, len(running))
        pool_peak = max(pool_peak, sum(a in {"a", "b"} for _, a in running))
        seen_independent |= aid == "c" and len(running) == 2
        await asyncio.sleep(.03)
        running.remove((jid, aid))
        return response(req)
    pool = AuthorPool(authors=[author("a", resource_pool="gpu"), author("b", resource_pool="gpu"), author("c")], resource_limits={"gpu": 1}, concurrency=2)
    _, service, campaign, engine = configured(setup_loop, handler, teacher=ParallelTeacher, pool=pool)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert peak == 2 and pool_peak == 1 and seen_independent


@pytest.mark.parametrize("status,kind", [(429, "rate_limit"), (402, "quota")])
def test_quota_wait_preserves_material_plan(setup_loop, status, kind):
    _, service, campaign, engine = configured(setup_loop, lambda req: httpx.Response(status, headers={"retry-after": "31"}))
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "waiting"
    assert service.store.one("SELECT COUNT(*) AS n FROM material_jobs WHERE status='waiting'")["n"]
    assert not service.store.query("SELECT * FROM stage_runs WHERE stage='material_select'")


def test_cancellation_closes_outstanding_calls_and_records_uncertain_usage(setup_loop):
    started = 0
    closed = 0
    async def handler(req):
        nonlocal started, closed
        started += 1
        try:
            await asyncio.sleep(10)
        finally:
            closed += 1
        return response(req)
    _, service, campaign, engine = configured(setup_loop, handler)
    engine.run(campaign["id"], controls=lambda: started >= 2)
    assert closed == started == 2
    assert service.store.campaign(campaign["id"])["status"] == "stopped"
    assert all(r["status"] == "cancelled" for r in service.store.query("SELECT status FROM material_calls"))


def test_budget_reservation_is_atomic_across_concurrent_calls(setup_loop):
    count = 0
    def handler(req):
        nonlocal count
        count += 1
        return httpx.Response(503)
    pool = AuthorPool(authors=[author()], max_output_tokens_per_round=1024)
    _, service, campaign, engine = configured(setup_loop, handler, pool=pool)
    engine.run(campaign["id"])
    engine.run(campaign["id"])
    engine.run(campaign["id"])
    assert count == 2
    assert service.store.campaign(campaign["id"])["status"] == "waiting"
    assert service.store.one("SELECT SUM(reserved_tokens) AS n FROM material_calls")["n"] == 1024


def test_unfunded_retry_does_not_dispatch_partial_package_or_renew_automatically(setup_loop):
    class ThreeJobs(AuthorTeacher):
        jobs = [("one", "a"), ("two", "a"), ("three", "a")]
    calls = []
    fail = [True]
    def handler(req):
        name = json.loads(json.loads(req.content)["messages"][1]["content"])["task"]["job"]["id"]
        calls.append(name)
        return httpx.Response(503) if fail[0] and name == "two" else response(req)
    pool = AuthorPool(authors=[author()], concurrency=1, max_output_tokens_per_round=1536)
    _, service, campaign, engine = configured(setup_loop, handler, teacher=ThreeJobs, pool=pool)
    service.store.execute("UPDATE campaigns SET teacher_budget_since='2000-01-01' WHERE id=?", (campaign["id"],))
    engine.run(campaign["id"])
    assert calls == ["one", "two"]
    service.action(campaign["id"], "resume", spawn=False, actor="orchestrator")
    assert service.store.campaign(campaign["id"])["teacher_budget_since"] == "2000-01-01"
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "waiting"
    assert calls == ["one", "two"]  # 512 left cannot fund the two 512-token jobs.
    assert "1024 required" in service.store.campaign(campaign["id"])["error"]
    fail[0] = False
    service.action(campaign["id"], "resume", spawn=False, actor="operator")
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert calls == ["one", "two", "two", "three"]


@pytest.mark.parametrize("fault", ["length", "json", "source", "teacher_field"])
def test_invalid_outputs_are_saved_but_never_trained(setup_loop, fault):
    def handler(req):
        d = response(req).json()
        if fault == "length": d["choices"][0]["finish_reason"] = "length"
        elif fault == "json": d["choices"][0]["message"]["content"] = "not JSON"
        else:
            batch = json.loads(d["choices"][0]["message"]["content"])
            if fault == "teacher_field":
                batch["rows"][0]["teacher"] = "Input-only feedback must not enter candidates"
            else:
                batch["rows"][0]["source_keys"] = ["invented"]
            d["choices"][0]["message"]["content"] = json.dumps(batch)
        return httpx.Response(200, json=d)
    _, service, campaign, engine = configured(setup_loop, handler)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert service.store.one("SELECT artifact FROM material_calls WHERE artifact IS NOT NULL")
    assert not FakeModel.datasets


def test_teacher_can_omit_all_candidates_and_disable_training(setup_loop):
    class Diagnostic(AuthorTeacher):
        def select_materials(self, manifest):
            d = super().select_materials(manifest)
            d.update(accepted_jobs=[], train_epochs=0, token_mix={"teacher": 0, "corpus": 0, "replay": 0})
            return d
    _, service, campaign, engine = configured(setup_loop, teacher=Diagnostic)
    engine.run(campaign["id"])
    detail = service.snapshot(campaign["id"])["round"]
    assert detail["status"] == "complete" and detail["checkpoint"] == detail["model_before"]
    assert not any(r["use_for_training"] for r in detail["materials"])


def test_local_endpoint_options_are_independent_and_registry_is_frozen(setup_loop):
    settings, service, original, _ = setup_loop
    pool = AuthorPool(authors=[author(location="local", options={"temperature": .3})])
    settings.material_authors_path.write_text(pool.model_dump_json())
    c = service.create("Default registry", CampaignConfig(student_model=original["config"]["student_model"]))
    settings.material_authors_path.write_text(AuthorPool().model_dump_json())
    assert c["config"]["material_authors"]["authors"][0]["location"] == "local"
    assert c["config"]["material_authors"]["authors"][0]["options"] == {"temperature": .3}


def test_dotenv_credentials_are_not_shell_evaluated_or_overwritten(tmp_path, monkeypatch):
    path = tmp_path/".env"
    path.write_text('DEEPSEEK_API_KEY="literal-$(do-not-run)" # comment\nOTHER=hidden\n')
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert credential("DEEPSEEK_API_KEY", path) == "literal-$(do-not-run)"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-value")
    assert credential("DEEPSEEK_API_KEY", path) == "environment-value"


def test_transport_never_persists_echoed_credentials(setup_loop, monkeypatch):
    monkeypatch.setenv("FIXTURE_AUTHOR_KEY", "fixture-secret-should-not-be-in-artifacts")
    def handler(req):
        assert req.headers["authorization"] == "Bearer fixture-secret-should-not-be-in-artifacts"
        d = response(req).json();d["echo"] = "fixture-secret-should-not-be-in-artifacts"
        return httpx.Response(200, json=d)
    settings, service, campaign, engine = configured(setup_loop, handler, pool=AuthorPool(authors=[author(api_key_env="FIXTURE_AUTHOR_KEY")]))
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    for p in settings.workspace.rglob('*'):
        if p.is_file(): assert b"fixture-secret-should-not-be-in-artifacts" not in p.read_bytes()


def test_teacher_can_edit_exact_candidates_and_preserve_originals(setup_loop):
    class Editing(AuthorTeacher):
        def select_materials(self, manifest):
            d = super().select_materials(manifest)
            d.update(accepted_jobs=[], seed_exclusions=["l1"], edits=[{"candidate_id": manifest["candidate_preview"][0]["id"], "replacement": {
                "id": "edited", "kind": "cpt", "concept": "Teacher correction", "training_tokenization": "full_text",
                "training_text": "An exact teacher replacement.", "rationale": "Corrected by primary teacher", "source_keys": [], "seed_ids": []}}])
            return d
    _, service, campaign, engine = configured(setup_loop, teacher=Editing)
    engine.run(campaign["id"])
    detail = service.snapshot(campaign["id"])["round"]
    assert detail["status"] == "complete"
    chosen = [r for r in detail["materials"] if r["use_for_training"]]
    assert len(chosen) == 1 and chosen[0]["teacher"] == "An exact teacher replacement."
    assert chosen[0]["material_origin"]["teacher_edited"] is True
    assert not next(r for r in detail["lessons"] if r["id"] == "l1")["use_for_training"]
    original = service.artifacts.get(chosen[0]["material_origin"]["response_artifact"])
    assert "An exact teacher replacement." not in json.dumps(original)


def test_unknown_selection_cannot_enter_frozen_data(setup_loop):
    class Invalid(AuthorTeacher):
        def select_materials(self, manifest):
            d = super().select_materials(manifest)
            d["accepted_ids"] = ["nonexistent"]
            return d
    _, service, campaign, engine = configured(setup_loop, teacher=Invalid)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert not FakeModel.datasets


def test_hashed_job_selection_is_diagnosed_and_retry_reuses_author_results(setup_loop):
    calls = []
    class WrongNamespace(AuthorTeacher):
        def select_materials(self, manifest):
            choice = super().select_materials(manifest)
            if not calls:
                choice["accepted_jobs"] = [manifest["jobs"][0]["job_id"]]
            calls.append(choice)
            return choice
    _, service, campaign, engine = configured(setup_loop, teacher=WrongNamespace)
    engine.run(campaign["id"])
    failed = service.store.campaign(campaign["id"])
    assert failed["status"] == "failed"
    assert "accepted_jobs must use manifest plan_id names, not job_id artifact hashes" in failed["error"]
    assert "allowed plan_id values: ['one', 'two']" in failed["error"]
    assert calls[0]["accepted_jobs"][0] in failed["error"]
    assert not FakeModel.datasets
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert len(service.store.query("SELECT id FROM material_calls")) == 2
    assert calls[1]["accepted_jobs"] == ["one", "two"]


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("selection", ["all", "none", "no_jobs"])
def test_material_adapter_binds_plan_names_and_preserves_empty_choice(setup_loop, provider, selection):
    from nekaise_loop.providers.teacher import CliTeacher
    from nekaise_loop.teaching import MaterialSelection
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"], pause=lambda: True)
    row = service.store.one("SELECT id FROM rounds")
    jobs = [] if selection == "no_jobs" else [
        {"plan_id": "named_plan", "job_id": "a" * 64},
        {"plan_id": "other_plan", "job_id": "b" * 64},
    ]
    choice = {"manifest_hash": "c" * 64, "accepted_ids": [],
              "accepted_jobs": [j["plan_id"] for j in jobs] if selection == "all" else [],
              "edits": [], "seed_exclusions": [], "token_mix": {"teacher": 0, "corpus": 0, "replay": 0},
              "train_epochs": 0, "review_scope": "Fixture diagnostic", "reason": "Teacher may omit all jobs"}
    class Runner:
        def run(self, command, **kwargs):
            saved = json.loads((kwargs["cwd"] / "input.json").read_text())
            schema = saved["schema"]
            field = schema["properties"]["accepted_jobs"]
            if jobs:
                assert field["items"]["enum"] == ["named_plan", "other_plan"]
                assert all(j["job_id"] not in field["items"]["enum"] for j in jobs)
            else:
                assert field["maxItems"] == 0
                assert "enum" not in field["items"]
            assert field.get("minItems", 0) == 0
            assert schema["additionalProperties"] is False
            assert "accepted_jobs" in schema["required"]
            if provider == "codex":
                assert json.loads(Path(command[command.index("--output-schema") + 1]).read_text()) == schema
                Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(choice))
                return ""
            assert json.loads(command[command.index("--json-schema") + 1]) == schema
            return json.dumps({"structured_output": choice})
    config = CampaignConfig.model_validate({**campaign["config"], "teacher_provider": provider})
    teacher = CliTeacher(config, settings, service.store, campaign["id"], row["id"], Runner(), settings.workspace / "selection-adapter")
    assert teacher.select_materials({"jobs": jobs, "candidate_ids": []}) == MaterialSelection.model_validate(choice).model_dump()
    # A per-request enum must not leak into another round's base model schema.
    assert "enum" not in MaterialSelection.model_json_schema()["properties"]["accepted_jobs"]["items"]


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("count,width", [(0, 1), (65, 1), (201, 1), (150, 160)])
def test_candidate_reference_schema_uses_complete_manifest_with_size_fallback(setup_loop, provider, count, width):
    from nekaise_loop.providers.teacher import CliTeacher
    from nekaise_loop.teaching import MaterialSelection
    from nekaise_loop.materials import selection_brief
    from types import SimpleNamespace
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"], pause=lambda: True)
    row = service.store.one("SELECT id FROM rounds")
    ids = [f"material-{'p' * (width // 2)}:variant_{i}{'x' * (width // 2)}" for i in range(count)]
    outputs = {"select": {"curriculum": {}}, "revise": {"lessons": []}}
    manifest = {"manifest_hash": "c" * 64, "jobs": [], "candidates": [
        {"id": key, "author_id": "a", "plan_id": "plan", "checks": {}, "candidate": {
            "concept": "Fixture", "kind": "sft", "student_prompt": "Question",
            "training_text": "", "training_response": "Answer"}} for key in ids]}
    brief = selection_brief(SimpleNamespace(round=row, output=outputs.__getitem__), manifest)
    assert brief["candidate_ids"] == ids
    assert len(brief["candidate_preview"]) == min(count, 40)
    choice = {"manifest_hash": "c" * 64, "accepted_ids": ids[-1:], "accepted_jobs": [],
              "edits": [], "seed_exclusions": [], "token_mix": {"teacher": 0, "corpus": 0, "replay": 0},
              "train_epochs": 0, "review_scope": "Fixture", "reason": "Exact reference beyond preview"}
    class Runner:
        def run(self, command, **kwargs):
            saved = json.loads((kwargs["cwd"] / "input.json").read_text())
            schema = saved["schema"]
            assert saved["inputs"]["task"]["candidate_ids"] == ids
            accepted = schema["properties"]["accepted_ids"]
            edit = schema["$defs"]["MaterialEdit"]["properties"]["candidate_id"]
            if count == 65:
                assert schema["$defs"]["MaterialCandidateId"]["enum"] == sorted(ids)
                assert accepted["items"] == edit == {"$ref": "#/$defs/MaterialCandidateId"}
                # Replacement IDs remain teacher-authored rather than references.
                assert "enum" not in schema["$defs"]["Candidate"]["properties"]["id"]
            elif not count:
                assert accepted["maxItems"] == schema["properties"]["edits"]["maxItems"] == 0
            else:
                assert "MaterialCandidateId" not in schema["$defs"]
                assert "enum" not in accepted["items"] and "enum" not in edit
            if provider == "codex":
                assert json.loads(Path(command[command.index("--output-schema") + 1]).read_text()) == schema
                Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(choice))
                return ""
            assert json.loads(command[command.index("--json-schema") + 1]) == schema
            return json.dumps({"structured_output": choice})
    config = CampaignConfig.model_validate({**campaign["config"], "teacher_provider": provider})
    teacher = CliTeacher(config, settings, service.store, campaign["id"], row["id"], Runner(), settings.workspace / "candidate-adapter")
    assert teacher.select_materials(brief) == MaterialSelection.model_validate(choice).model_dump()
    assert "MaterialCandidateId" not in MaterialSelection.model_json_schema()["$defs"]


@pytest.mark.parametrize("failure", ["unknown_edit", "unknown_accept", "duplicate_accept", "duplicate_edit", "overlap"])
def test_candidate_reference_errors_preserve_artifacts_and_retry_jobs(setup_loop, failure):
    calls = []
    class BadReference(AuthorTeacher):
        def select_materials(self, manifest):
            choice = super().select_materials(manifest)
            key = manifest["candidate_ids"][0]
            edit = {"candidate_id": key, "replacement": {"id": "teacher_replacement", "kind": "cpt",
                    "concept": "Fixture correction", "training_text": "Corrected teaching text.",
                    "training_tokenization": "full_text", "rationale": "Teacher correction"}}
            if not calls:
                if failure == "unknown_edit":
                    edit["candidate_id"] = key + "_missing"
                    choice["edits"] = [edit]
                elif failure == "unknown_accept": choice["accepted_ids"] = [key + "_missing"]
                elif failure == "duplicate_accept": choice["accepted_ids"] = [key, key]
                elif failure == "duplicate_edit": choice["edits"] = [edit, edit]
                else: choice.update(accepted_ids=[key], edits=[edit])
            else:
                choice["edits"] = [edit]
            calls.append(choice)
            return choice
    _, service, campaign, engine = configured(setup_loop, teacher=BadReference)
    engine.run(campaign["id"])
    failed = service.store.campaign(campaign["id"])
    assert failed["status"] == "failed"
    field = {"unknown_edit": "unknown edits.candidate_id", "unknown_accept": "unknown accepted_ids",
             "duplicate_accept": "duplicate accepted_ids", "duplicate_edit": "duplicate edits.candidate_id",
             "overlap": "accepted/edited overlap"}[failure]
    bad_key = "material-one:variant" + ("_missing" if failure.startswith("unknown") else "")
    assert f"{field}: ['{bad_key}']" in failed["error"]
    assert not FakeModel.datasets
    original = service.store.one("SELECT artifact FROM stage_runs WHERE stage='expand' AND status='complete'")["artifact"]
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert len(service.store.query("SELECT id FROM material_calls")) == 2
    assert service.store.one("SELECT artifact FROM stage_runs WHERE stage='expand' AND status='complete'")["artifact"] == original


def test_request_deadline_saves_failure_without_fake_completion(setup_loop):
    async def handler(req):
        await asyncio.sleep(3)
        return response(req)
    _, service, campaign, engine = configured(setup_loop, handler, pool=AuthorPool(authors=[author(timeout_seconds=1)]))
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert "TimeoutError" in service.store.campaign(campaign["id"])["error"]
    assert not service.store.query("SELECT id FROM material_jobs WHERE status='complete'")


def test_material_api_pagination_has_no_content_cutoff(setup_loop):
    from fastapi.testclient import TestClient
    from nekaise_loop.api import create_app
    settings, service, campaign, engine = configured(setup_loop)
    engine.run(campaign["id"])
    detail = service.snapshot(campaign["id"])["round"]
    originals = detail["materials"]
    rows = [{**originals[0], "id": f"material-{n}"} for n in range(45)]
    service.store.execute("DELETE FROM records WHERE round_id=? AND kind='material'", (detail["id"],))
    service.store.put_records(detail["id"], "material", rows)
    client = TestClient(create_app(settings))
    initial = client.get(f"/api/rounds/{detail['id']}").json()
    assert initial["material_count"] == 45 and len(initial["materials"]) == 20
    page = client.get(f"/api/rounds/{detail['id']}/materials?offset=20&limit=20").json()
    assert page["next_offset"] == 40 and page["rows"][0]["id"] == "material-20"
    last = client.get(f"/api/rounds/{detail['id']}/materials?offset=40&limit=20").json()
    assert len(last["rows"]) == 5 and last["next_offset"] is None


def test_malformed_envelope_and_missing_usage_preserve_evidence(setup_loop):
    _, service, campaign, engine = configured(setup_loop, lambda req: httpx.Response(200, content=b"bad body"))
    engine.run(campaign["id"])
    raw = service.store.one("SELECT artifact FROM material_calls WHERE artifact IS NOT NULL")
    assert service.artifacts.get(raw["artifact"])["response"]["raw_body"] == "bad body"
    assert service.store.campaign(campaign["id"])["status"] == "failed"


def test_local_serving_endpoint_uses_real_http_transport(setup_loop):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    paths = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            paths.append(self.path)
            body = self.rfile.read(int(self.headers['Content-Length']))
            payload = response(httpx.Request('POST', 'http://fixture.invalid', content=body)).content
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        local = author(location='local').model_copy(update={'base_url': f'http://127.0.0.1:{server.server_port}/v1'})
        _, service, campaign, engine = configured(setup_loop, pool=AuthorPool(authors=[local]))
        engine.material_client_factory = None
        engine.run(campaign['id'])
        assert service.store.campaign(campaign['id'])['status'] == 'complete'
        assert paths == ['/v1/chat/completions', '/v1/chat/completions']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_distinct_job_candidate_pairs_cannot_collapse_during_selection(setup_loop):
    class DistinctJobs(AuthorTeacher):
        jobs = [("a-b", "a"), ("a", "a")]

    def handler(request):
        envelope = response(request).json()
        task = json.loads(json.loads(request.content)["messages"][1]["content"])["task"]
        batch = json.loads(envelope["choices"][0]["message"]["content"])
        batch["rows"][0]["id"] = "c" if task["job"]["id"] == "a-b" else "b-c"
        envelope["choices"][0]["message"]["content"] = json.dumps(batch)
        return httpx.Response(200, json=envelope)

    _, service, campaign, engine = configured(setup_loop, handler, teacher=DistinctJobs)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    rows = service.snapshot(campaign["id"])["round"]["materials"]
    assert len(rows) == 2
    assert len({row["id"] for row in rows}) == 2
    assert len({row["material_origin"]["job_id"] for row in rows}) == 2
    assert all(row["use_for_training"] for row in rows)


def test_required_training_rejects_empty_expansion_before_student_work(setup_loop):
    _, service, campaign, engine = configured(setup_loop, teacher=FakeTeacher)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert "requires Material Author expansion_jobs" in service.store.campaign(campaign["id"])["error"]
    assert not FakeModel.prompts and not FakeModel.datasets


def test_required_diagnostic_without_expansion_preserves_weights(setup_loop):
    class Diagnostic(FakeTeacher):
        def curriculum(self, brief):
            plan = super().curriculum(brief)
            plan["train_epochs"] = 0
            return plan
    _, service, campaign, engine = configured(setup_loop, teacher=Diagnostic)
    engine.run(campaign["id"])
    detail = service.snapshot(campaign["id"])["round"]
    assert detail["status"] == "complete" and detail["checkpoint"] == detail["model_before"]
    assert job_work(service.store, detail["id"]) is None
    assert not FakeModel.datasets


@pytest.mark.parametrize("mode", ["reject_all", "zero_teacher_share", "empty_batch"])
def test_required_training_cannot_bypass_expanded_target_consumption(setup_loop, mode):
    class Selected(AuthorTeacher):
        def select_materials(self, manifest):
            choice = super().select_materials(manifest)
            if mode == "reject_all": choice["accepted_jobs"] = []
            if mode == "zero_teacher_share": choice["token_mix"] = {"teacher": 0, "corpus": 1, "replay": 0}
            return choice
    def handler(req):
        payload = response(req).json()
        if mode == "empty_batch": payload["choices"][0]["message"]["content"] = '{"rows": []}'
        return httpx.Response(200, json=payload)
    _, service, campaign, engine = configured(setup_loop, handler, teacher=Selected)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert not FakeModel.datasets
    assert "teacher-selected expanded training targets" in service.store.campaign(campaign["id"])["error"]


def test_required_receipt_binds_frozen_expansion_to_round_and_targets(setup_loop):
    _, service, campaign, engine = configured(setup_loop)
    engine.run(campaign["id"])
    detail = service.snapshot(campaign["id"])["round"]
    assert detail["status"] == "complete"
    receipt = detail["learning_work"]["material_expansion"]
    assert receipt["round_id"] == detail["id"]
    assert receipt["policy"] == "required_v1"
    assert receipt["produced_candidates"] == receipt["selected_candidates"] == 2
    assert receipt["prepared_expanded_targets_per_pass"] > 0


def test_train_rechecks_receipt_and_effective_training_choice(setup_loop):
    from types import SimpleNamespace
    from nekaise_loop.stages import train
    _, service, campaign, engine = configured(setup_loop)
    engine.run(campaign["id"])
    detail = service.snapshot(campaign["id"])["round"]
    outputs = {s["stage"]: service.artifacts.get(s["artifact"]) for s in detail["stages"] if s["status"] == "complete"}
    context = SimpleNamespace(config=CampaignConfig.model_validate(campaign["config"]), round=detail, output=outputs.__getitem__)
    outputs["freeze"]["material_expansion"]["round_id"] = "another-round"
    with pytest.raises(ValueError, match="receipt does not match"):
        train(context)
    outputs["material_select"]["curriculum"]["expansion_jobs"] = []
    with pytest.raises(ValueError, match="requires Material Author expansion_jobs"):
        train(context)
