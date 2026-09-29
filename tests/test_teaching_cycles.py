"""Real buffered control plane with explicitly fake model, web and Author transports."""
import json
import shutil
import threading
from pathlib import Path

import pytest

from test_progressive_curriculum import configured, ProgressiveTeacher, ProgressiveModel
from nekaise_loop.cycle_store import cycle_for
from nekaise_loop.material_allowance import allowance_totals
from nekaise_loop.author_config import AuthorPool
from nekaise_loop.storage import now


class CycleTeacher(ProgressiveTeacher):
    calls = []
    review_failure = False

    def cycle_research(self, brief):
        self.calls.append("research")
        return {"units": [{"unit_id": u["id"], "plan": self.research({"unit": u})} for u in brief["units"]]}

    def cycle_plan(self, brief):
        self.calls.append("plan")
        blocks = []
        for unit in brief["units"]:
            plan = self.curriculum({"round_number": 1, "progression": {"unit": unit, "required_authors": brief["required_authors"]}})
            plan["raw_target_tokens"] = 512
            blocks.append({"unit_id": unit["id"], "curriculum": plan, "revisions": self.revise(plan["lessons"])})
        return {"blocks": blocks, "assessments": self.evaluate({}, []), "reason": "Fixture cycle"}

    def cycle_review(self, brief):
        self.calls.append("review")
        if self.review_failure:
            raise ValueError("Fixture review interruption")
        return {"grades": self.grade(brief["items"]), "reflection": self.reflect(brief)}


class CycleModel(ProgressiveModel):
    hook = None

    def train(self, checkpoint, dataset, dataset_hash, on_metric):
        if type(self).hook:
            type(self).hook(self, checkpoint)
        result = super().train(checkpoint, dataset, dataset_hash, on_metric)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(Path(checkpoint)/name, Path(result["checkpoint"])/name)
        return result


def setup_cycle(setup_loop, monkeypatch, **updates):
    settings, service, campaign, engine, calls = configured(setup_loop, monkeypatch,
        teaching_cycle=updates.pop("teaching_cycle", {"initial_blocks_per_cycle":2,"blocks_per_cycle":4}), **updates)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (Path(campaign["config"]["student_model"])/name).write_text('{"fixture":true}')
    CycleTeacher.calls = []
    CycleTeacher.review_failure = False
    CycleModel.hook = None
    engine.teacher_factory = CycleTeacher
    engine.model_factory = CycleModel
    return settings, service, campaign, engine, calls


def test_cycle_saves_each_block_all_authors_and_reviews_once(setup_loop, monkeypatch):
    _, service, campaign, engine, calls = setup_cycle(setup_loop, monkeypatch, rounds=2)
    engine.run(campaign["id"])
    current = service.store.campaign(campaign["id"])
    assert current["status"] == "complete", current["error"]
    assert CycleTeacher.calls == ["research", "plan", "review"]
    assert len(calls) == 6
    rows = service.store.query("SELECT * FROM rounds WHERE campaign_id=? ORDER BY number", (campaign["id"],))
    assert rows[1]["model_before"] == rows[0]["checkpoint"]
    snapshot = service.snapshot(campaign["id"])
    assert snapshot["curriculum_progress"]["state"]["gpc_completed"] == 2
    assert [r["score"] for r in reversed(snapshot["rounds"])] == [None, .25]
    assert len(service.store.query("SELECT * FROM curriculum_receipts")) == 2
    for row in rows:
        cycle = cycle_for(service.store, row["id"])
        assert service.artifacts.get(cycle["parent_binding_artifact"])["parent_checkpoint"] == row["model_before"]


def test_cycle_assessment_resolves_previous_campaign_research(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=2)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    old_cycle = service.store.one("SELECT * FROM teaching_cycles WHERE campaign_id=?", (campaign["id"],))
    old_research = service.artifacts.get(old_cycle["research_artifact"])
    original = CycleTeacher.cycle_plan

    # The fixture Author requires research-fixture on its current jobs. Give the
    # archived research an additional distinct reference for the later assessment.
    historical = {**old_research["units"][0]["sources"][0], "id": "research-historical-citation"}
    old_research["units"][0]["sources"].append(historical)
    old_cycle["research_artifact"] = service.artifacts.put(old_research)
    service.store.execute("UPDATE teaching_cycles SET research_artifact=? WHERE id=?",
                          (old_cycle["research_artifact"], old_cycle["id"]))

    def plan(self, brief):
        value = original(self, brief)
        value["assessments"][0]["sources"] = [{"document_id": "research-historical-citation", "start": 8, "length": 12}]
        return value

    monkeypatch.setattr(CycleTeacher, "cycle_plan", plan)
    child = service.continue_campaign(campaign["id"], {"rounds": 2})
    engine.run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete", service.store.campaign(child["id"])["error"]
    saved = service.store.one("SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id "
                             "WHERE r.campaign_id=? AND s.stage='evaluate' ORDER BY r.number DESC LIMIT 1", (child["id"],))
    source = service.artifacts.get(saved["artifact"])["items"][0]["sources"][0]
    assert source["text"] == old_research["units"][0]["sources"][0]["text"][8:20]
    assert source["research_artifact"] == old_cycle["research_artifact"]
    assert source["source_sha256"] == old_research["units"][0]["sources"][0]["source_sha256"]
    assert service.artifacts.get(old_cycle["research_artifact"]) == old_research


@pytest.mark.parametrize("failure", ["unknown_id", "outside_span", "tampered_artifact"])
def test_historical_assessment_reference_checks_remain_enforced(setup_loop, monkeypatch, failure):
    from types import SimpleNamespace
    from nekaise_loop.cycle_stages import assessment_sources
    from nekaise_loop.stages import _sources
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=2)
    engine.run(campaign["id"])
    old = service.store.one("SELECT research_artifact FROM teaching_cycles WHERE campaign_id=?", (campaign["id"],))
    current = {"research_artifact": service.artifacts.put({"units": []})}
    reference = {"document_id": "research-unknown" if failure == "unknown_id" else "research-fixture",
                 "start": 0, "length": 1000000 if failure == "outside_span" else 0}
    ctx = SimpleNamespace(store=service.store, artifacts=service.artifacts,
                          config=SimpleNamespace(curriculum_loop=None, corpus_path="."), engine=engine)
    if failure == "tampered_artifact":
        path = service.artifacts.root / old["research_artifact"][:2] / (old["research_artifact"] + ".json")
        path.write_text('{}')
        with pytest.raises(ValueError, match="Artifact integrity failed"):
            assessment_sources(ctx, current, [{"sources": [reference]}])
    else:
        ctx.additional_sources = assessment_sources(ctx, current, [{"sources": [reference]}])
        def rejected(*args):
            raise ValueError("Unresolved reference reached ordinary source admission")
        monkeypatch.setattr("nekaise_loop.stages.read_source", rejected)
        with pytest.raises(ValueError, match="Unresolved reference"):
            _sources(ctx, [reference])


def test_author_preparation_overlaps_gpu_without_claiming_future_coverage(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=4)
    observed = []
    def hook(model, parent):
        if observed:
            return
        import time
        deadline = time.monotonic()+5
        while time.monotonic()<deadline:
            prepared = service.store.one("SELECT r.id FROM rounds r JOIN teaching_blocks b ON b.round_id=r.id WHERE b.position=1 AND r.status='prepared'")
            if prepared:
                break
            time.sleep(.02)
        assert prepared, "Second block should freeze while first GPU call is active"
        observed.append(prepared["id"])
        assert len(service.store.query("SELECT * FROM curriculum_assignments")) == 1
        assert not service.store.query("SELECT * FROM curriculum_receipts")
        assert service.store.one("SELECT model_before FROM rounds WHERE id=?", (prepared["id"],))["model_before"] == ""
    CycleModel.hook = hook
    engine.run(campaign["id"])
    current = service.store.campaign(campaign["id"])
    assert current["status"] == "complete", current["error"]
    assert observed
    assert len(service.store.query("SELECT * FROM curriculum_receipts")) == 4


def test_review_retry_preserves_saves_progress_and_author_budgets(setup_loop, monkeypatch):
    _, service, campaign, engine, calls = setup_cycle(setup_loop, monkeypatch, rounds=2)
    CycleTeacher.review_failure = True
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert len(service.store.query("SELECT * FROM curriculum_receipts")) == 2
    before = service.store.query("SELECT * FROM material_calls")
    service.store.execute("UPDATE campaigns SET teacher_budget_since=? WHERE id=?", (now(), campaign["id"]))
    rows = service.store.query("SELECT id FROM rounds WHERE campaign_id=? ORDER BY number", (campaign["id"],))
    with service.store.connect() as db:
        allowance = allowance_totals(db, rows[0]["id"], now(), AuthorPool.model_validate(campaign["config"]["material_authors"]))
    assert allowance["used_calls"] == 3 and allowance["cycle"]["used_calls"] == 6
    CycleTeacher.review_failure = False
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete", service.store.campaign(campaign["id"])["error"]
    assert service.store.query("SELECT * FROM material_calls") == before
    assert len(calls) == 6 and len(service.store.query("SELECT * FROM curriculum_receipts")) == 2
    assert CycleTeacher.calls == ["research", "plan", "review", "review"]


def test_future_prepare_failure_reports_exact_stage_after_current_save(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=2, auto_recover=True)
    import nekaise_loop.cycle_engine as runner
    original = runner.execute
    entered = threading.Event()
    def execute(engine, campaign, round_id, stage, cancelled):
        position = cycle_for(engine.store, round_id)["position"]
        if position == 1 and stage == "freeze":
            entered.wait(5)
            # Raise inside the real stage so the exact failure ID is recorded.
        return original(engine, campaign, round_id, stage, cancelled)
    monkeypatch.setattr(runner, "execute", execute)
    import nekaise_loop.stages as stages
    freeze = stages.freeze
    def fail(ctx):
        if cycle_for(ctx.store, ctx.round["id"])["position"] == 1:
            raise ValueError("Fixture future freeze failure")
        return freeze(ctx)
    monkeypatch.setattr(stages, "freeze", fail)
    CycleModel.hook = lambda *a: entered.set()
    engine.run(campaign["id"])
    recovery = service.store.one("SELECT * FROM recoveries WHERE campaign_id=?", (campaign["id"],))
    assert recovery
    failed = service.store.one("SELECT * FROM stage_runs WHERE id=?", (recovery["stage_id"],))
    assert failed["stage"] == "freeze" and failed["status"] == "failed"
    assert len(service.store.query("SELECT * FROM curriculum_receipts")) == 1
    assert service.store.one("SELECT status FROM rounds WHERE campaign_id=? AND number=1", (campaign["id"],))["status"] == "complete"


def test_prefetch_is_bounded_and_stop_does_not_credit_preparation(setup_loop, monkeypatch):
    _, service, campaign, engine, calls = setup_cycle(setup_loop, monkeypatch, rounds=4,
        teaching_cycle={"initial_blocks_per_cycle":4})
    stopped = threading.Event()
    def hook(model, parent):
        import time
        deadline = time.monotonic()+5
        prepared = None
        while time.monotonic()<deadline:
            prepared = service.store.one("SELECT id FROM rounds WHERE campaign_id=? AND number=3 AND status='prepared'", (campaign["id"],))
            if prepared:
                break
            time.sleep(.02)
        assert prepared
        assert service.store.one("SELECT status FROM rounds WHERE campaign_id=? AND number=4", (campaign["id"],))["status"] == "ready"
        assert len(calls) == 9
        stopped.set()
    CycleModel.hook = hook
    engine.run(campaign["id"], controls=stopped.is_set)
    assert service.store.campaign(campaign["id"])["status"] == "stopped"
    assert len(service.store.query("SELECT * FROM curriculum_receipts")) == 1
    assert len(service.store.query("SELECT * FROM curriculum_assignments")) == 1
    assert service.snapshot(campaign["id"])["round"]["number"] == 1
    CycleModel.hook = None
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete", service.store.campaign(campaign["id"])["error"]
    assert len(calls) == 12


def test_changed_parent_tokenizer_stops_before_next_gpu_update(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=2)
    original = CycleModel.train
    def corrupt(model, *args):
        result = original(model, *args)
        (Path(result["checkpoint"])/"tokenizer.json").write_text('{"changed":true}')
        return result
    monkeypatch.setattr(CycleModel, "train", corrupt)
    engine.run(campaign["id"])
    result = service.store.campaign(campaign["id"])
    assert result["status"] == "failed" and "tokenizer differs" in result["error"]
    assert len(service.store.query("SELECT * FROM curriculum_receipts")) == 1


def test_recovery_cannot_renew_cycle_by_noop_continuation(setup_loop, monkeypatch):
    from nekaise_loop.service import Conflict
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=2)
    CycleTeacher.review_failure = True
    engine.run(campaign["id"])
    with pytest.raises(Conflict, match="no-op continuation"):
        service.continue_campaign(campaign["id"], actor="orchestrator")
    with pytest.raises(Conflict, match="preserve.*buffered cycle"):
        service.continue_campaign(campaign["id"], {"teaching_cycle":None}, actor="orchestrator")


@pytest.mark.parametrize("saved_reference", ["assignment", "legacy_block", "other_assignment", "wrong_unit"])
def test_continuation_reuses_fetched_research_for_pending_assignment(setup_loop, monkeypatch, saved_reference):
    import nekaise_loop.stages as stages
    import nekaise_loop.curriculum_research as web
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=1,
        teaching_cycle={"initial_blocks_per_cycle": 1})
    freeze = stages.freeze
    def fail(ctx):
        raise ValueError("Fixture preparation interruption")
    monkeypatch.setattr(stages, "freeze", fail)
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    block = service.store.one("SELECT * FROM teaching_blocks")
    assignment = service.store.one("SELECT * FROM curriculum_assignments")
    assert assignment["research_artifact"] == block["research_artifact"]
    assert not service.store.query("SELECT * FROM curriculum_receipts")
    if saved_reference != "assignment":
        service.store.execute("UPDATE curriculum_assignments SET research_artifact=NULL")
    if saved_reference == "other_assignment":
        other = service.artifacts.get(block["assignment_artifact"])
        other["namespace"] = "different-frontier"
        service.store.execute("UPDATE teaching_blocks SET assignment_artifact=?",
                              (service.artifacts.put(other),))
    if saved_reference == "wrong_unit":
        other = service.artifacts.get(block["research_artifact"])
        other["unit_id"] = "different.unit"
        service.store.execute("UPDATE teaching_blocks SET research_artifact=?",
                              (service.artifacts.put(other),))
    child = service.continue_campaign(campaign["id"], {"workload_guidance": "Fixture repaired preparation"}, start=False)
    monkeypatch.setattr(stages, "freeze", freeze)
    def unavailable(*args, **kwargs):
        raise AssertionError("Continuation must reuse fetched references without another web request")
    monkeypatch.setattr(web, "fetch_source", unavailable)
    engine.run(child["id"])
    result = service.store.campaign(child["id"])
    if saved_reference in {"other_assignment", "wrong_unit"}:
        assert result["status"] == "failed"
        expected = "another web request" if saved_reference == "other_assignment" else "different unit"
        assert expected in result["error"]
        assert not service.store.query("SELECT * FROM curriculum_receipts")
        return
    assert result["status"] == "complete", result["error"]
    assert CycleTeacher.calls.count("research") == 1
    next_block = service.store.one("SELECT b.* FROM teaching_blocks b JOIN rounds r ON r.id=b.round_id WHERE r.campaign_id=?", (child["id"],))
    assert next_block["assignment_artifact"] == block["assignment_artifact"]
    assert next_block["research_artifact"] == block["research_artifact"]
    assert len(service.store.query("SELECT * FROM curriculum_receipts")) == 1


def test_explicit_raw_allocation_does_not_shrink_with_small_author_yield():
    from nekaise_loop.progressive_preparation import prepare_progressive
    from nekaise_loop.config import CampaignConfig
    from conftest import FakeTokenizer
    config = CampaignConfig().model_dump()
    config["curriculum_loop"] = {"forward_corpus_share":.5,"remediation_cap":.2,"raw_target_tokens":3000}
    rows = [{"id":"g","stream":"cpt","text":"g"*50,"learning_track":"gpc","curriculum_unit_id":"u"},
        *[{"id":str(i),"stream":"corpus","text":"c"*500,"learning_track":"corpus","curriculum_span":True} for i in range(10)]]
    frozen = prepare_progressive(rows, FakeTokenizer(), config)
    assert frozen["progressive_preparation"]["raw_targets_prepared"] >= 3000
    assert frozen["progressive_preparation"]["raw_allocation_basis"] == "teacher_explicit_target"
    assert all(s["repeated_tokens"] == 0 for s in frozen["ledger"]["streams"].values())


def test_cycle_caps_include_unknown_reservations_and_reported_overruns(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = setup_cycle(setup_loop, monkeypatch, rounds=2,
        teaching_cycle={"initial_blocks_per_cycle":2,"max_author_calls_per_cycle":6,"max_author_output_tokens_per_cycle":3072})
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    calls = service.store.query("SELECT id FROM material_calls ORDER BY id")
    service.store.execute("UPDATE material_calls SET usage='{}',status='uncertain' WHERE id=?", (calls[0]["id"],))
    service.store.execute("UPDATE material_calls SET usage=? WHERE id=?", (json.dumps({"completion_tokens":600}), calls[1]["id"]))
    row = service.store.one("SELECT id FROM rounds WHERE campaign_id=? ORDER BY number DESC LIMIT 1", (campaign["id"],))
    with service.store.connect() as db:
        total = allowance_totals(db, row["id"], now(), AuthorPool.model_validate(campaign["config"]["material_authors"]))
    assert total["remaining_calls"] == 0
    assert total["remaining_output_tokens"] == -88
    assert total["cycle"]["reserved_tokens"] == 3072
    assert total["cycle"]["reported_output_overrun_tokens"] == 88


def test_compact_teacher_view_keeps_exact_original_references():
    from nekaise_loop.config import CampaignConfig
    from nekaise_loop.cycle_context import compact_inputs
    source = {"id":"research-test","text":"x"*30000,"url":"https://example.org"}
    inputs = {"config_hints":CampaignConfig().model_dump(), "latest_strategy":{"student_notes":"Observed note","large_history":"z"*50000},
        "operations":{}, "current":{}, "task":{"research":[{"unit_id":"unit","sources":[source]}]}}
    result = compact_inputs(inputs)
    assert result["latest_strategy"]["student_notes"] == "Observed note"
    assert result["complete_strategy"]["$evidence"]["pointer"] == "/latest_strategy"
    preview = result["task"]["research"][0]["sources"][0]
    assert len(preview["preview"]) == 1600
    assert preview["complete_text"]["$evidence"]["pointer"] == "/task/research/0/sources/0/text"
    assert source["text"] == "x"*30000
