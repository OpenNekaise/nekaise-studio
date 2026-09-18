"""Fixture-only checks: intent, evidence and browsing never select teaching."""
import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeModel, FakeTeacher
from nekaise_loop.api import create_app
from nekaise_loop.engine import Engine
from nekaise_loop.config import CampaignConfig
from nekaise_loop.experiments import catalog, detail, latest_context, rebuild_index
from nekaise_loop.teacher_tools import query


def plan():
    return {"title": "Fixture comparison", "hypothesis": "Contrasting examples may help",
        "intervention": "Explain commonly confused units", "budget_basis": "Sequential observation, unmatched budgets",
        "observation_plan": "Inspect same-question evidence if useful", "reconsider_if": "No transfer observed",
        "related_round_ids": [], "strategy_version": "",
        "strategy": {"name": "Contrastive examples", "approach": "Explain distinctions", "parent_version": ""}}


class ExperimentTeacher(FakeTeacher):
    briefs = []
    observations = []

    def curriculum(self, brief):
        self.briefs.append(brief)
        proposal = plan()
        if brief["latest_experiment"]:
            previous = brief["latest_experiment"]
            proposal.update(strategy=None, strategy_version=previous["strategy_version"],
                            related_round_ids=[previous["round_id"]])
        return {**super().curriculum(brief), "experiment": proposal}

    def reflect(self, observations):
        self.observations.append(observations)
        return {**super().reflect(observations), "experiment_review": {
            "status": "concluded", "findings": "No improvement established in this fixture",
            "limitations": "No matched-start control", "next_action": "Reconsider the teaching approach",
            "evaluation_ids": [observations["assessment"]["items"][0]["id"]]}}


def execute(setup_loop, teacher=ExperimentTeacher, model=FakeModel):
    settings, service, campaign, _ = setup_loop
    ExperimentTeacher.briefs, ExperimentTeacher.observations = [], []
    engine = Engine(settings, teacher, model)
    engine.run(campaign["id"])
    return settings, service, campaign, engine


def test_versions_reuse_pretraining_intent_and_link_real_work(setup_loop):
    _, service, campaign, engine = execute(setup_loop)
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    rows = catalog(service.store)["items"]
    assert len(rows) == 2 and rows[0]["strategy_version"] == rows[1]["strategy_version"]
    assert ExperimentTeacher.briefs[0]["latest_experiment"] is None
    first = detail(service.store, service.artifacts, rows[1]["round_id"])
    second = detail(service.store, service.artifacts, rows[0]["round_id"])
    assert second["plan"]["related_round_ids"] == [first["id"]]
    assert second["starting_point"]["parent_train_artifact"] == first["artifacts"]["train"]["hash"]
    assert first["review"]["findings"].startswith("No improvement")
    assert first["learning_work"]["measured_work_all_attempts"]["tokens"] == 300
    assert first["learning_work"]["teacher_efficiency"]["ratio"] is None
    for stage in ("freeze", "train", "adapt"):
        saved = engine.artifacts.get(first["artifacts"][stage]["hash"])
        assert saved["experiment_plan_artifact"] == first["artifacts"]["select"]["hash"]
    selected = engine.artifacts.get(first["artifacts"]["select"]["hash"])
    assert "review" not in selected["experiment"]
    assert selected["experiment"]["plan"]["hypothesis"] == plan()["hypothesis"]
    assert ExperimentTeacher.observations[0]["experiment"]["review"] is None
    assert first["assessment"] == {"status": "complete", "question_count": 2, "paired": [], "requested_pairs": 0}


def test_new_definition_changes_version_but_keeps_parent(setup_loop):
    settings, service, parent, _ = execute(setup_loop)
    prior = catalog(service.store)["items"][0]["strategy_version"]
    child = service.continue_campaign(parent["id"], {"rounds": 1})

    class Revised(ExperimentTeacher):
        def curriculum(self, brief):
            result = super().curriculum(brief)
            result["experiment"].update(strategy_version="", strategy={
                "name": "Revised approach", "approach": "Add transfer examples", "parent_version": prior})
            return result

    Engine(settings, Revised, FakeModel).run(child["id"])
    card = detail(service.store, service.artifacts, catalog(service.store)["items"][0]["round_id"])
    assert card["strategy_version"] != prior and card["strategy"]["parent_version"] == prior
    assert len(catalog(service.store, strategy_version=prior)["items"]) == 2


def test_failed_round_preserves_plan_without_inventing_review_and_retry_is_idempotent(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Interrupted(ExperimentTeacher):
        fail = True
        def reflect(self, observations):
            if self.fail:
                raise RuntimeError("Fixture interruption before conclusion")
            return super().reflect(observations)
    engine = Engine(settings, Interrupted, FakeModel)
    engine.run(campaign["id"])
    entry = catalog(service.store)["items"][0]
    card = detail(service.store, engine.artifacts, entry["round_id"])
    assert card["review"] is None and card["execution"]["status"] == "failed"
    assert card["assessment"]["status"] == "complete"
    Interrupted.fail = False
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert service.store.records(entry["round_id"], "experiment")[0]["select_artifact"] == entry["select_artifact"]
    assert len(catalog(service.store)["items"]) == 2


def test_index_is_atomic_and_rebuilds_only_recorded_plans(setup_loop, monkeypatch):
    import nekaise_loop.experiments as experiments
    settings, service, campaign, _ = setup_loop
    original = experiments.index_selection
    def fail_after_insert(*args):
        original(*args)
        raise RuntimeError("Fixture interrupted commit")
    monkeypatch.setattr(experiments, "index_selection", fail_after_insert)
    Engine(settings, ExperimentTeacher, FakeModel).run(campaign["id"])
    assert catalog(service.store)["items"] == []
    assert not service.store.one("SELECT id FROM stage_runs WHERE stage='select' AND status='complete'")
    monkeypatch.setattr(experiments, "index_selection", original)
    Engine(settings, ExperimentTeacher, FakeModel).run(campaign["id"])
    before = catalog(service.store)
    assert len(before["items"]) == 2
    service.store.execute("DELETE FROM records WHERE kind='experiment'")
    assert rebuild_index(service.store, service.artifacts) == {"indexed_experiments": 2}
    assert catalog(service.store) == before
    assert rebuild_index(service.store, service.artifacts) == {"indexed_experiments": 2}


@pytest.mark.parametrize("fault", ["version", "parent", "related", "review", "both"])
def test_invalid_references_fail_structurally_without_fabricating_evidence(setup_loop, fault):
    class Bad(ExperimentTeacher):
        def curriculum(self, brief):
            result = super().curriculum(brief)
            proposed = result["experiment"]
            if fault == "version":
                proposed.update(strategy=None, strategy_version="f"*64)
            elif fault == "parent":
                proposed["strategy"]["parent_version"] = "f"*64
            elif fault == "related":
                proposed["related_round_ids"] = [brief["round_id"]]
            elif fault == "both":
                proposed["strategy_version"] = "f"*64
            return result
        def reflect(self, observations):
            result = super().reflect(observations)
            result["experiment_review"]["evaluation_ids"] = ["independent-bench-question"]
            return result
    _, service, campaign, _ = execute(setup_loop, Bad)
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    if fault == "review":
        entry = catalog(service.store)["items"][0]
        assert detail(service.store, service.artifacts, entry["round_id"])["review"] is None
    else:
        assert catalog(service.store)["items"] == []


def test_optional_legacy_rounds_are_not_backfilled_as_experiments(setup_loop):
    _, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    rnd = service.store.one("SELECT id FROM rounds WHERE campaign_id=?", (campaign["id"],))
    assert detail(service.store, service.artifacts, rnd["id"]) is None
    assert service.round_detail(rnd["id"])["experiment"] is None
    assert rebuild_index(service.store, service.artifacts)["indexed_experiments"] == 0


def test_unavailable_experiment_evidence_is_explicit_in_dashboard(setup_loop, monkeypatch):
    settings, service, campaign, engine = execute(setup_loop)
    entry = catalog(service.store)["items"][0]
    selected = engine.artifacts.get(entry["select_artifact"])
    selected["experiment"]["strategy_version"] = "f"*64
    service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?",
                          (engine.artifacts.put(selected), entry["select_stage_id"]))
    assert "unavailable" in service.round_detail(entry["round_id"])["experiment"]["error"]
    app = create_app(settings)
    monkeypatch.setattr(app.state.service, "ensure_worker", lambda: None)
    with TestClient(app) as client:
        result = client.get(f"/api/experiments/{entry['round_id']}")
        assert result.status_code == 409
        assert "unavailable" in result.json()["detail"]


@pytest.mark.parametrize("matched", [True, False])
def test_paired_evidence_reads_completed_artifact_not_mutable_projection(setup_loop, matched):
    from test_online_comparison import PairedModel, PairedTeacher
    settings, service, parent, engine = setup_loop
    engine.run(parent["id"])
    child = service.continue_campaign(parent["id"], {"rounds": 1})
    class Model(PairedModel):
        def compare(self, *args, **kwargs):
            result = super().compare(*args, **kwargs)
            if not matched:
                result["reference"][0]["runtime"] = {"dtype": "different"}
            return result
    class Teacher(ExperimentTeacher, PairedTeacher):
        grading_inputs = []
    Engine(settings, Teacher, Model).run(child["id"])
    entry = catalog(service.store)["items"][0]
    card = detail(service.store, service.artifacts, entry["round_id"])
    row = card["assessment"]["paired"][0]
    assert row["score_delta"] == (pytest.approx(.6) if matched else None)
    assert row["comparable"] is matched
    service.store.execute("DELETE FROM records WHERE round_id=? AND kind='evaluation'", (entry["round_id"],))
    assert detail(service.store, service.artifacts, entry["round_id"])["assessment"] == card["assessment"]


def test_api_and_readonly_archive_page_across_campaigns_and_preserve_bench_boundary(setup_loop, monkeypatch):
    settings, service, campaign, engine = execute(setup_loop)
    first = catalog(service.store, limit=1)
    child = service.continue_campaign(campaign["id"], {"rounds": 1})
    Engine(settings, ExperimentTeacher, FakeModel).run(child["id"])
    older = catalog(service.store, before=first["next_before"], limit=1)
    assert older["items"][0]["round_number"] == 1 and older["next_before"] is None
    context = {"workspace": str(settings.workspace), "campaign_id": child["id"], "corpus_path": campaign["config"]["corpus_path"]}
    assert query(context, {"op": "experiments"}) == catalog(service.store)
    card = query(context, {"op": "experiment", "round_id": first["items"][0]["round_id"]})
    assert card["review"] and card["learning_work"]
    assert query(context, {"op": "strategy", "version": card["strategy_version"]})["definition"] == card["strategy"]
    assert not any("benchmark" in key for key in card)
    def no_bench(*args):
        raise AssertionError("Experiment reader must not access Bench")
    monkeypatch.setattr("nekaise_loop.benchmark.read_benchmark", no_bench)
    app = create_app(settings)
    monkeypatch.setattr(app.state.service, "ensure_worker", lambda: None)
    with TestClient(app) as client:
        assert client.get("/api/experiments").json()["total"] == 3
        assert client.get(f"/api/experiments?campaign_id={campaign['id']}").json()["total"] == 2
        assert client.get(f"/api/experiments/{card['id']}").json() == card
        assert client.get("/api/experiments/missing").status_code == 404
        assert client.get("/api/experiments?before=0").status_code == 422
        assert client.get("/api/experiments?strategy_version=invalid").status_code == 422
    assert not service.store.query("SELECT id FROM actions")


def test_lineage_context_does_not_inherit_sibling_or_future_parent_experiments(setup_loop):
    settings, service, parent, _ = execute(setup_loop)
    original = catalog(service.store)["items"][0]
    child = service.continue_campaign(parent["id"], {"rounds": 1})
    # Preserve the real temporal boundary in a deterministic fixture.
    service.store.execute("UPDATE campaigns SET created_at='2026-01-01' WHERE id=?", (child["id"],))
    assert latest_context(service.store, service.artifacts, child["id"]) is None
    service.store.execute("UPDATE campaigns SET created_at='9999-01-01' WHERE id=?", (child["id"],))
    assert latest_context(service.store, service.artifacts, child["id"])["round_id"] == original["round_id"]
    unrelated = service.create("Unrelated fixture", CampaignConfig.model_validate(parent["config"]))
    Engine(settings, ExperimentTeacher, FakeModel).run(unrelated["id"])
    assert latest_context(service.store, service.artifacts, child["id"])["round_id"] == original["round_id"]
