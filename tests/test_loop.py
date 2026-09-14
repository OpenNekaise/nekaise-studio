import json
from pathlib import Path

import pytest

from conftest import FakeTeacher, FakeModel
from nekaise_loop.artifacts import Artifacts
from nekaise_loop.config import STAGES, CampaignConfig
from nekaise_loop.corpus import read_source, search_sources
from nekaise_loop.service import Conflict


def test_two_rounds_chain_checkpoints_and_adapt_without_eval_leakage(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    snapshot = service.snapshot(campaign["id"])
    assert snapshot["campaign"]["status"] == "complete"
    assert len(snapshot["rounds"]) == 2
    second, first = snapshot["rounds"]
    assert second["model_before"] == first["checkpoint"]
    first_lessons = service.store.records(first["id"], "lesson")
    second_lessons = service.store.records(second["id"], "lesson")
    assert {r["document"]["id"] for r in first_lessons} != {r["document"]["id"] for r in second_lessons}
    assert any("Addresses" in r["document"]["selection_reason"] for r in second_lessons)
    assert len(snapshot["round"]["stages"]) == len(STAGES)
    assert len(snapshot["round"]["metrics"]) == 3
    assert snapshot["round"]["gaps"]
    assert all("HIDDEN_REFERENCE" not in r["prompt"] for r in FakeModel.prompts)
    assert "HIDDEN_REFERENCE" not in json.dumps(FakeModel.datasets)
    assert all({"id", "stream", "text", "lesson_id", "document_id", "source_sha256"} <= set(r) for rows in FakeModel.datasets for r in rows)
    completed = service.store.query("SELECT * FROM stage_runs WHERE status='complete'")
    for r in completed:
        assert engine.artifacts.get(r["artifact"]) is not None


def test_failed_stage_retry_preserves_completed_artifacts(setup_loop):
    _, service, campaign, engine = setup_loop
    FakeTeacher.fail_gate_once = True
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    row = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (campaign["id"],))
    before = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='draft'", (row["id"],))
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    drafts = service.store.query("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='draft'", (row["id"],))
    assert drafts == [before]
    gates = service.store.query("SELECT status FROM stage_runs WHERE round_id=? AND stage='revise' ORDER BY attempt", (row["id"],))
    assert [r["status"] for r in gates] == ["failed", "complete"]


def test_teacher_is_trusted_without_a_second_grading_call(setup_loop):
    _, service, campaign, engine = setup_loop
    FakeTeacher.reject = True
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert len(FakeModel.datasets) == 2
    assert service.store.query("SELECT * FROM metrics")


def test_actions_prevent_parallel_campaigns(setup_loop):
    _, service, campaign, _ = setup_loop
    second = service.create("Second", CampaignConfig.model_validate(campaign["config"]))
    service.action(campaign["id"], "start", spawn=False)
    with pytest.raises(Conflict):
        service.action(second["id"], "start", spawn=False)
    with pytest.raises(Conflict):
        service.action(campaign["id"], "start", spawn=False)
    assert service.store.one("SELECT COUNT(*) AS n FROM actions")["n"] == 1


def test_pause_and_resume_at_stage_boundary(setup_loop):
    _, service, campaign, engine = setup_loop
    engine.run(campaign["id"], pause=lambda: True)
    assert service.store.campaign(campaign["id"])["status"] == "paused"
    assert not service.store.query("SELECT * FROM stage_runs")
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"


def test_tampered_artifact_fails_resume(setup_loop):
    settings, service, campaign, engine = setup_loop
    FakeTeacher.fail_gate_once = True
    engine.run(campaign["id"])
    row = service.store.one("SELECT artifact FROM stage_runs WHERE stage='draft'")
    path = settings.workspace/"artifacts"/row["artifact"][:2]/f"{row['artifact']}.json"
    path.write_text('{"wrong":true}')
    engine.run(campaign["id"])
    assert "Artifact integrity failed" in service.store.campaign(campaign["id"])["error"]


def test_hash_mismatch_and_restricted_sources_excluded(corpus):
    rows = search_sources(corpus, limit=2)["rows"]
    blocked = rows[0]["id"]
    (corpus/"registry/eligibility.json").write_text(json.dumps({"version":1,"restrictions":{"test":{"status":"restricted","match":{"id_prefix":blocked}}}}))
    (corpus/"corpus"/f"{rows[1]['id']}.md").write_text("Unverified replacement text"*100)
    with pytest.raises(ValueError, match="ineligible"):
        read_source(corpus, blocked)
    with pytest.raises(ValueError, match="hash mismatch"):
        read_source(corpus, rows[1]["id"])


def test_teacher_controls_practice_even_when_score_is_high(setup_loop):
    from nekaise_loop.engine import Engine
    settings, service, campaign, _ = setup_loop
    class Teacher(FakeTeacher):
        def grade(self, items):
            return [{**r, "score": 1.0, "needs_practice": True, "priority": .9} for r in super().grade(items)]
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    assert service.snapshot(campaign["id"])["round"]["gaps"][0]["priority"] == .9


def test_cross_round_parent_tampering_stops_before_new_training(setup_loop):
    _, service, campaign, engine = setup_loop
    tampered = [False]
    def control():
        first = service.store.one("SELECT * FROM rounds WHERE campaign_id=? AND number=1 AND status='complete'", (campaign["id"],))
        if first and not tampered[0]:
            (Path(first["checkpoint"])/"model.safetensors").write_bytes(b"corrupted checkpoint")
            tampered[0] = True
        return False
    engine.run(campaign["id"], controls=control)
    assert tampered[0]
    assert service.store.campaign(campaign["id"])["status"] == "failed"
    assert "Checkpoint integrity failed" in service.store.campaign(campaign["id"])["error"]
    assert len(FakeModel.datasets) == 1


def test_resume_consumes_obsolete_stop_requests(setup_loop):
    _, service, campaign, _ = setup_loop
    service.store.set_status(campaign["id"], "failed")
    service.store.execute("INSERT INTO actions(campaign_id,kind,created_at) VALUES(?,'stop','2026-01-01')", (campaign["id"],))
    service.action(campaign["id"], "resume", spawn=False)
    rows = service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL")
    assert rows == [{"kind": "resume"}]
