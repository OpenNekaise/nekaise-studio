"""Fixture diagnostics exercise provenance and ownership, never learning quality."""
import json
from pathlib import Path

import pytest

from conftest import FakeModel, FakeTeacher
from nekaise_loop.engine import Engine
from nekaise_loop.providers.local import LocalModel


def scoring_setup(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    reference = service.store.one("SELECT * FROM rounds WHERE campaign_id=? AND number=2", (campaign["id"],))
    child = service.continue_campaign(campaign["id"], {"rounds": 1})
    class Teacher(FakeTeacher):
        def curriculum(self, brief):
            result = super().curriculum(brief)
            return {**result, "lessons": [], "readings": [], "replay": [],
                    "scoring_round_ids": [reference["id"]], "train_epochs": 0,
                    "token_mix": {"teacher": 0, "corpus": 0, "replay": 0}}
        def evaluate(self, *args):
            return []
    return settings, service, child, reference, Teacher


def test_historical_scoring_keeps_exact_samples_lineage_and_reflection(setup_loop):
    settings, service, child, reference, Teacher = scoring_setup(setup_loop)
    observations = []
    class Reflect(Teacher):
        def reflect(self, result):
            observations.append(result)
            return super().reflect(result)
    class Model(FakeModel):
        def score_history(self, pairs):
            pair, = pairs
            assert pair["before"] == reference["model_before"]
            assert pair["after"] == reference["checkpoint"]
            frozen = service.artifacts.get(pair["artifacts"]["freeze"])
            assert [{k: s[k] for k in ("row_id", "stream", "input_ids")} for s in pair["samples"]] == frozen["samples"]
            assert all(s["prompt_tokens"] is None for s in pair["samples"])
            return [{"reference": pair, "before": "fixture likelihood", "after": "fixture likelihood"}]
    FakeModel.datasets = []
    Engine(settings, Reflect, Model).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    result, = observations
    assert result["historical_scoring"][0]["reference"]["round_id"] == reference["id"]
    assert result["training"]["trained"] is False and result["metrics"] == []
    assert result["training"]["checkpoint"] == child["config"]["student_model"]
    assert FakeModel.datasets == []


@pytest.mark.parametrize("defect", ["incomplete", "dataset", "parent", "checkpoint", "after_selection"])
def test_invalid_scoring_provenance_never_loads_models(setup_loop, defect):
    settings, service, child, reference, Teacher = scoring_setup(setup_loop)
    class NoModel(FakeModel):
        def score_history(self, *args):
            pytest.fail("Invalid scoring reference must not load")
        def generate(self, *args):
            pytest.fail("Invalid scoring reference must fail before generation")
    engine = Engine(settings, Teacher, NoModel)
    if defect == "after_selection":
        engine.run(child["id"], pause=lambda: bool(service.store.one("SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='select' AND s.status='complete'", (child["id"],))))
    if defect == "incomplete":
        service.store.execute("UPDATE rounds SET status='failed' WHERE id=?", (reference["id"],))
    elif defect == "dataset":
        stage = service.store.one("SELECT * FROM stage_runs WHERE round_id=? AND stage='freeze'", (reference["id"],))
        frozen = service.artifacts.get(stage["artifact"])
        frozen["samples"][0]["input_ids"][0] += 1
        key = service.artifacts.put(frozen)
        service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (key, stage["id"]))
    else:
        path = reference["model_before"] if defect == "parent" else reference["checkpoint"]
        (Path(path)/"model.safetensors").write_bytes(b"CORRUPT FIXTURE")
    engine.run(child["id"])
    row = service.store.one("SELECT status,stage FROM rounds WHERE campaign_id=?", (child["id"],))
    if defect == "checkpoint":
        # The reference is also this continuation's parent, so entry validation
        # rejects its corruption even before creating a round.
        assert row is None and service.store.campaign(child["id"])["status"] == "failed"
        return
    assert row == {"status": "failed", "stage": "draft" if defect == "after_selection" else "select"}


def test_scoring_and_generation_share_budget_and_separate_logs(setup_loop, monkeypatch):
    from nekaise_loop.config import CampaignConfig
    settings, _, campaign, _ = setup_loop
    config = CampaignConfig.model_validate(campaign["config"]).model_copy(update={"max_stage_seconds": 10})
    clock = [0]
    monkeypatch.setattr("nekaise_loop.providers.local.time.monotonic", lambda: clock[0])
    calls = []
    class Runner:
        def run(self, command, **kwargs):
            payload = json.loads(Path(command[-1]).read_text())
            calls.append((command, payload, kwargs["timeout"], kwargs["log"]))
            clock[0] += 4
            kwargs["on_message"]({"type": "result", "data": {"fixture": True}})
    local = LocalModel(config, settings, Runner(), settings.workspace/"scoring")
    samples = [{"input_ids": [1, 4, 2], "prompt_tokens": 1}]
    local.score_history([{"before": "before", "after": "after", "samples": samples, "dataset_hash": "fixture"}])
    local.generate("current", [])
    assert [c[2] for c in calls] == [10, 6, 2]
    assert len({c[3] for c in calls}) == 3
    assert all(c[1]["samples"] == samples and c[0][2].endswith("scoring.py") for c in calls[:2])
    with pytest.raises(TimeoutError, match="shared stage budget"):
        local.generate("current", [])


def test_operator_requests_survive_new_actions_reports_and_continuation(setup_loop):
    from nekaise_loop.reports import agent_context
    from nekaise_loop.teacher_tools import operational_context
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    config = {**campaign["config"], "auto_recover": True}
    service.store.execute("UPDATE campaigns SET config=?,status='paused' WHERE id=?", (json.dumps(config), campaign["id"]))
    service.action(campaign["id"], "review", spawn=False, reason="Unresolved exact original controls")
    first = service.store.one("SELECT id FROM actions WHERE kind='review'")["id"]
    child = service.continue_campaign(campaign["id"], {"rounds": 1}, start=True)
    service.action(child["id"], "pause", spawn=False, actor="operator", reason="Explicit hold")
    for i in range(30):
        service.store.event(child["id"], None, "fixture", "Later activity", {"i": i})
    before = service.store.query("SELECT * FROM actions")
    context = operational_context(settings.workspace, child["id"])
    assert context["operator_hold"] == "pause"
    assert context["operator_review_requests"]["requests"][0]["id"] == first
    assert agent_context(service)["operator_review_requests"] == context["operator_review_requests"]
    assert service.store.query("SELECT * FROM actions") == before
    assert service.store.campaign(child["id"])["operator_hold"] == "pause"
