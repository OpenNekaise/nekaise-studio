"""Fixture-only checks of paired online evidence and teacher-owned criteria."""
from pathlib import Path

import pytest

from conftest import FakeModel, FakeTeacher
from nekaise_loop.engine import Engine
from nekaise_loop.teaching import Evaluation


def fixture_answer(row, text):
    return {"id": row["id"], "text": text, "prompt_token_ids": [1, 20, 30],
            "generation_settings": {"do_sample": False, "max_new_tokens": 10},
            "runtime": {"dtype": "float32", "device": "fixture"}}


class PairedTeacher(FakeTeacher):
    observations = []
    grading_inputs = []

    def evaluate(self, curriculum, lessons):
        items = super().evaluate(curriculum, lessons)
        for i, row in enumerate(items):
            row.update(compare_before=i == 0, dimensions=[{"name": "meaning", "rubric": "Judge content independently of form"}])
        return items

    def grade(self, items):
        self.grading_inputs.append(items)
        result = super().grade(items)
        for row, grade in zip(items, result):
            score = .2 if row["student"] == "Before fixture" else .8
            grade.update(score=score, dimensions=[{"name": "meaning", "score": score, "feedback": "Fixture observation"}])
        return result

    def reflect(self, result):
        self.observations.append(result)
        return super().reflect(result)


class PairedModel(FakeModel):
    comparisons = []

    def compare(self, checkpoint, reference, rows, on_answer, *, reference_rows=None):
        self.comparisons.append((checkpoint, reference, rows, reference_rows))
        current = [fixture_answer(r, "After fixture") for r in rows]
        for answer in current:
            on_answer(answer)
        return {"current": current, "reference": [fixture_answer(r, "Before fixture") for r in reference_rows]}


def child_setup(setup_loop):
    settings, service, parent, engine = setup_loop
    engine.run(parent["id"])
    child = service.continue_campaign(parent["id"], {"rounds": 1})
    PairedTeacher.observations, PairedTeacher.grading_inputs, PairedModel.comparisons = [], [], []
    return settings, service, child


def test_online_pair_preserves_lineage_hides_labels_and_grades_frozen_dimensions(setup_loop):
    settings, service, child = child_setup(setup_loop)
    Engine(settings, PairedTeacher, PairedModel).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    rnd = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (child["id"],))
    selected, ordinary = service.store.records(rnd["id"], "evaluation")
    pair = selected["comparison"]
    assert pair["before"] == rnd["model_before"] and pair["after"] == rnd["checkpoint"]
    assert selected["student"] == "After fixture" and pair["generation"]["text"] == "Before fixture"
    assert pair["score_delta"] == pytest.approx(.6)
    assert pair["conditions"]["comparable"] is True
    assert selected["grade"]["dimensions"][0]["score"] == .8
    assert pair["grade"]["dimensions"][0]["score"] == .2
    assert "comparison" not in ordinary
    assert len(PairedModel.comparisons) == 1
    _, _, prompts, reference_prompts = PairedModel.comparisons[0]
    assert len(prompts) == 2 and len(reference_prompts) == 1
    assert all(set(p) == {"id", "prompt"} and "HIDDEN_REFERENCE" not in p["prompt"] for p in prompts)
    grading = PairedTeacher.grading_inputs[0]
    assert len(grading) == 3 and all(len(r["id"]) == 24 for r in grading)
    assert all("comparison" not in r and "checkpoint" not in str(r) for r in grading)
    assert all(r["id"] in {"e1", "e2"} for r in (selected["grade"], ordinary["grade"], pair["grade"]))
    assert PairedTeacher.observations[0]["assessment"]["items"][0]["comparison"]["grade"]["score"] == .2
    assert PairedTeacher.observations[0]["learning_work"]["measured_work_all_attempts"]["updates"] == 3


@pytest.mark.parametrize("failed_stage", ["evaluate", "reflect"])
@pytest.mark.parametrize("defect", [None, "incomplete_train", "weights"])
def test_online_pair_from_completed_training_in_failed_round(setup_loop, failed_stage, defect):
    settings, service, parent, _ = setup_loop

    class InterruptedTeacher(FakeTeacher):
        def evaluate(self, curriculum, lessons):
            if failed_stage == "evaluate":
                raise RuntimeError("Fixture assessment interruption after training")
            return super().evaluate(curriculum, lessons)

        def reflect(self, result):
            raise RuntimeError("Fixture reflection interruption after training")

    Engine(settings, InterruptedTeacher, FakeModel).run(parent["id"])
    rnd = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (parent["id"],))
    assert rnd["status"] == "failed"
    producer = service.store.one("SELECT id,artifact FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete'", (rnd["id"],))
    child = service.continue_campaign(parent["id"], {"rounds": 1})
    assert child["config"]["student_model"] == rnd["checkpoint"]
    PairedTeacher.observations, PairedTeacher.grading_inputs, PairedModel.comparisons = [], [], []
    engine = Engine(settings, PairedTeacher, PairedModel)
    # Freeze the comparison before corrupting provenance, to exercise its recheck
    # immediately before inference as well as its initial identity lookup.
    if defect:
        engine.run(child["id"], pause=lambda: bool(service.store.one(
            "SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='evaluate' AND s.status='complete'",
            (child["id"],))))
        assert service.store.campaign(child["id"])["status"] == "paused"
        if defect == "incomplete_train":
            service.store.execute("UPDATE stage_runs SET status='failed' WHERE id=?", (producer["id"],))
        else:
            (Path(rnd["checkpoint"]) / "model.safetensors").write_bytes(b"changed")
    engine.run(child["id"])
    if defect:
        assert service.store.campaign(child["id"])["status"] == "failed"
        assert not PairedModel.comparisons and not PairedTeacher.grading_inputs
    else:
        assert service.store.campaign(child["id"])["status"] == "complete"
        pair = PairedTeacher.observations[0]["assessment"]["items"][0]["comparison"]
        assert pair["before_identity"]["artifact"] == producer["artifact"]
        assert pair["before"] == rnd["checkpoint"]
        assert len(PairedModel.comparisons) == 1


@pytest.mark.parametrize("defect", ["missing", "weights", "identity_changed"])
def test_online_pair_rechecks_parent_before_any_answer(setup_loop, defect):
    settings, service, child = child_setup(setup_loop)
    engine = Engine(settings, PairedTeacher, PairedModel)
    engine.run(child["id"], pause=lambda: bool(service.store.one("SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='evaluate' AND s.status='complete'", (child["id"],))))
    checkpoint = Path(child["config"]["student_model"])
    if defect == "missing":
        (checkpoint/"model.safetensors").unlink()
    elif defect == "weights":
        (checkpoint/"model.safetensors").write_bytes(b"changed")
    else:
        parent = service.store.one("SELECT s.id,s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.checkpoint=? AND s.stage='train' AND s.status='complete'", (str(checkpoint),))
        saved = service.artifacts.get(parent["artifact"])
        saved["extra_fixture_observation"] = True
        service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (service.artifacts.put(saved), parent["id"]))
    engine.run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "failed"
    assert not PairedModel.comparisons and not PairedTeacher.grading_inputs


@pytest.mark.parametrize("defect", ["runtime", "missing_evidence", "missing_id"])
def test_noncomparable_or_incomplete_answers_never_produce_gain(setup_loop, defect):
    settings, service, child = child_setup(setup_loop)
    class Model(PairedModel):
        def compare(self, *args, **kwargs):
            result = super().compare(*args, **kwargs)
            if defect == "runtime":
                result["reference"][0]["runtime"]["dtype"] = "bfloat16"
            elif defect == "missing_evidence":
                result["reference"][0].pop("prompt_token_ids")
            else:
                result["reference"] = []
            return result
    Engine(settings, PairedTeacher, Model).run(child["id"])
    if defect == "missing_id":
        assert service.store.campaign(child["id"])["status"] == "failed"
        assert not PairedTeacher.grading_inputs
    else:
        assert service.store.campaign(child["id"])["status"] == "complete"
        pair = PairedTeacher.observations[0]["assessment"]["items"][0]["comparison"]
        assert not pair["conditions"]["comparable"] and pair["score_delta"] is None


def test_diagnostic_round_reuses_one_observation_without_claiming_gain(setup_loop):
    settings, service, child = child_setup(setup_loop)
    class Teacher(PairedTeacher):
        def curriculum(self, brief):
            return {**super().curriculum(brief), "train_epochs": 0}
    Engine(settings, Teacher, PairedModel).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    assert not PairedModel.comparisons
    assert len(PairedTeacher.grading_inputs[0]) == 2
    pair = PairedTeacher.observations[0]["assessment"]["items"][0]["comparison"]
    assert pair["weights_changed"] is False and pair["score_delta"] is None
    assert pair["observation"] == "same_generation_reused_weights_unchanged"


def test_grader_cannot_change_predeclared_dimensions(setup_loop):
    settings, service, child = child_setup(setup_loop)
    class Teacher(PairedTeacher):
        def grade(self, items):
            result = super().grade(items)
            result[0]["dimensions"][0]["name"] = "undeclared"
            return result
    Engine(settings, Teacher, PairedModel).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "failed"
    assert "dimensions frozen" in service.store.one("SELECT error FROM rounds WHERE campaign_id=?", (child["id"],))["error"]


def test_duplicate_criterion_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        Evaluation(id="e", sources=[], question="Q", student_prompt="Q", reference="A", rubric=[], concept="x", evidence="",
                   dimensions=[{"name": "x", "rubric": "one"}, {"name": "x", "rubric": "two"}])


def test_no_update_history_does_not_replace_weight_producer_identity(setup_loop):
    settings, service, diagnostic = child_setup(setup_loop)
    before = diagnostic["config"]["student_model"]
    producer = service.store.one("SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.checkpoint=? AND r.model_before!=r.checkpoint AND s.stage='train'", (before,))
    class Teacher(FakeTeacher):
        def curriculum(self, brief):
            return {**super().curriculum(brief), "train_epochs": 0}
    Engine(settings, Teacher, FakeModel).run(diagnostic["id"])
    child = service.continue_campaign(diagnostic["id"], {"rounds": 1})
    Engine(settings, PairedTeacher, PairedModel).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    assert PairedTeacher.observations[-1]["assessment"]["items"][0]["comparison"]["before_identity"]["artifact"] == producer["artifact"]


def test_plain_grading_keeps_evidence_but_omits_execution_metadata():
    from nekaise_loop.assessment import grade_requests
    item = {"id": "original", "question": "Q", "student_prompt": "Q", "reference": "A", "rubric": [], "concept": "c", "dimensions": [],
            "student": "Answer", "evidence": "Source evidence", "sources": [{"text": "source"}],
            "generation_audit": {"irrelevant": "verbose runtime evidence"},
            "comparison": {"weights_changed": False, "before": "checkpoint", "generation": {"text": "Answer"}}}
    requests, mapping = grade_requests([item], "round")
    assert len(requests) == 1 and requests[0]["evidence"] == "Source evidence"
    assert requests[0]["sources"] == item["sources"]
    assert "comparison" not in requests[0] and "generation_audit" not in requests[0]
    assert mapping[requests[0]["id"]] == ("original", "after")
def test_batch_context_is_part_of_comparability_when_recorded():
    from nekaise_loop.assessment import comparable_generations
    evidence = {"prompt_token_ids": [1, 2], "generation_settings": {"do_sample": False},
                "runtime": {"dtype": "torch.float32"}}
    batched = {**evidence, "generation_execution": {"batch_prompt_hash": "group-a", "position_in_batch": 0}}
    assert comparable_generations(batched, batched)["comparable"]
    assert not comparable_generations(batched, evidence)["comparable"]
    other = {**batched, "generation_execution": {"batch_prompt_hash": "group-b", "position_in_batch": 0}}
    assert comparable_generations(batched, other)["differences"] == ["generation_execution_differ"]
