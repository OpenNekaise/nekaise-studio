"""Fixture checks of comparison provenance, evidence separation and execution budgets."""
import json
from pathlib import Path

import pytest

from conftest import FakeModel, FakeTeacher
from nekaise_loop.engine import Engine
from nekaise_loop.providers.local import LocalModel


def comparison_setup(setup_loop):
    settings, service, parent, engine = setup_loop
    engine.run(parent["id"])
    reference = service.store.one("SELECT * FROM rounds WHERE campaign_id=? AND number=1", (parent["id"],))
    child = service.continue_campaign(parent["id"], {"rounds": 1})
    return settings, service, child, reference


def teacher_for(reference):
    class Teacher(FakeTeacher):
        def curriculum(self, brief):
            result = super().curriculum(brief)
            result.update(comparison_round_id=reference["id"], train_epochs=0,
                          token_mix={"teacher": 0, "corpus": 0, "replay": 0})
            return result

        def evaluate(self, curriculum, lessons):
            return []
    return Teacher


@pytest.mark.parametrize("reference_status", ["complete", "failed"])
def test_comparison_keeps_current_lineage_and_delivers_reference_to_teacher(setup_loop, reference_status):
    settings, service, child, reference = comparison_setup(setup_loop)
    service.store.execute("UPDATE rounds SET status=? WHERE id=?", (reference_status, reference["id"]))
    observations = []

    class Teacher(teacher_for(reference)):
        def revise(self, lessons):
            assert all(r["student"] != r["comparison"]["generation"]["text"] for r in lessons)
            assert all(r["comparison"]["round_id"] == reference["id"] for r in lessons)
            return super().revise(lessons)

        def reflect(self, result):
            observations.append(result)
            return super().reflect(result)

    class Model(FakeModel):
        def compare(self, checkpoint, historical, rows, on_answer):
            assert historical == reference["checkpoint"] and historical != checkpoint
            return {"current": self.generate(checkpoint, rows, on_answer),
                    "reference": [{"id": r["id"], "text": "Reference fixture", "prompt_token_ids": [1, 2]} for r in rows]}

    FakeModel.datasets = []
    Engine(settings, Teacher, Model).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    result = observations[0]
    assert result["training"]["trained"] is False and result["metrics"] == []
    assert result["training"]["checkpoint"] == child["config"]["student_model"]
    assert all(r["comparison"]["generation"]["prompt"] == r["student_prompt"] for r in result["lessons"])
    assert all(r["comparison"]["generation"]["prompt_token_ids"] == [1, 2] for r in result["lessons"])
    assert not FakeModel.datasets


@pytest.mark.parametrize("defect", ["incomplete", "corrupt", "outside", "identical", "missing"])
def test_invalid_reference_fails_before_any_model_execution(setup_loop, defect):
    settings, service, child, reference = comparison_setup(setup_loop)
    if defect == "incomplete":
        service.store.execute("UPDATE stage_runs SET status='failed' WHERE round_id=? AND stage='train'", (reference["id"],))
    elif defect == "corrupt":
        (Path(reference["checkpoint"])/"model.safetensors").write_bytes(b"corrupt fixture")
    elif defect == "outside":
        stage = service.store.one("SELECT id,artifact FROM stage_runs WHERE round_id=? AND stage='train'", (reference["id"],))
        result = service.artifacts.get(stage["artifact"])
        result["checkpoint"] = str(settings.workspace.parent/"foreign-checkpoint")
        key = service.artifacts.put(result)
        service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (key, stage["id"]))
        service.store.execute("UPDATE rounds SET checkpoint=? WHERE id=?", (result["checkpoint"], reference["id"]))
    elif defect == "identical":
        reference = service.store.one("SELECT * FROM rounds WHERE checkpoint=?", (child["config"]["student_model"],))
    else:
        reference = {"id": "does-not-exist"}

    class NoModel(FakeModel):
        def __init__(self, *args):
            super().__init__(*args)
        def generate(self, *args):
            pytest.fail("Invalid reference must fail before generation")
        def compare(self, *args):
            pytest.fail("Invalid reference must fail before comparison")

    Engine(settings, teacher_for(reference), NoModel).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "failed"
    assert service.store.one("SELECT stage FROM rounds WHERE campaign_id=?", (child["id"],))["stage"] == "select"


def test_reference_revalidated_after_selection(setup_loop):
    settings, service, child, reference = comparison_setup(setup_loop)

    class NoComparison(FakeModel):
        def compare(self, *args):
            pytest.fail("Corrupt frozen reference must never load")

    def pause_after_selection():
        return bool(service.store.one("SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='select' AND s.status='complete'", (child["id"],)))

    engine = Engine(settings, teacher_for(reference), NoComparison)
    engine.run(child["id"], pause=pause_after_selection)
    (Path(reference["checkpoint"])/"model.safetensors").write_bytes(b"changed after selection")
    engine.run(child["id"])
    row = service.store.one("SELECT status,stage,error FROM rounds WHERE campaign_id=?", (child["id"],))
    assert row["status"] == "failed" and row["stage"] == "draft"
    assert "integrity failed" in row["error"]


@pytest.mark.parametrize("exhausted", [False, True])
def test_local_comparison_preserves_prompts_logs_and_shared_timeout(setup_loop, monkeypatch, exhausted):
    settings, _, campaign, _ = setup_loop
    from nekaise_loop.config import CampaignConfig
    config = CampaignConfig.model_validate(campaign["config"]).model_copy(update={"max_stage_seconds": 10})
    ticks = iter([0, 1, 11 if exhausted else 4])
    monkeypatch.setattr("nekaise_loop.providers.local.time.monotonic", lambda: next(ticks))
    calls, projected = [], []
    rows = [{"id": "probe", "prompt": "Exact prompt\nAnswer:"}]

    class Runner:
        def run(self, command, **kwargs):
            payload = json.loads(Path(command[-1]).read_text())
            calls.append((payload, kwargs["timeout"], kwargs["log"]))
            assert payload["rows"] == rows
            answer = {"id": "probe", "text": payload["checkpoint"]}
            kwargs["on_message"]({"type": "answer", "data": answer})
            kwargs["on_message"]({"type": "result", "data": [answer]})
            kwargs["log"].write_text(json.dumps(answer))

    model = LocalModel(config, settings, Runner(), settings.workspace/"comparison")
    if exhausted:
        with pytest.raises(TimeoutError, match="stage budget"):
            model.compare("current-path", "reference-path", rows, projected.append)
        assert len(calls) == 1
    else:
        result = model.compare("current-path", "reference-path", rows, projected.append)
        assert [call[1] for call in calls] == [9, 6]
        assert calls[0][2] != calls[1][2]
        assert all(call[2].exists() for call in calls)
        assert result["reference"][0]["text"] == "reference-path"
    assert projected == [{"id": "probe", "text": "current-path"}]


def test_local_comparison_runs_only_selected_reference_prompts(setup_loop, monkeypatch):
    from nekaise_loop.config import CampaignConfig
    settings, _, campaign, _ = setup_loop
    config = CampaignConfig.model_validate(campaign["config"]).model_copy(update={"max_stage_seconds": 10})
    ticks = iter([0, 1, 4])
    monkeypatch.setattr("nekaise_loop.providers.local.time.monotonic", lambda: next(ticks))
    calls = []
    class Runner:
        def run(self, command, **kwargs):
            payload = json.loads(Path(command[-1]).read_text())
            calls.append((payload, kwargs["timeout"]))
            kwargs["on_message"]({"type": "result", "data": [{"id": r["id"], "text": "Fixture"} for r in payload["rows"]]})
    rows = [{"id": "one", "prompt": "First"}, {"id": "two", "prompt": "Second"}]
    model = LocalModel(config, settings, Runner(), settings.workspace/"subset")
    result = model.compare("after", "before", rows, reference_rows=rows[:1])
    assert [len(call[0]["rows"]) for call in calls] == [2, 1]
    assert [call[1] for call in calls] == [9, 6]
    assert calls[0][0]["config"] == calls[1][0]["config"]
    assert calls[0][0]["batch_groups"] == [["one"], ["two"]]
    assert calls[1][0]["batch_groups"] == [["one"]]
    assert len(result["current"]) == 2 and len(result["reference"]) == 1


def snapshot_fixture(tmp_path):
    import hashlib
    root = tmp_path/"hub/models--fixture--base/snapshots"/("a"*40)
    root.mkdir(parents=True)
    blobs = root.parent.parent/"blobs"
    blobs.mkdir()
    contents = {"config.json": b'{}', "tokenizer_config.json": b'{}', "tokenizer.json": b'{}',
                "model.safetensors.index.json": b'{"weight_map":{"weight":"model-00001.safetensors"}}',
                "model-00001.safetensors": b'FAKE BASE WEIGHTS'}
    for name, data in contents.items():
        key = hashlib.sha256(data).hexdigest() if name.endswith(".safetensors") else hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
        (blobs/key).write_bytes(data)
        (root/name).symlink_to(blobs/key)
    return root


def input_comparison_setup(setup_loop, tmp_path):
    from nekaise_loop.config import CampaignConfig
    settings, service, _, engine = setup_loop
    snapshot = snapshot_fixture(tmp_path)
    config = CampaignConfig.model_validate(setup_loop[2]["config"])
    parent = service.create("Pinned Base fixture", config.model_copy(update={"student_model": str(snapshot)}))
    engine.run(parent["id"])
    reference = service.store.one("SELECT * FROM rounds WHERE campaign_id=? AND number=1", (parent["id"],))
    child = service.continue_campaign(parent["id"], {"rounds": 1})
    class Teacher(teacher_for(reference)):
        def curriculum(self, brief):
            return {**super().curriculum(brief), "comparison_checkpoint": "input"}
    return settings, service, child, reference, snapshot, Teacher


def test_input_comparison_preserves_lineage_and_freezes_snapshot_identity(setup_loop, tmp_path):
    settings, service, child, reference, snapshot, Teacher = input_comparison_setup(setup_loop, tmp_path)
    observed = []
    class Model(FakeModel):
        def compare(self, checkpoint, historical, rows, on_answer):
            assert historical == str(snapshot) == reference["model_before"]
            assert checkpoint == child["config"]["student_model"]
            observed.append(historical)
            return {"current": self.generate(checkpoint, rows, on_answer),
                    "reference": [{"id": r["id"], "text": "Base fixture"} for r in rows]}
    Engine(settings, Teacher, Model).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    row = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (child["id"],))
    assert row["checkpoint"] == row["model_before"] == child["config"]["student_model"]
    lesson = service.store.records(row["id"], "lesson")[0]
    assert lesson["comparison"]["checkpoint_kind"] == "input"
    assert lesson["comparison"]["identity"]["revision"] == "a"*40
    assert len(lesson["comparison"]["identity"]["files"]) == 5
    assert observed == [str(snapshot)]


@pytest.mark.parametrize("defect", ["parent", "artifact", "weights", "config", "missing_shard", "outside_blob", "unpinned", "changed_after_select"])
def test_input_comparison_rejects_invalid_provenance_before_generation(setup_loop, tmp_path, defect):
    import hashlib
    settings, service, child, reference, snapshot, Teacher = input_comparison_setup(setup_loop, tmp_path)
    class NoModel(FakeModel):
        def generate(self, *args):
            pytest.fail("Invalid input reference must fail before generation")
        def compare(self, *args):
            pytest.fail("Invalid input reference must fail before comparison")
    engine = Engine(settings, Teacher, NoModel)
    if defect == "changed_after_select":
        engine.run(child["id"], pause=lambda: bool(service.store.one("SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='select' AND s.status='complete'", (child["id"],))))
        data = b'VALID BUT DIFFERENT WEIGHTS'
        target = snapshot.parent.parent/"blobs"/hashlib.sha256(data).hexdigest()
        target.write_bytes(data)
        (snapshot/"model-00001.safetensors").unlink()
        (snapshot/"model-00001.safetensors").symlink_to(target)
    elif defect == "parent":
        service.store.execute("UPDATE rounds SET model_before=? WHERE id=?", (str(snapshot)+"-other", reference["id"]))
    elif defect == "artifact":
        stage = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train'", (reference["id"],))
        key = stage["artifact"]
        (service.artifacts.root/key[:2]/f"{key}.json").write_text('{}')
    elif defect in {"weights", "config"}:
        (snapshot/("model-00001.safetensors" if defect == "weights" else "config.json")).write_bytes(b'CORRUPT')
    elif defect == "missing_shard":
        (snapshot/"model-00001.safetensors").unlink()
    elif defect == "outside_blob":
        outside = tmp_path/("0"*64)
        outside.write_bytes(b'OUTSIDE')
        (snapshot/"model-00001.safetensors").unlink()
        (snapshot/"model-00001.safetensors").symlink_to(outside)
    elif defect == "unpinned":
        # Corroborated historical parent still must be a pinned snapshot.
        stage = service.store.one("SELECT id,artifact FROM stage_runs WHERE round_id=? AND stage='train'", (reference["id"],))
        result = service.artifacts.get(stage["artifact"])
        result["manifest"]["parent"] = str(tmp_path)
        key = service.artifacts.put(result)
        service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (key, stage["id"]))
        service.store.execute("UPDATE rounds SET model_before=? WHERE id=?", (str(tmp_path), reference["id"]))
    engine.run(child["id"])
    row = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (child["id"],))
    assert row["status"] == "failed"
    assert row["stage"] == ("draft" if defect == "changed_after_select" else "select")
    if defect == "changed_after_select":
        assert "changed after selection" in row["error"]


def local_input_setup(setup_loop):
    settings, service, child, parent = comparison_setup(setup_loop)
    reference = service.store.one("SELECT * FROM rounds WHERE campaign_id=? AND number=2", (parent["campaign_id"],))

    class Teacher(teacher_for(reference)):
        def curriculum(self, brief):
            return {**super().curriculum(brief), "comparison_checkpoint": "input"}

    return settings, service, child, parent, reference, Teacher


@pytest.mark.parametrize("failed_rounds", [False, True])
def test_local_input_comparison_verifies_parent_and_preserves_active_lineage(setup_loop, failed_rounds):
    settings, service, child, parent, reference, Teacher = local_input_setup(setup_loop)
    if failed_rounds:
        service.store.execute("UPDATE rounds SET status='failed' WHERE id IN (?,?)", (parent["id"], reference["id"]))
    observed = []

    class Model(FakeModel):
        def compare(self, checkpoint, historical, rows, on_answer):
            assert historical == parent["checkpoint"] == reference["model_before"]
            assert checkpoint == child["config"]["student_model"] != historical
            observed.append(historical)
            return {"current": self.generate(checkpoint, rows, on_answer),
                    "reference": [{"id": r["id"], "text": "Local parent fixture"} for r in rows]}

    Engine(settings, Teacher, Model).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    row = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (child["id"],))
    assert row["checkpoint"] == row["model_before"] == child["config"]["student_model"]
    comparison = service.store.records(row["id"], "lesson")[0]["comparison"]
    artifact = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete'", (parent["id"],))["artifact"]
    assert comparison["identity"] == {"kind": "workspace_checkpoint", "artifact": artifact,
                                      "files": service.artifacts.get(artifact)["manifest"]["files"]}
    assert comparison["checkpoint_kind"] == "input"
    assert observed == [parent["checkpoint"]]


@pytest.mark.parametrize("defect", ["missing", "incomplete", "artifact", "path", "manifest", "weights", "changed_after_select"])
def test_local_input_rejects_invalid_parent_before_generation(setup_loop, defect):
    settings, service, child, parent, reference, Teacher = local_input_setup(setup_loop)

    class NoModel(FakeModel):
        def generate(self, *args):
            pytest.fail("Invalid local parent must fail before generation")
        def compare(self, *args):
            pytest.fail("Invalid local parent must fail before comparison")

    engine = Engine(settings, Teacher, NoModel)
    stage = service.store.one("SELECT id,artifact FROM stage_runs WHERE round_id=? AND stage='train'", (parent["id"],))
    if defect == "changed_after_select":
        engine.run(child["id"], pause=lambda: bool(service.store.one("SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='select' AND s.status='complete'", (child["id"],))))
    if defect == "missing":
        service.store.execute("UPDATE stage_runs SET status='failed' WHERE id=?", (stage["id"],))
    elif defect == "incomplete":
        service.store.execute("UPDATE stage_runs SET status='running' WHERE id=?", (stage["id"],))
    elif defect == "artifact":
        key = stage["artifact"]
        (service.artifacts.root/key[:2]/f"{key}.json").write_text('{}')
    elif defect == "path":
        result = service.artifacts.get(stage["artifact"])
        result["checkpoint"] = reference["checkpoint"]
        key = service.artifacts.put(result)
        service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (key, stage["id"]))
    elif defect == "manifest":
        (Path(parent["checkpoint"])/"checkpoint.json").write_text('{}')
    else:
        (Path(parent["checkpoint"])/"model.safetensors").write_bytes(b'CORRUPT LOCAL PARENT')
    engine.run(child["id"])
    row = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (child["id"],))
    assert row["status"] == "failed"
    assert row["stage"] == ("draft" if defect == "changed_after_select" else "select")
    assert any(fragment in row["error"] for fragment in
               ("parent training artifact", "integrity failed", "provenance differs", "manifest"))


def test_mixed_interfaces_use_reference_provenance_and_remain_visible(setup_loop):
    from nekaise_loop.config import CampaignConfig
    settings, service, _, reference = comparison_setup(setup_loop)
    base = CampaignConfig.model_validate(setup_loop[2]["config"])
    child = service.create("Fresh chat fixture", base.model_copy(update={
        "student_format": "chat_template", "rounds": 1}))
    observed = []

    class Model(FakeModel):
        def compare(self, checkpoint, historical, rows, on_answer, *, reference_format=None):
            assert self.config.student_format == "chat_template"
            assert reference_format == "raw_text"
            observed.append(rows)
            return {"current": self.generate(checkpoint, rows, on_answer),
                    "reference": [{"id": r["id"], "text": "Base fixture"} for r in rows]}

    Engine(settings, teacher_for(reference), Model).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    rnd = service.store.one("SELECT * FROM rounds WHERE campaign_id=?", (child["id"],))
    for lesson in service.store.records(rnd["id"], "lesson"):
        assert lesson["comparison"]["basis"] == "checkpoint_and_interface"
        assert lesson["comparison"]["current_student_format"] == "chat_template"
        assert lesson["comparison"]["reference_student_format"] == "raw_text"
    stage = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train'", (rnd["id"],))
    assert service.artifacts.get(stage["artifact"])["student_format"] == "chat_template"
    assert observed


def test_local_comparison_sets_each_config_and_preserves_explicit_override(setup_loop):
    from nekaise_loop.config import CampaignConfig
    settings, _, campaign, _ = setup_loop
    config = CampaignConfig.model_validate(campaign["config"]).model_copy(update={"student_format": "chat_template"})
    calls = []
    rows = [{"id": "native", "prompt": "question"}, {"id": "raw", "prompt": "continuation", "format": "raw_text"}]

    class Runner:
        def run(self, command, **kwargs):
            payload = json.loads(Path(command[-1]).read_text())
            calls.append(payload)
            kwargs["on_message"]({"type": "result", "data": [{"id": r["id"], "text": "fixture"} for r in rows]})

    LocalModel(config, settings, Runner(), settings.workspace/"mixed-comparison").compare(
        "sft", "base", rows, reference_format="raw_text")
    assert [p["config"]["student_format"] for p in calls] == ["chat_template", "raw_text"]
    assert all(p["rows"] == rows for p in calls)
