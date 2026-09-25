import json
from pathlib import Path
import sqlite3

import pytest

from conftest import FakeTeacher, FakeModel
from nekaise_loop.config import CampaignConfig
from nekaise_loop.engine import Engine
from nekaise_loop.storage import now
from nekaise_loop.teaching import Curriculum
from nekaise_loop.teacher_tools import archive, query, latest_strategy, operational_context


def plan():
    return {"lessons": [{"id":"own-task", "kind":"sft", "sources":[], "concept":"Teacher's own exercise", "prompt":"Explain a relationship", "student_prompt":"You may use this background: A implies B. Explain B.", "reason":"Teacher chose to provide context"}], "readings":[], "replay":[], "token_mix":{"teacher":1,"corpus":0,"replay":0}, "train_epochs":2, "evaluation_instructions":"Choose the assessment freely", "notes":"Teach one task instead of the configured two"}


class AutonomousTeacher(FakeTeacher):
    def curriculum(self, brief):
        return plan()

    def evaluate(self, curriculum, lessons):
        return [{"id":f"check-{i}","sources":[],"question":"Explain B", "student_prompt":f"Exact teacher assessment prompt {i}","reference":"HIDDEN_REFERENCE expected result", "rubric":["Use the relationship"],"concept":"Relationship", "evidence":""} for i in range(3)]


def test_teacher_chooses_counts_formats_context_and_mix(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Model(FakeModel):
        def train(self, checkpoint, dataset, dataset_hash, on_metric):
            assert self.config.train_epochs == 2
            assert self.config.token_mix.teacher == 1
            return super().train(checkpoint, dataset, dataset_hash, on_metric)
    Engine(settings, AutonomousTeacher, Model).run(campaign["id"])
    snapshot = service.snapshot(campaign["id"])
    assert snapshot["campaign"]["status"] == "complete"
    assert len(snapshot["round"]["lessons"]) == 1
    assert len(snapshot["round"]["evaluations"]) == 3
    assert snapshot["round"]["lessons"][0]["kind"] == "sft"
    assert all(row["stream"] == "sft" and row["text"].startswith(plan()["lessons"][0]["student_prompt"]) for rows in FakeModel.datasets for row in rows)
    prompts = {row["prompt"] for row in FakeModel.prompts}
    assert plan()["lessons"][0]["student_prompt"] in prompts
    assert "Exact teacher assessment prompt 0" in prompts
    assert not any("HIDDEN_REFERENCE" in prompt for prompt in prompts)
    assert snapshot["round"]["curriculum"]["notes"] == plan()["notes"]
    assert snapshot["round"]["teaching_strategy"]["student_notes"]


def test_teacher_can_omit_material_and_run_diagnostics_without_weight_updates(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Teacher(AutonomousTeacher):
        def revise(self, lessons):
            return [{**r,"use_for_training":False,"text":"","reason":"Assessment first"} for r in super().revise(lessons)]
        def reflect(self, observations):
            assert observations["training"]["trained"] is False
            assert observations["metrics"] == []
            return {"student_notes":"Observed answers only", "next_round_instructions":"Review later", "action":"complete", "reason":"Diagnostic objective complete"}
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    snapshot = service.snapshot(campaign["id"])
    assert snapshot["campaign"]["status"] == "complete"
    assert len(snapshot["rounds"]) == 1
    assert snapshot["round"]["checkpoint"] == campaign["config"]["student_model"]
    assert snapshot["round"]["token_ledger"]["total_tokens"] == 0
    assert not FakeModel.datasets
    assert not service.store.query("SELECT * FROM metrics")


def test_teacher_pause_goes_through_the_command_queue(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Teacher(AutonomousTeacher):
        def reflect(self, observations):
            return {"student_notes":"Need owner feedback", "next_round_instructions":"Wait", "action":"pause", "reason":"Teaching pause"}
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "pausing"
    assert service.store.query("SELECT kind FROM actions WHERE handled_at IS NULL") == [{"kind":"pause"}]


def test_teacher_can_prepare_material_and_assess_without_enabling_training(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Teacher(AutonomousTeacher):
        def curriculum(self, brief):
            result = plan()
            result["train_epochs"] = 0
            return result
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    snapshot = service.snapshot(campaign["id"])
    assert snapshot["campaign"]["status"] == "complete"
    assert snapshot["round"]["checkpoint"] == campaign["config"]["student_model"]
    assert snapshot["round"]["token_ledger"]["total_tokens"] > 0
    assert snapshot["round"]["lessons"][0]["gate"]["passed"]
    assert len(snapshot["round"]["evaluations"]) == 3
    assert not FakeModel.datasets
    assert not service.store.query("SELECT * FROM metrics")


@pytest.mark.parametrize("diagnostic_rounds", [0, 2])
def test_optimizer_reset_waits_for_first_actual_training_round(setup_loop, diagnostic_rounds):
    settings, service, original, _ = setup_loop
    config = CampaignConfig.model_validate(original["config"]).model_copy(update={"inherit_optimizer":False, "rounds":diagnostic_rounds+2})
    campaign = service.create("Explicit reset after diagnostics", config)
    inherited = []
    class Teacher(AutonomousTeacher):
        def curriculum(self, brief):
            result = plan()
            if brief["round_number"] <= diagnostic_rounds:
                result.update(train_epochs=0, token_mix={"teacher":0,"corpus":0,"replay":0})
            return result
    class Model(FakeModel):
        def train(self, checkpoint, dataset, dataset_hash, on_metric):
            inherited.append(self.config.inherit_optimizer)
            return super().train(checkpoint, dataset, dataset_hash, on_metric)
    Engine(settings, Teacher, Model).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert inherited == [False, True]
    child = service.continue_campaign(campaign["id"], {"rounds": 1})
    assert child["config"]["inherit_optimizer"] is True
    reset = service.continue_campaign(campaign["id"], {"rounds": 1, "inherit_optimizer": False})
    assert reset["config"]["inherit_optimizer"] is False


@pytest.mark.parametrize("trained_first", [False, True])
def test_continuation_preserves_pending_reset_through_diagnostics_only(setup_loop, trained_first):
    settings, service, original, _ = setup_loop
    config = CampaignConfig.model_validate(original["config"]).model_copy(update={"inherit_optimizer": False, "rounds": 2})
    campaign = service.create("Reset consumption fixture", config)
    class Teacher(AutonomousTeacher):
        def curriculum(self, brief):
            result = plan()
            if not trained_first or brief["round_number"] == 2:
                result.update(train_epochs=0, token_mix={"teacher": 0, "corpus": 0, "replay": 0})
            return result
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    child = service.continue_campaign(campaign["id"], {"rounds": 1})
    assert child["config"]["inherit_optimizer"] is trained_first


def test_zero_mix_diagnostic_round_preserves_decision_and_next_round_trains(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Teacher(AutonomousTeacher):
        def curriculum(self, brief):
            result = plan()
            if brief["round_number"] == 1:
                result.update(train_epochs=0, token_mix={"teacher":0,"corpus":0,"replay":0})
            return result
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    rounds = service.store.query("SELECT * FROM rounds WHERE campaign_id=? ORDER BY number", (campaign["id"],))
    first, second = (service.round_detail(r["id"]) for r in rounds)
    assert first["curriculum"]["token_mix"] == {"teacher":0,"corpus":0,"replay":0}
    assert first["checkpoint"] == first["model_before"] == campaign["config"]["student_model"]
    assert first["lessons"][0]["student"] and first["lessons"][0]["training_text"]
    assert len(first["evaluations"]) == 3
    assert first["metrics"] == []
    frozen = next(s for s in first["stages"] if s["stage"] == "freeze")
    dataset = service.artifacts.get(frozen["artifact"])
    assert dataset["ledger"]["total_tokens"] == 0
    assert dataset["samples"] == [] and dataset["rows"]
    assert second["model_before"] == first["checkpoint"]
    assert second["checkpoint"] != second["model_before"]
    assert second["metrics"] and len(FakeModel.datasets) == 1


@pytest.mark.parametrize("epochs,mix", [
    (1, {"teacher":0,"corpus":0,"replay":0}),
    (0, {"teacher":.2,"corpus":.2,"replay":0}),
    (0, {"teacher":-1,"corpus":1,"replay":0}),
])
def test_invalid_curriculum_mix_is_not_silently_changed(epochs, mix):
    with pytest.raises(ValueError):
        Curriculum.model_validate({**plan(), "train_epochs":epochs, "token_mix":mix})


def test_teacher_can_defer_assessment(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Teacher(AutonomousTeacher):
        def evaluate(self, curriculum, lessons):
            return []
        def grade(self, items):
            pytest.fail("An empty assessment does not require a grading call")
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    snapshot = service.snapshot(campaign["id"])
    assert snapshot["campaign"]["status"] == "complete"
    assert snapshot["round"]["evaluations"] == []
    assert snapshot["round"]["score"] is None


def test_full_history_is_paginated_searchable_read_only_and_cross_campaign(setup_loop):
    settings, service, campaign, _ = setup_loop
    for i in range(140):
        rid = f"old-{i}"
        service.store.execute("INSERT INTO rounds(id,campaign_id,number,status,model_before,created_at,updated_at) VALUES(?,?,?,'complete','fixture',?,?)", (rid,campaign["id"],i+1,now(),now()))
        service.store.put_records(rid,"lesson",[{"id":"old-lesson","teacher":f"Archive lesson {i}","special":"earliest-only" if i==0 else ""}])
    child = service.create("Descendant", CampaignConfig.model_validate(campaign["config"]), parent_id=campaign["id"])
    context = {"workspace":str(settings.workspace), "corpus_path":campaign["config"]["corpus_path"], "campaign_id":child["id"]}
    offset, ids = 0, []
    while offset is not None:
        page = query(context,{"op":"rounds","campaign_id":campaign["id"],"offset":offset,"limit":13})
        ids.extend(r["id"] for r in page["rows"])
        offset = page["next_offset"]
    assert len(ids) == 140 and "old-0" in ids
    found = query(context,{"op":"records","query":"earliest-only"})
    assert found["rows"][0]["round_id"] == "old-0"
    with archive(settings.workspace) as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("DELETE FROM campaigns")


@pytest.mark.parametrize("mode", [None, "full_text", "prompt_prefix"])
def test_replay_can_select_an_old_lesson_from_another_campaign(setup_loop, mode):
    settings, service, parent, engine = setup_loop
    if mode is not None:
        class Parent(FakeTeacher):
            def revise(self, lessons):
                return [{**r, "training_tokenization": mode} for r in super().revise(lessons)]
        engine = Engine(settings, Parent, FakeModel)
    engine.run(parent["id"])
    first = service.store.one("SELECT id FROM rounds WHERE campaign_id=? AND number=1", (parent["id"],))
    child = service.continue_campaign(parent["id"])
    assert latest_strategy(settings.workspace,child["id"])["student_notes"]
    class Teacher(AutonomousTeacher):
        def curriculum(self, brief):
            result = plan()
            result["lessons"] = []
            result["replay"] = [{"round_id":first["id"],"lesson_id":"l1"}]
            result["token_mix"] = {"teacher":0,"corpus":0,"replay":1}
            return result
    FakeModel.datasets = []
    Engine(settings, Teacher, FakeModel).run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    assert all(row["origin_round_id"] == first["id"] and row["stream"] == "replay" for rows in FakeModel.datasets for row in rows)
    for rows in FakeModel.datasets:
        for row in rows:
            if mode == "prompt_prefix":
                assert row["training_tokenization"] == mode
                assert row["training_prompt"] == "Explain thermal resistance."
                assert row["text"].startswith(row["training_prompt"])
            else:
                assert "training_tokenization" not in row and "training_prompt" not in row


def test_explicit_prefix_mismatch_fails_before_preparation_without_rewriting(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Teacher(AutonomousTeacher):
        def revise(self, lessons):
            return [{**r, "training_tokenization": "prompt_prefix", "training_text": "A different context"}
                    for r in super().revise(lessons)]
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    failed = service.store.one("SELECT * FROM stage_runs WHERE stage='freeze' AND status='failed'")
    assert "exact student_prompt" in failed["error"]
    assert not FakeModel.datasets


def test_teacher_can_select_sources_outside_configured_prefix(setup_loop):
    settings, service, old, _ = setup_loop
    config = CampaignConfig.model_validate({**old["config"],"source_prefix":"not-present"})
    campaign = service.create("Unrestricted corpus discovery",config)
    context = {"workspace":str(settings.workspace),"corpus_path":config.corpus_path,"campaign_id":campaign["id"]}
    available = query(context,{"op":"sources","limit":1})["rows"]
    document_id = available[0]["id"]
    class Teacher(AutonomousTeacher):
        def curriculum(self, brief):
            result = plan()
            result["lessons"][0]["sources"] = [{"document_id":document_id,"start":5,"length":37}]
            return result
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    source = service.snapshot(campaign["id"])["round"]["lessons"][0]["sources"][0]
    assert source["span_start"] == 5 and len(source["text"]) == 37


def test_continuation_normalizes_corpus_paths_for_the_teacher_working_directory(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    child = service.continue_campaign(campaign["id"], {"corpus_path":"../nekaise-corpus"})
    assert child["config"]["corpus_path"] == str((settings.root.parent/"nekaise-corpus").resolve())


def test_archive_resolves_legacy_relative_corpus_paths_from_repository_root(setup_loop, monkeypatch):
    settings, _, campaign, _ = setup_loop
    root = Path(campaign["config"]["corpus_path"]).parent/"repo"
    monkeypatch.setattr("nekaise_loop.teacher_tools.ROOT", root)
    context = {"workspace":str(settings.workspace),"corpus_path":"../corpus-source","campaign_id":campaign["id"]}
    rows = query(context,{"op":"sources","limit":1})["rows"]
    assert len(query(context,{"op":"source","document_id":rows[0]["id"],"length":37})["text"]) == 37


def test_archive_help_describes_retrievable_context_without_repeating_it(setup_loop, tmp_path):
    settings, _, campaign, engine = setup_loop
    engine.run(campaign["id"])
    path = settings.workspace/"runs"/"fixture"/"recorded-data.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task": {"values": ["first", "last"]}}))
    context = {"workspace": str(settings.workspace), "corpus_path": campaign["config"]["corpus_path"],
               "campaign_id": campaign["id"], "recorded_data_path": str(path)}
    help_result = query(context, {"op": "help"})
    assert "latest_strategy" not in help_result and "operational_context" not in help_result
    assert query(context, help_result["latest_strategy_ref"]) == latest_strategy(settings.workspace, campaign["id"])
    assert query(context, help_result["operations_ref"]) == operational_context(settings.workspace, campaign["id"])
    page = query(context, {"op": "request_data", "pointer": "/task/values", "offset": 1, "limit": 1})
    assert page["value"] == ["last"] and page["next_offset"] is None
    with pytest.raises(ValueError, match="unavailable for this historical call"):
        query({k:v for k,v in context.items() if k != "recorded_data_path"}, {"op": "request_data"})
    outside = tmp_path/"outside"/"recorded-data.json"
    outside.parent.mkdir(); outside.write_text('{}')
    path.unlink(); path.symlink_to(outside)
    with pytest.raises(ValueError, match="belong to this workspace"):
        query(context, {"op": "request_data", "pointer": ""})


@pytest.mark.parametrize("provider", ["codex","claude"])
@pytest.mark.parametrize("large_field", [None, "task", "latest_strategy"])
def test_live_adapter_supplies_handbook_archive_tools_and_strict_decisions(setup_loop, provider, large_field, monkeypatch):
    from nekaise_loop.providers.teacher import CliTeacher
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"],pause=lambda:True)
    row = service.store.one("SELECT id FROM rounds")
    recovery_id = service.store.recover(campaign["id"], "status_review", "Teacher requested diagnostics")
    applied = {"action": "retry", "reason": "Collect generation evidence", "report": "Fixture checks passed; live generation remains unverified."}
    service.store.execute("UPDATE recoveries SET status='resolved',decision=? WHERE id=?", (json.dumps(applied), recovery_id))
    large_data = {"rows": [{"text": "教学 evidence " * 100_000}, {"text": "last record preserved"}]}
    if large_field == "latest_strategy":
        monkeypatch.setattr("nekaise_loop.providers.teacher.latest_strategy", lambda *args: large_data)
    payload = large_data if large_field == "task" else {}
    calls = []
    class Runner:
        def run(self, command, **kwargs):
            calls.append(command)
            saved = json.loads((kwargs["cwd"]/"input.json").read_text())
            assert "full educational authority" in saved["prompt"]
            assert "TEACHING HANDBOOK" in saved["prompt"]
            assert "nekaise_loop.teacher_tools" in saved["prompt"]
            assert saved["inputs"]["operations"]["latest_applied_review"]["decision"] == applied
            assert saved["inputs"]["config_hints"]["student_identity"] == identity
            expected_strategy = large_data if large_field == "latest_strategy" else latest_strategy(settings.workspace, campaign["id"])
            assert saved["inputs"]["latest_strategy"] == expected_strategy
            assert kwargs["stdin"] == saved["prompt"]
            assert len(kwargs["stdin"]) < 1_048_576
            evidence_path = kwargs["cwd"] / "recorded-data.json"
            if large_field:
                from nekaise_loop.artifacts import digest
                evidence = json.loads(evidence_path.read_text())
                assert evidence == saved["inputs"]
                assert evidence[large_field] == large_data
                assert str(evidence_path.resolve()) in kwargs["stdin"]
                assert digest(evidence) in kwargs["stdin"]
                assert "last record preserved" not in kwargs["stdin"]
            else:
                assert json.loads(evidence_path.read_text()) == saved["inputs"]
                assert "teaching_evidence_v1" in kwargs["stdin"]
            assert (settings.workspace/"loop.sqlite3-shm").exists()
            tool_context = json.loads((kwargs["cwd"]/"context.json").read_text())
            assert query(tool_context,{"op":"campaigns"})["rows"]
            assert query(tool_context,{"op":"request_data", "pointer":"/operations/latest_applied_review/decision"})["value"] == applied
            schema = saved["schema"]
            assert set(schema["properties"]["token_mix"]) == {"$ref"}
            mix_schema = schema["$defs"][schema["properties"]["token_mix"]["$ref"].split("/")[-1]]
            assert mix_schema["description"] == "Shares sum to 1; all zero is also valid when train_epochs=0"
            assert set(mix_schema["required"]) == {"teacher","corpus","replay"}
            if provider == "codex":
                assert json.loads(Path(command[command.index("--output-schema")+1]).read_text()) == schema
                Path(command[command.index("--output-last-message")+1]).write_text(json.dumps(plan()))
                assert "--json" in command
                return ""
            assert command[command.index("--tools")+1] == "Read,Glob,Grep,Bash"
            assert json.loads(command[command.index("--json-schema")+1]) == schema
            return json.dumps({"structured_output":plan()})
    identity = {"name": "Kai", "version": "0.0", "developer": "Nekaise", "origin": "Sweden",
                "foundation_model": "openbmb/MiniCPM5-1B-SFT", "charter": "Exact fixture charter.\n"}
    config = CampaignConfig.model_validate({**campaign["config"],"teacher_provider":provider,"student_identity":identity})
    teacher = CliTeacher(config,settings,service.store,campaign["id"],row["id"],Runner(),settings.workspace/"adapter-check")
    assert teacher.curriculum(payload) == Curriculum.model_validate(plan()).model_dump()
    assert len(calls)==1


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("count,width", [(0, 24), (126, 24), (201, 24), (126, 160)])
def test_grade_adapter_binds_coverage_without_limiting_teacher_panel(setup_loop, provider, count, width):
    from nekaise_loop.providers.teacher import CliTeacher
    from nekaise_loop.teaching import Grades
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"], pause=lambda: True)
    row = service.store.one("SELECT id FROM rounds")
    items = [{"id": str(i).zfill(width), "student": "Fixture answer", "dimensions": []}
             for i in range(count)]
    # Reverse order and a non-binary judgment remain Teacher choices.
    grades = [{"id": item["id"], "score": .37, "verdict": "partial",
               "feedback": "Fixture judgment", "gap_type": "teacher-defined",
               "needs_practice": False, "priority": .12, "dimensions": []}
              for item in reversed(items)]
    class Runner:
        def run(self, command, **kwargs):
            saved = json.loads((kwargs["cwd"] / "input.json").read_text())
            assert saved["inputs"]["task"]["items"] == items
            schema = saved["schema"]
            rows = schema["properties"]["rows"]
            assert rows["minItems"] == rows["maxItems"] == count
            id_schema = schema["$defs"]["Grade"]["properties"]["id"]
            if count == 126 and width == 24:
                assert id_schema["enum"] == sorted(item["id"] for item in items)
                assert "e1e5a1635d835a1c9db04489_dummy" not in id_schema["enum"]
                assert 13 < rows["minItems"]  # Recovery 401's partial response.
            else:
                assert "enum" not in id_schema
            if provider == "codex":
                assert json.loads(Path(command[command.index("--output-schema") + 1]).read_text()) == schema
                Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps({"rows": grades}))
                return ""
            assert json.loads(command[command.index("--json-schema") + 1]) == schema
            return json.dumps({"structured_output": {"rows": grades}})
    config = CampaignConfig.model_validate({**campaign["config"], "teacher_provider": provider})
    teacher = CliTeacher(config, settings, service.store, campaign["id"], row["id"],
                         Runner(), settings.workspace / "grade-adapter")
    assert teacher.grade(items) == grades
    generic = Grades.model_json_schema()
    assert "minItems" not in generic["properties"]["rows"]
    assert "enum" not in generic["$defs"]["Grade"]["properties"]["id"]


def test_operational_handoff_follows_applied_lineage_and_preserves_hold(setup_loop):
    settings, service, parent, engine = setup_loop
    engine.run(parent["id"])
    strategy = latest_strategy(settings.workspace, parent["id"])
    child = service.create("Child", CampaignConfig.model_validate(parent["config"]), parent_id=parent["id"])
    sibling = service.create("Sibling", CampaignConfig.model_validate(parent["config"]), parent_id=parent["id"])
    applied = {"action":"continue", "report":"Collect diagnostic evidence; no learning claim."}
    rid = service.store.recover(parent["id"], "status_review", "Teacher pause")
    service.store.execute("UPDATE recoveries SET status='resolved',decision=?,continuation_id=? WHERE id=?", (json.dumps(applied), child["id"], rid))
    other = service.store.recover(parent["id"], "status_review", "Other branch")
    service.store.execute("UPDATE recoveries SET status='resolved',decision=?,continuation_id=? WHERE id=?", (json.dumps({"action":"continue","report":"Sibling only"}), sibling["id"], other))
    pending = service.store.recover(child["id"], "status_review", "Unapplied proposal")
    service.store.execute("UPDATE recoveries SET status='decided',decision=? WHERE id=?", (json.dumps({"action":"retry","report":"Not applied"}), pending))
    service.action(child["id"], "pause", spawn=False, actor="operator", reason="Explicit hold")
    result = operational_context(settings.workspace, child["id"])
    assert result["operator_hold"] == "pause"
    assert result["latest_action"]["actor"] == "operator"
    assert result["latest_applied_review"]["id"] == rid
    assert result["latest_applied_review"]["decision"] == applied
    assert latest_strategy(settings.workspace, child["id"]) == strategy
    # An applied retry on the same campaign supersedes the ancestor handoff.
    service.store.execute("UPDATE recoveries SET status='resolved' WHERE id=?", (pending,))
    assert operational_context(settings.workspace, child["id"])["latest_applied_review"]["id"] == pending


def test_teacher_report_archive_exposes_checks_outcomes_and_all_pages_read_only(setup_loop):
    settings, service, campaign, _ = setup_loop
    context = {"workspace":str(settings.workspace), "campaign_id":campaign["id"], "corpus_path":campaign["config"]["corpus_path"]}
    first = service.store.recover(campaign["id"], "failure", "Original fault")
    service.store.execute("UPDATE recoveries SET status='cancelled' WHERE id=?", (first,))
    second = service.store.recover(campaign["id"], "status_review", "Current investigation")
    directory = settings.workspace/"recoveries"/str(first)/"attempt-1"/"turn-1"
    directory.mkdir(parents=True)
    (directory/"decision.json").write_text(json.dumps({"action":"check", "report":"Unverified proposal"}))
    (directory/"checks.json").write_text(json.dumps({"status":"failed", "error":"Fixture environment failure"}))
    service.store.event(campaign["id"], None, "recovery_applied", "Fixture outcome", {"recovery_id":first})
    with archive(settings.workspace) as db:
        before = list(db.iterdump())
    help_ = query(context, {"op":"help"})
    assert "reports" in help_["operations"] and "report" in help_["operations"]
    page = query(context, {"op":"reports", "limit":1})
    assert page["items"][0]["id"] == second
    older = query(context, {"op":"reports", "limit":1, "before":page["next_before"]})
    assert older["items"][0]["id"] == first and older["next_before"] is None
    full = query(context, {"op":"report", "recovery_id":first})
    assert full["turns"][0]["checks"]["status"] == "failed"
    assert full["turns"][0]["decision"]["action"] == "check"
    assert full["recovery"]["status"] == "cancelled"
    assert any(event["message"] == "Fixture outcome" for event in full["events"])
    with archive(settings.workspace) as db:
        assert list(db.iterdump()) == before
def test_teacher_work_estimate_and_operator_hint_are_not_training_gates(setup_loop):
    settings, service, campaign, _ = setup_loop
    class Teacher(AutonomousTeacher):
        def curriculum(self, brief):
            assert brief["execution_capabilities"]["max_prompts_per_batch"] == 4
            return {**plan(), "work_plan": {"estimated_targets_per_pass": 100000,
                "material_strategy": "Fixture coverage estimate", "dose_rationale": "Diagnostic information justified a small plan"}}
    Engine(settings, Teacher, FakeModel).run(campaign["id"])
    snapshot = service.snapshot(campaign["id"])
    assert snapshot["campaign"]["status"] == "complete"
    work = snapshot["round"]["learning_work"]
    assert work["teacher_work_plan"]["estimated_targets_per_pass"] == 100000
    assert work["preparation"]["targets_per_pass"] < 100000
    assert len(snapshot["round"]["lessons"]) == 1
