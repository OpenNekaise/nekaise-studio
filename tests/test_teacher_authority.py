import json
from pathlib import Path
import sqlite3

import pytest

from conftest import FakeTeacher, FakeModel
from nekaise_loop.config import CampaignConfig
from nekaise_loop.engine import Engine
from nekaise_loop.storage import now
from nekaise_loop.teaching import Curriculum
from nekaise_loop.teacher_tools import archive, query, latest_strategy


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


def test_replay_can_select_an_old_lesson_from_another_campaign(setup_loop):
    settings, service, parent, engine = setup_loop
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


@pytest.mark.parametrize("provider", ["codex","claude"])
def test_live_adapter_supplies_handbook_archive_tools_and_strict_decisions(setup_loop, provider):
    from nekaise_loop.providers.teacher import CliTeacher
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"],pause=lambda:True)
    row = service.store.one("SELECT id FROM rounds")
    calls = []
    class Runner:
        def run(self, command, **kwargs):
            calls.append(command)
            saved = json.loads((kwargs["cwd"]/"input.json").read_text())
            assert "full educational authority" in saved["prompt"]
            assert "TEACHING HANDBOOK" in saved["prompt"]
            assert "nekaise_loop.teacher_tools" in saved["prompt"]
            assert (settings.workspace/"loop.sqlite3-shm").exists()
            tool_context = json.loads((kwargs["cwd"]/"context.json").read_text())
            assert query(tool_context,{"op":"campaigns"})["rows"]
            schema = saved["schema"]
            assert set(schema["$defs"]["TokenMix"]["required"]) == {"teacher","corpus","replay"}
            if provider == "codex":
                Path(command[command.index("--output-last-message")+1]).write_text(json.dumps(plan()))
                assert "--json" in command
                return ""
            assert command[command.index("--tools")+1] == "Read,Glob,Grep,Bash"
            return json.dumps({"structured_output":plan()})
    config = CampaignConfig.model_validate({**campaign["config"],"teacher_provider":provider})
    teacher = CliTeacher(config,settings,service.store,campaign["id"],row["id"],Runner(),settings.workspace/"adapter-check")
    assert teacher.curriculum({}) == Curriculum.model_validate(plan()).model_dump()
    assert len(calls)==1
