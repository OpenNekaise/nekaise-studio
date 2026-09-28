"""Real control-plane integration with explicitly fake models, Authors and web pages."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from conftest import FakeTeacher, FakeModel, FakeTokenizer
from test_chat_serialization import ChatTokenizer
from test_material_authors import author
from nekaise_loop.author_config import AuthorPool
from nekaise_loop.config import CampaignConfig, ROOT
from nekaise_loop.curriculum_progress import assignment, commit_progress, initial_state
from nekaise_loop.curriculum_research import fetch_source, public_url
from nekaise_loop.general_curriculum import import_curriculum, project_curriculum
from nekaise_loop.progressive_preparation import prepare_progressive
from nekaise_loop.storage import encode, now
from nekaise_loop.training import update_batches
from nekaise_loop.engine import Engine


def test_import_preserves_teaching_taxonomy_but_discards_runtime_and_benchmark_routing(tmp_path):
    path = ROOT / "curricula/general_purpose_curriculum.json"
    projection = project_curriculum(path)
    assert len(projection["domains"]) == 20 and len(projection["units"]) == 265
    assert projection["source_sha256"] == "56e047acde55c0e8bf4eac5df9207deba770fa5c554e2d3a398262f74a109c5d"
    assert projection["units"][0]["id"] == "foundations.task_contracts"
    serialized = json.dumps(projection)
    for forbidden in ("benchmark_leaf_ids", "assessment_profile", "benchmark_profiles", "learner_state", "mastery_threshold", "grading_context_refs"):
        assert forbidden not in serialized
    raw = json.loads(path.read_text())
    raw["learning_order"][-1] = raw["learning_order"][0]
    invalid = tmp_path / "bad.json"
    invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exactly once"):
        project_curriculum(invalid)


class ProgressiveTeacher(FakeTeacher):
    omit = None
    fail_evaluate = False
    diagnostic = False
    seen = []

    def research(self, brief):
        return {"queries": brief["unit"]["search_queries"][:1], "sources": [
            {"url": "https://reference.example/guide", "title": "Fixture guide", "purpose": "Explain the concept"}],
            "teaching_direction": "Original examples"}

    def curriculum(self, brief):
        work = brief["progression"]
        type(self).seen.append(work)
        unit = work["unit"]["id"]
        plan = super().curriculum(brief)
        plan["lessons"] = [plan["lessons"][0]]
        plan["lessons"][0].update(learning_track="gpc", curriculum_unit_id=unit, kind="sft",
                                  material_scope="general_chat", sources=[], student_prompt="Explain the next concept.")
        plan.update(readings=[], replay=[], train_epochs=0 if self.diagnostic else 1,
                    forward_corpus_share=.5, token_mix={"teacher":.01,"corpus":.99,"replay":0})
        plan["expansion_jobs"] = [{"id": aid, "author_id": aid, "seed_ids": ["l1"],
            "learning_track": "gpc", "curriculum_unit_id": unit, "material_scope": "general_prose",
            "instructions": "Write an original explanation", "expected_items": 1, "max_output_tokens": 512}
            for aid in work["required_authors"] if aid != self.omit] if not self.diagnostic else []
        return plan

    def revise(self, lessons):
        return [{"id": r["id"], "text": "Fixture explanation", "training_text": "",
                 "training_tokenization": "chat_response", "training_response": "A practical explanation using evidence and clear units.",
                 "errors": [], "evidence": [], "use_for_training": not self.diagnostic, "reason": "Forward teaching"} for r in lessons]

    def evaluate(self, curriculum, lessons):
        if self.fail_evaluate:
            raise RuntimeError("Fixture assessment interruption")
        return [{"id":"check", "sources":[], "question":"What evidence supports a claim?",
                 "student_prompt":"What evidence supports a claim?", "reference":"Relevant observations",
                 "rubric":["Connects evidence to claim"], "concept":"Evidence", "evidence":"Fixture reference"}]


class ProgressiveModel(FakeModel):
    wrong_counters = False

    def prepare(self, checkpoint, rows):
        return prepare_progressive(rows, ChatTokenizer(), self.config.model_dump(), [1,4])

    def train(self, checkpoint, dataset, dataset_hash, on_metric):
        result = super().train(checkpoint, dataset, dataset_hash, on_metric)
        batches = list(update_batches(dataset["samples"], self.config.tokens_per_update, self.config.train_epochs, self.config.seed))
        result["manifest"].update(tokens=sum(len(s["input_ids"])-1 for b in batches for s in b) + int(self.wrong_counters), steps=len(batches))
        (Path(result["checkpoint"])/"checkpoint.json").write_text(json.dumps(result["manifest"]))
        return result


def configured(setup_loop, monkeypatch, **changes):
    settings, service, original, _ = setup_loop
    policy = import_curriculum(settings.workspace, ROOT/"curricula/general_purpose_curriculum.json", span_chars=128, corpus_window_chars=1024)
    projection = service.artifacts.get(policy.projection_artifact)
    projection["units"] = projection["units"][:2]
    policy = policy.model_copy(update={"projection_artifact": service.artifacts.put(projection)})
    config = CampaignConfig.model_validate({**original["config"], "train_steps":0, "rounds":3,
        "curriculum_loop":policy.model_dump(), "expansion_policy":"required_v1", "general_material_policy":"required_v1",
        "material_review_policy":"trusted_author_v1", "material_authors":AuthorPool(authors=[author(a) for a in "abc"]).model_dump(), **changes})
    campaign = service.create("Progression integration fixture", config)
    ProgressiveTeacher.omit = None
    ProgressiveTeacher.fail_evaluate = False
    ProgressiveTeacher.diagnostic = False
    ProgressiveTeacher.seen = []
    ProgressiveModel.wrong_counters = False
    def fetch(request, **kwargs):
        text = "Fixture reference about teaching task contracts and evidence. " * 3
        return {"id":"research-fixture", "title":"Fixture guide", "url":request["url"], "text":text,
                "span_start":0,"span_length":len(text),"document_chars":len(text),"source_sha256":hashlib.sha256(text.encode()).hexdigest()}
    monkeypatch.setattr("nekaise_loop.curriculum_research.fetch_source", fetch)
    calls = []
    def response(request):
        task = json.loads(json.loads(request.content)["messages"][1]["content"])["task"]
        calls.append(task)
        assert task["curriculum_unit"]["id"] == task["job"]["curriculum_unit_id"]
        assert any(s["id"] == "research-fixture" for s in task["sources"].values())
        row = {"id":"example", "kind":"cpt", "concept":"Forward concept", "training_text":
               "Original fixture explanation from " + task["job"]["author_id"] + ". " + "Use evidence to support the conclusion. "*3,
               "training_tokenization":"full_text", "source_keys":list(task["sources"])[:1], "rationale":"Varied original explanation"}
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"rows":[row]})}}],
                                        "usage":{"prompt_tokens":50,"completion_tokens":40}})
    factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(response), **kwargs)
    engine = Engine(settings, ProgressiveTeacher, ProgressiveModel, material_client_factory=factory)
    return settings, service, campaign, engine, calls


def test_both_loops_move_all_authors_train_and_gpc_cycles_despite_low_scores(setup_loop, monkeypatch):
    _, service, campaign, engine, calls = configured(setup_loop, monkeypatch)
    engine.run(campaign["id"])
    current = service.store.campaign(campaign["id"])
    assert current["status"] == "complete", current["error"]
    progress = service.snapshot(campaign["id"])["curriculum_progress"]
    assert progress["state"]["gpc_completed"] == 3
    assert progress["gpc_cycle"] == 1
    assert progress["state"]["chars_trained"] > 0
    assert set(progress["last_receipt"]["targets"]["forward_authors"]) == set("abc")
    assert len(calls) == 9
    assert [w["unit_position"] for w in ProgressiveTeacher.seen] == [1,2,1]
    receipts = service.store.query("SELECT * FROM curriculum_receipts ORDER BY sequence")
    assert len(receipts) == 3
    assert all(r["score"] == .25 for r in service.snapshot(campaign["id"])["rounds"])
    with service.store.connect(immediate=True) as db:
        item = receipts[-1]
        commit_progress(db, item["round_id"], service.artifacts.get(item["artifact"]), item["artifact"])
    assert service.snapshot(campaign["id"])["curriculum_progress"]["state"]["gpc_completed"] == 3


@pytest.mark.parametrize("fault", ["author", "counter", "evaluation", "diagnostic"])
def test_progress_credits_only_verified_training_not_selection_or_assessment(setup_loop, monkeypatch, fault):
    _, service, campaign, engine, _ = configured(setup_loop, monkeypatch, rounds=1)
    if fault == "author": ProgressiveTeacher.omit = "c"
    if fault == "counter": ProgressiveModel.wrong_counters = True
    if fault == "evaluation": ProgressiveTeacher.fail_evaluate = True
    if fault == "diagnostic": ProgressiveTeacher.diagnostic = True
    engine.run(campaign["id"])
    progress = service.snapshot(campaign["id"])["curriculum_progress"]
    assert progress["state"]["gpc_completed"] == (1 if fault == "evaluation" else 0)
    if fault != "diagnostic":
        assert service.store.campaign(campaign["id"])["status"] == "failed"
    else:
        assert service.store.campaign(campaign["id"])["status"] == "complete"


def test_failed_selection_reuses_assignment_and_research_in_continuation(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = configured(setup_loop, monkeypatch, rounds=1)
    ProgressiveTeacher.omit = "c"
    engine.run(campaign["id"])
    first = service.store.one("SELECT * FROM curriculum_assignments")
    child = service.continue_campaign(campaign["id"], start=False)
    ProgressiveTeacher.omit = None
    engine.run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    assert service.store.one("SELECT * FROM curriculum_assignments") == first
    assert service.snapshot(child["id"])["curriculum_progress"]["state"]["gpc_completed"] == 1


def test_continuation_after_assessment_failure_moves_forward_and_old_branch_cannot_claim_progress(setup_loop, monkeypatch):
    _, service, campaign, engine, _ = configured(setup_loop, monkeypatch, rounds=1)
    ProgressiveTeacher.fail_evaluate = True
    engine.run(campaign["id"])
    assert service.snapshot(campaign["id"])["curriculum_progress"]["state"]["gpc_completed"] == 1
    child = service.continue_campaign(campaign["id"], start=False)
    ProgressiveTeacher.fail_evaluate = False
    engine.run(child["id"])
    assert service.store.campaign(child["id"])["status"] == "complete"
    assert service.snapshot(child["id"])["curriculum_progress"]["state"]["gpc_completed"] == 2
    older = service.continue_campaign(campaign["id"], start=False)
    before = len(ProgressiveTeacher.seen)
    engine.run(older["id"])
    result = service.store.campaign(older["id"])
    assert result["status"] == "failed" and "older-weight branch" in result["error"]
    assert len(ProgressiveTeacher.seen) == before  # No paid call or coverage claim.


@pytest.mark.parametrize("change", [None, {"namespace":"bypass"}, {"remediation_cap":.3}, {"projection_artifact":"f"*64}])
def test_recovery_cannot_bypass_operator_progression_contract(setup_loop, monkeypatch, change):
    from nekaise_loop.service import Conflict
    _, service, campaign, _, _ = configured(setup_loop, monkeypatch)
    policy = None if change is None else {**campaign["config"]["curriculum_loop"], **change}
    with pytest.raises(Conflict,match="preserve forward progression"):
        service.continue_campaign(campaign["id"], {"curriculum_loop":policy}, actor="orchestrator", start=False)


def test_full_pass_partitions_every_document_and_new_arrivals_enter_next_pass(setup_loop):
    settings, service, campaign, _ = setup_loop
    policy = import_curriculum(settings.workspace, ROOT/"curricula/general_purpose_curriculum.json", span_chars=256, corpus_window_chars=2000000)
    config = CampaignConfig.model_validate({**campaign["config"],"train_steps":0,"curriculum_loop":policy.model_dump()})
    ctx = SimpleNamespace(config=config, engine=SimpleNamespace(settings=settings), store=service.store, artifacts=service.artifacts, cancelled=lambda:False)
    first = assignment(ctx)
    assert assignment(ctx) == first
    by_doc = {}
    for span in first["spans"]:
        by_doc.setdefault(span["document_id"], []).append(span)
    assert len(by_doc) == 24
    for document_id, spans in by_doc.items():
        raw = (Path(config.corpus_path)/"corpus"/(document_id+".md")).read_text()
        assert ''.join(s["text"] for s in spans) == raw
        assert [s["span_start"] for s in spans] == list(range(0,len(raw),256))
    assert first["spans"][-1]["after"]["corpus_cycle"] == 1
    assert first["spans"][-1]["after"]["inventory"] is None
    # Selection itself never credits coverage.
    state = json.loads(service.store.one("SELECT state FROM curriculum_progress")["state"])
    assert state == initial_state()
    root = Path(config.corpus_path)
    raw = "New corpus material. " * 50
    (root/"corpus/new.md").write_text(raw)
    with (root/"manifest/curated.jsonl").open("a") as handle:
        handle.write(json.dumps({"id":"new","status":"ok","license":"cc-by","source":"fixture","corpus_sha256":hashlib.sha256(raw.encode()).hexdigest()})+'\n')
    # Model a committed cursor at the pass boundary; no legacy coverage inference.
    service.store.execute("UPDATE curriculum_progress SET sequence=1,state=?", (encode(first["spans"][-1]["after"]),))
    second = assignment(ctx)
    assert "new" in {s["document_id"] for s in second["spans"]}
    assert "new" not in by_doc


def test_zero_length_source_ref_keeps_rest_of_document_semantics(setup_loop):
    from nekaise_loop.stages import _sources
    settings, service, campaign, _ = setup_loop
    policy = import_curriculum(settings.workspace, ROOT/"curricula/general_purpose_curriculum.json", span_chars=256, corpus_window_chars=1024)
    config = CampaignConfig.model_validate({**campaign["config"],"train_steps":0,"curriculum_loop":policy.model_dump()})
    ctx = SimpleNamespace(config=config, engine=SimpleNamespace(settings=settings), store=service.store, artifacts=service.artifacts, cancelled=lambda:False)
    work = assignment(ctx)
    ctx.progression = {"assignment_artifact":service.artifacts.put(work),"research_artifact":service.artifacts.put({"sources":[]})}
    doc = work["spans"][0]["document_id"]
    raw = (Path(config.corpus_path)/"corpus"/(doc+".md")).read_text()
    result = _sources(ctx,[{"document_id":doc,"start":300,"length":0}])[0]
    assert result["text"] == raw[300:]
    with pytest.raises(ValueError,match="empty"):
        _sources(ctx,[{"document_id":doc,"start":len(raw)+20,"length":0}])


def test_inventory_tampering_never_becomes_source_coverage(setup_loop):
    from nekaise_loop.curriculum_inventory import build_inventory, next_document
    settings, _, campaign, _ = setup_loop
    inventory = build_inventory(Path(campaign["config"]["corpus_path"]), settings.workspace)
    assert next_document(settings.workspace, inventory, "")
    path = settings.workspace/"curriculum/inventories"/(inventory["sha256"]+".sqlite3")
    with path.open("ab") as handle:
        handle.write(b"CORRUPT")
    with pytest.raises(ValueError,match="integrity"):
        next_document(settings.workspace,inventory,"")


def test_whole_rows_are_never_ratio_cropped_or_repeated_and_followup_cap_is_real():
    policy = {"forward_corpus_share":.5,"remediation_cap":.2}
    config = {**CampaignConfig().model_dump(),"curriculum_loop":policy}
    rows = [{"id":"g","stream":"cpt","text":"g"*100,"learning_track":"gpc","curriculum_unit_id":"unit"},
            *[{"id":str(i),"stream":"corpus","text":"c"*30,"learning_track":"corpus","curriculum_span":True} for i in range(20)]]
    data = prepare_progressive(rows,FakeTokenizer(),config)
    assert len(data["progressive_preparation"]["corpus_row_ids"]) == 4
    assert data["ledger"]["streams"]["teacher"]["prepared_tokens"] == 101
    assert all(part["repeated_tokens"] == 0 for part in data["ledger"]["streams"].values())
    assert all(len(r["text"]) == 30 for r in data["rows"] if r.get("curriculum_span"))
    with pytest.raises(ValueError, match="Remediation exceeds"):
        prepare_progressive(rows + [{"id":"r","stream":"replay","text":"r"*100,"learning_track":"remediation"}],FakeTokenizer(),config)
    with pytest.raises(ValueError, match="complete passes"):
        CampaignConfig(curriculum_loop={"namespace":"test","projection_artifact":"a"*64}, train_steps=1)


def test_research_fetch_records_real_hashes_excerpts_and_checks_redirect_targets(monkeypatch):
    monkeypatch.setattr("nekaise_loop.curriculum_research.socket.getaddrinfo", lambda *a,**k:[(0,0,0,"",("93.184.216.34",443))])
    body = b"<html><script>do not teach this</script><p>Actual retrieved reference. " + b"Evidence matters. "*20 + b"</p></html>"
    factory = lambda **kwargs:httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,headers={"content-type":"text/html"},content=body)),**kwargs)
    source = fetch_source({"url":"https://example.com","title":"Real response fixture","purpose":"Test evidence"}, client_factory=factory)
    assert source["source_sha256"] == hashlib.sha256(body).hexdigest()
    assert "do not teach" not in source["text"]
    assert source["retrieved_at"] and source["span_length"] == len(source["text"])
    monkeypatch.setattr("nekaise_loop.curriculum_research.socket.getaddrinfo", lambda *a,**k:[(0,0,0,"",("127.0.0.1",443))])
    with pytest.raises(ValueError,match="non-public"):
        public_url("https://example.com")
