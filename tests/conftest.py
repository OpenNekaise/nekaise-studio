import hashlib
import json
from pathlib import Path

import pytest

from nekaise_loop.config import CampaignConfig, Settings
from nekaise_loop.engine import Engine
from nekaise_loop.service import Service


@pytest.fixture(autouse=True)
def isolated_source_lock(tmp_path, monkeypatch):
    # Recovery runs tests while holding the production source lock. Fixtures must
    # exercise their own locks, without interacting with live workers or repairs.
    monkeypatch.setattr("nekaise_loop.ownership.LOCK_DIRECTORY", tmp_path/"source-locks")


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus-source"
    (root/"registry").mkdir(parents=True)
    (root/"manifest").mkdir()
    (root/"corpus").mkdir()
    (root/"registry/eligibility.json").write_text(json.dumps({"version": 1, "restrictions": {}}))
    rows = []
    for i in range(24):
        sid = f"crawl-energyplus-docs-{i:03d}-heat-transfer"
        body = (f"Building energy study {i}. Heat transfer depends on the temperature difference and thermal resistance. "
                "The thermal resistance is measured in kelvin per watt. A larger resistance reduces heat flow. "
                "Use consistent units when calculating energy and power. ") * 4
        (root/"corpus"/f"{sid}.md").write_text(body)
        rows.append({"id": sid, "title": f"Heat transfer and building energy {i}", "url": f"https://example.org/{i}", "source": "fixture", "license": "cc-by", "topic": "building_energy", "status": "ok", "corpus_sha256": hashlib.sha256(body.encode()).hexdigest()})
    (root/"manifest/curated.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return root


class FakeTeacher:
    fail_gate_once = False
    reject = False

    def __init__(self, config, settings, store, campaign_id, round_id, runner, directory):
        self.config, self.settings, self.store, self.campaign_id = config, settings, store, campaign_id

    def curriculum(self, brief):
        from nekaise_loop.corpus import search_sources
        prior = self.store.query("SELECT id FROM rounds WHERE campaign_id=? AND number<? ORDER BY number", (self.campaign_id, brief["round_number"]))
        previous = [lesson for row in prior for lesson in self.store.records(row["id"], "lesson")]
        candidates = search_sources(Path(self.config.corpus_path), prefix=self.config.source_prefix, limit=200)["rows"]
        seen = {r["document"]["id"] for r in previous}
        candidates.sort(key=lambda r: r["id"] in seen)
        documents = [{**r, "selection_reason": "Addresses previous teacher gaps" if previous else "Fixture teacher choice"} for r in candidates[:self.config.lessons_per_round]]
        refs = [{"document_id": d["id"], "start": 0, "length": self.config.passage_chars} for d in documents]
        lessons = [{"id": f"l{i+1}", "kind": "cpt" if i%2==0 else "sft", "sources": [refs[i]], "concept": "Thermal resistance", "prompt": "Explain thermal resistance.", "student_prompt": "Explain thermal resistance.", "reason": d["selection_reason"]} for i,d in enumerate(documents)]
        return {"lessons": lessons, "readings": refs, "replay": [{"round_id":r["id"],"lesson_id":"l1"} for r in prior], "token_mix": self.config.token_mix.model_dump(), "train_epochs": 1, "evaluation_instructions": "Test thermal resistance", "notes": "Fixture teaching strategy"}

    def revise(self, lessons):
        if self.fail_gate_once:
            type(self).fail_gate_once = False
            raise RuntimeError("Temporary teacher failure")
        return [{"id": r["id"], "text": "Thermal resistance is measured in kelvin per watt. Higher resistance reduces heat flow.", "training_text": f"{r['student_prompt']}\nThermal resistance is measured in kelvin per watt. Higher resistance reduces heat flow.", "errors": ["Confused resistance with conductance"], "evidence": [r["document"]["text"][:90]], "use_for_training": True, "reason": "Teach the missing unit"} for r in lessons]

    def gate(self, lessons):
        if self.fail_gate_once:
            type(self).fail_gate_once = False
            raise RuntimeError("Temporary teacher failure")
        return [{"id": r["id"], "passed": not self.reject, "reason": "Checked every claim against the passage"} for r in lessons]

    def evaluate(self, curriculum, lessons):
        return [{"id": f"e{i+1}", "sources": curriculum["lessons"][i]["sources"], "question": "Which unit describes thermal resistance?", "student_prompt": "Which unit describes thermal resistance?", "reference": "HIDDEN_REFERENCE kelvin per watt", "rubric": ["Names kelvin per watt"], "evidence": "", "concept": "thermal resistance units"} for i in range(self.config.eval_questions)]

    def grade(self, items):
        return [{"id": r["id"], "score": 0.25, "verdict": "partial", "feedback": "Missing the correct unit", "gap_type": "knowledge", "needs_practice": True, "priority": 0.75} for r in items]

    def reflect(self, observations):
        return {"student_notes": "Fixture student still needs unit practice", "next_round_instructions": "Practice thermal resistance", "action": "continue", "reason": "More teaching needed"}


class FakeModel:
    prompts = []
    datasets = []

    def __init__(self, config, settings, runner, directory):
        self.config, self.directory = config, directory

    def generate(self, checkpoint, rows, on_answer):
        self.prompts.extend(rows)
        answers = [{"id": r["id"], "text": "Resistance is measured in watts.", "tokens": 7, "seconds": 0.01} for r in rows]
        for answer in answers:
            on_answer(answer)
        return answers

    def train(self, checkpoint, dataset, dataset_hash, on_metric):
        self.datasets.append(dataset["rows"])
        for i in range(1, self.config.train_steps+1):
            on_metric({"step": i, "total_steps": self.config.train_steps, "loss": 3.0/i, "learning_rate": self.config.learning_rate, "tokens": i*100, "tokens_per_second": 100, "elapsed_seconds": i, "gpu_memory_gb": 0})
        path = self.directory/"checkpoint"
        path.mkdir()
        (path/"model.safetensors").write_bytes(b"TEST FAKE WEIGHTS")
        manifest = {"dataset_hash": dataset_hash, "parent": checkpoint, "files": {"model.safetensors": hashlib.sha256(b"TEST FAKE WEIGHTS").hexdigest()}}
        (path/"checkpoint.json").write_text(json.dumps(manifest))
        return {"checkpoint": str(path), "manifest": manifest}

    def prepare(self, checkpoint, rows):
        from nekaise_loop.training import prepare_dataset
        return prepare_dataset(rows, FakeTokenizer(), self.config.model_dump())


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(c)+2 for c in text]


@pytest.fixture
def setup_loop(tmp_path, corpus):
    FakeTeacher.fail_gate_once = False
    FakeTeacher.reject = False
    FakeModel.prompts = []
    FakeModel.datasets = []
    settings = Settings(tmp_path/"workspace")
    base = tmp_path/"base-model"
    base.mkdir()
    (base/"model.safetensors").write_bytes(b"TEST FAKE BASE WEIGHTS")
    config = CampaignConfig(student_model=str(base), corpus_path=str(corpus), rounds=2, lessons_per_round=2, eval_questions=2, train_steps=3, auto_recover=False, manage_history=False)
    service = Service(settings)
    campaign = service.create("Integration test", config)
    engine = Engine(settings, FakeTeacher, FakeModel)
    return settings, service, campaign, engine
