from copy import deepcopy

import pytest

from conftest import FakeModel, FakeTeacher, FakeTokenizer
from nekaise_loop.artifacts import Artifacts, digest
from nekaise_loop.engine import Engine
from nekaise_loop.prompt_evidence import prompt_training_prefixes
from nekaise_loop.training import prepare_dataset


def test_missing_omitted_chunked_and_repeated_samples_are_described_without_mutation():
    lessons = [
        {"id": "row", "student_prompt": "Answer:", "prompt_token_ids": [1, 2, 47], "prompt_truncated": True},
        {"id": "missing", "student_prompt": "Answer:"},
        {"id": "omitted", "student_prompt": "Answer:", "prompt_token_ids": [1, 2, 47]},
    ]
    dataset = {
        "rows": [{"id": "row", "stream": "sft", "text": "Answer:\nresult"}],
        "samples": [
            {"row_id": "row", "stream": "teacher", "input_ids": [1, 2]},
            {"row_id": "row", "stream": "teacher", "input_ids": [2, 990, 3]},
            {"row_id": "row", "stream": "teacher", "input_ids": [1, 2]},
            {"row_id": "row", "stream": "replay", "input_ids": [1, 2, 47, 3]},
        ],
    }
    original = deepcopy((lessons, dataset))
    result = prompt_training_prefixes(lessons, dataset)["lessons"]
    assert (lessons, dataset) == original
    assert result[0]["text_prefix_matches"] is True
    assert result[0]["prompt_truncated"] is True
    comparisons = result[0]["sample_comparisons"]
    assert [c["sample_index"] for c in comparisons] == [0, 1, 2]
    assert [c["common_prefix_tokens"] for c in comparisons] == [2, 0, 2]
    assert comparisons[0]["sample_ends_before_prompt"] is True
    assert not any(c["prompt_is_sample_prefix"] for c in comparisons)
    assert result[1]["status"] == "prompt_tokens_unavailable"
    assert result[2]["status"] == "no_frozen_samples"
    assert result[2]["text_prefix_matches"] is None


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize("epochs", [0, 1])
@pytest.mark.parametrize("mode", ["full_text", "prompt_prefix"])
def test_frozen_prefix_evidence_reaches_reflection_without_changing_teacher_choices(setup_loop, newline, epochs, mode):
    settings, service, campaign, _ = setup_loop
    prompt = "Task: fixture\nAnswer:" + ("\n" if newline else "")
    training_text = "Task: fixture\nAnswer:\nresult"
    observed = []

    class BoundaryTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=True):
            # Test-only merger reproduces the distinction between ':' and ':\n'.
            return super().encode(text.replace(":\n", "\u03de"), add_special_tokens)

    class Teacher(FakeTeacher):
        def curriculum(self, brief):
            return {"lessons": [{"id": "row", "kind": "sft", "sources": [], "concept": "fixture",
                    "prompt": "fixture", "student_prompt": prompt, "reason": "chosen format"}],
                    "readings": [], "replay": [], "token_mix": {"teacher": 1, "corpus": 0, "replay": 0},
                    "train_epochs": epochs, "evaluation_instructions": "none", "notes": "fixture"}

        def revise(self, lessons):
            assert lessons[0]["student_prompt"] == prompt
            return [{**r, "training_text": training_text, "training_tokenization": mode} for r in super().revise(lessons)]

        def evaluate(self, curriculum, lessons):
            return []

        def reflect(self, observations):
            observed.append(observations)
            return {"student_notes": "fixture", "next_round_instructions": "fixture", "action": "complete", "reason": "fixture"}

    class Model(FakeModel):
        def generate(self, checkpoint, rows, on_answer):
            assert rows == [{"id": "row", "prompt": prompt}]
            result = {"id": "row", "text": "fixture", "prompt_token_ids": BoundaryTokenizer().encode(prompt), "prompt_truncated": False}
            on_answer(result)
            return [result]

        def prepare(self, checkpoint, rows):
            assert rows[0]["text"] == training_text
            if mode == "prompt_prefix":
                assert rows[0]["training_prompt"] == prompt
            else:
                assert "training_prompt" not in rows[0]
            return prepare_dataset(rows, BoundaryTokenizer(), self.config.model_dump())

        def train(self, checkpoint, dataset, dataset_hash, on_metric):
            prepared = {k: dataset[k] for k in ("rows", "samples", "ledger")}
            assert dataset_hash == digest(prepared)
            return super().train(checkpoint, dataset, dataset_hash, on_metric)

    Engine(settings, Teacher, Model).run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert len(observed) == 1
    evidence = observed[0]["prompt_training_prefixes"]
    lesson = evidence["lessons"][0]
    assert lesson["text_prefix_matches"] is True
    assert lesson["sample_comparisons"][0]["prompt_is_sample_prefix"] is (newline or mode == "prompt_prefix")
    assert lesson["training_tokenization"] == mode
    assert lesson["prompt_text_tail"] == prompt
    stage = service.store.one("SELECT artifact FROM stage_runs WHERE stage='freeze' AND status='complete'")
    frozen = Artifacts(settings.workspace).get(stage["artifact"])
    assert evidence == frozen["prompt_training_prefixes"]
    assert observed[0]["frozen_dataset_hash"] == frozen["dataset_hash"]
    assert bool(FakeModel.datasets) is bool(epochs)
