import copy
import json
import random

from nekaise_loop.artifacts import canonical, digest
from nekaise_loop.providers.teacher import recorded_prompt
from nekaise_loop.teacher_context import MIN_SHARED_CHARS, shared_view


def expand(view):
    def resolve(value):
        if isinstance(value, dict):
            if set(value) == {view["reference_key"]}:
                return view["shared_values"][value[view["reference_key"]]]
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(view["data"])


def repeated_evidence(count=66):
    source = {"id": "fixture-source", "text": "Synthetic evidence, not student results. " * 100}
    return {"task": {"lessons": [{"id": f"example-{i}", "sources": [copy.deepcopy(source)],
        "student_prompt": f"Explain fixture {i}.", "training_response": f"Exact answer {i}."}
        for i in range(count)]}, "history": [{"id": "old-but-accessible"}, {"id": "last-record"}]}


def test_sharing_preserves_every_exact_record_without_mutating_input():
    data = repeated_evidence()
    before = canonical(data)
    view = shared_view(data)
    assert canonical(data) == before == canonical(expand(view))
    assert len(view["data"]["task"]["lessons"]) == 66
    assert view["data"]["history"][-1]["id"] == "last-record"
    assert len(canonical(view)) < len(before) * .15
    assert canonical(shared_view(copy.deepcopy(data))) == canonical(view)
    assert len(view["shared_values"]) == 1  # The outer repeated list owns its original children.


def test_near_duplicates_and_scalar_types_are_not_conflated():
    base = {"text": "x" * MIN_SHARED_CHARS, "number": 1, "maybe": None}
    variants = [base, {**base, "text": base["text"] + "y"}, {**base, "number": 1.0},
                {k:v for k,v in base.items() if k != "maybe"}]
    data = [copy.deepcopy(x) for x in variants for _ in range(2)]
    view = shared_view(data)
    assert len(view["shared_values"]) == 4
    assert canonical(expand(view)) == canonical(data)


def test_small_unique_and_marker_colliding_inputs_keep_original_representation():
    assert shared_view(["x" * (MIN_SHARED_CHARS - 3)] * 2) is None
    assert shared_view({"only_once": "x" * MIN_SHARED_CHARS}) is None
    collision = repeated_evidence()
    collision["literal"] = {"$shared": "user-data"}
    assert shared_view(collision) is None


def test_generated_nested_json_roundtrips():
    randomizer = random.Random(29)
    for _ in range(30):
        repeated = {"text": "Evidence α" * 400, "value": randomizer.choice([None, 1, 1.0, True, "x"])}
        data = {"rows": [{"nested": [copy.deepcopy(repeated), randomizer.randint(0, 99)],
                          "tail": {"value": randomizer.choice([None, False, "tail"])}}
                         for _ in range(randomizer.randint(2, 15))]}
        assert canonical(expand(shared_view(data))) == canonical(data)


def test_rendered_prompt_keeps_original_file_hash_and_full_inline_shared_evidence(tmp_path):
    data = repeated_evidence()
    before = canonical(data)
    prompt = recorded_prompt("Trusted teacher prefix", data, tmp_path)
    view = json.loads(prompt[prompt.index('{"format":'):])
    original = json.loads((tmp_path / "recorded-data.json").read_text())
    assert before == canonical(data) == canonical(original) == canonical(expand(view))
    assert view["original_data"]["canonical_sha256"] == digest(data)
    assert "last-record" in prompt and "exact original value" in prompt
    assert len(prompt) < len(before) * .2


def test_original_large_context_stays_externalized_with_optional_lossless_view(tmp_path):
    data = repeated_evidence(300)
    assert len(json.dumps(data)) > 900_000
    prompt = recorded_prompt("prefix", data, tmp_path)
    assert "externalized in full" in prompt and len(prompt) < 3000
    assert json.loads((tmp_path / "recorded-data.json").read_text()) == data
    view = json.loads((tmp_path / "shared-data.json").read_text())
    assert canonical(expand(view)) == canonical(data)
    assert "lossless_shared_data_path" in prompt
