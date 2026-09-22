import copy
import json
import random
import pytest

from nekaise_loop.artifacts import canonical, digest
from nekaise_loop.providers.teacher import recorded_prompt
from nekaise_loop.teacher_context import MIN_SHARED_CHARS, EVIDENCE_KEY, evidence_page, evidence_view, resolve_pointer, shared_view


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


@pytest.mark.parametrize("purpose", ["curriculum", "revise", "material_select", "evaluate", "grade", "reflect"])
def test_teaching_view_roundtrips_all_evidence_and_keeps_judgments_and_failures(purpose, tmp_path):
    source = {"id": "fixture", "text": "Synthetic source, not student evidence. " * 100}
    lesson = {"id": "seed", "document": source, "sources": [source], "student_prompt": "Exact user question",
        "student": "", "raw_text": "Exact raw answer", "training_response": "Exact training answer",
        "generated_token_ids": list(range(400)), "generation_audit": {"mismatches": [{"position": 3}], "finite_logits": False},
        "comparison": {"generation": {"text": "Previous exact answer"}}, "stop_reason": "max_new_tokens"}
    data = {"operations": {"operator_hold": "pause", "requests": ["oldest request", "newest request"]},
        "task": {"lessons": [lesson, {**lesson, "id": "last-lesson"}],
                 "usage": {"missing": 2}, "failures": [{"error": "partial author failure"}]}}
    before = canonical(data)
    view = evidence_view(data, purpose, {"op": "request_data"})
    def resolve(value):
        if isinstance(value, dict):
            if set(value) == {EVIDENCE_KEY}:
                ref = value[EVIDENCE_KEY]
                original = resolve_pointer(data, ref["pointer"])
                assert digest(original) == ref["canonical_sha256"]
                return original
            return {k: resolve(v) for k,v in value.items()}
        return [resolve(v) for v in value] if isinstance(value, list) else value
    assert canonical(resolve(view)) == before == canonical(data)
    for original, visible in zip(data["task"]["lessons"], view["task"]["lessons"]):
        for key in ("student_prompt", "student", "raw_text", "training_response", "generation_audit", "comparison", "stop_reason"):
            assert visible[key] == original[key]
        if purpose in {"revise", "evaluate", "grade"}:
            assert visible["sources"][0]["text"] == source["text"]
    assert view["operations"] == data["operations"]
    assert view["task"]["failures"] == data["task"]["failures"]
    prompt = recorded_prompt("prefix", data, tmp_path, purpose=purpose)
    rendered = json.loads(prompt[prompt.index('{"format":'):])
    evidence = rendered["evidence"]
    if evidence.get("format") == "shared_values_v1":
        evidence = expand(evidence)
    assert canonical(resolve(evidence)) == before
    assert canonical(json.loads((tmp_path/"recorded-data.json").read_text())) == before


def test_reference_collisions_preserve_literal_values_and_identical_values_share_locations():
    data = {"task": {"lessons": [{"sources": [{"text": "x"*1000}]}]*2}}
    view = evidence_view(data, "reflect", {"op": "request_data"})
    assert view["task"]["lessons"][0]["sources"] == view["task"]["lessons"][1]["sources"]
    data["literal"] = {EVIDENCE_KEY: {"pointer": "this is literal data"}}
    assert evidence_view(data, "reflect", {"op": "request_data"}) == data


def test_reference_metadata_can_be_passed_as_a_query_without_overriding_page_limits(tmp_path):
    data = {"task": {"lessons": [{"sources": [{"text": "文"*25000}]}]}}
    view = evidence_view(data, "reflect", {"op": "request_data"})
    ref = view["task"]["lessons"][0]["sources"][0]["text"][EVIDENCE_KEY]
    page = evidence_page(data, ref)
    assert page["total"] == 25000 and page["next_start"] == 4000 and len(page["value"]) == 4000
    data["literal"] = {EVIDENCE_KEY: "literal"}
    prompt = recorded_prompt("prefix", data, tmp_path, purpose="reflect")
    rendered = json.loads(prompt[prompt.index('{"format":'):])
    assert rendered["evidence_references"] is False


@pytest.mark.parametrize("purpose", ["revise", "evaluate"])
def test_grounding_keeps_document_text_when_no_identical_inline_source_exists(purpose):
    text = "Only grounding passage " * 100
    data = {"task": {"lessons": [{"document": {"text": text}, "sources": None},
                                 {"document": {"text": text}, "sources": [{"text": text+"different span"}]}]}}
    view = evidence_view(data, purpose, {"op": "request_data"})
    assert view == data


@pytest.mark.parametrize("purpose", ["evaluate", "reflect"])
def test_trusted_synthetic_view_references_targets_but_preserves_observed_student_evidence(purpose):
    source = {"text": "Fixture source "*200}
    text = "Exact authored target "*100
    seed = {"id": "seed", "student_prompt": "Exact observed task", "student": "Actual fixture answer", "teacher": text,
            "training_response": text, "sources": [source], "document": source}
    trusted = {**seed, "id": "trusted", "student": None, "student_observation": "not_requested",
               "material_origin": {"type": "auxiliary_synthetic", "review_policy": "trusted_author_v1", "plan_id": "job"}}
    legacy = {**trusted, "id": "legacy", "material_origin": {"type": "auxiliary_synthetic"}}
    data = {"task": {"lessons": [seed, trusted, legacy], "assessment": {"student": text, "grade": {"score": 0}}}}
    view = evidence_view(data, purpose, {"op": "request_data"})
    rows = view["task"]["lessons"]
    assert rows[0]["teacher"] == rows[2]["teacher"] == text
    ref = rows[1][EVIDENCE_KEY]
    assert resolve_pointer(data, ref["pointer"]) == trusted
    assert ref["canonical_sha256"] == digest(trusted)
    assert ref["coverage"]["student_prompt"] == trusted["student_prompt"]
    assert ref["coverage"]["student"] is None and ref["coverage"]["material_origin"] == trusted["material_origin"]
    assert ref["coverage"]["training_response_chars"] == len(text)
    assert view["task"]["assessment"] == data["task"]["assessment"]
    if purpose == "evaluate":
        assert rows[0]["sources"] == rows[2]["sources"] == [source]
    trusted["student"] = ""  # Even an actually observed empty answer remains eager.
    assert EVIDENCE_KEY not in evidence_view(trusted, purpose, {"op": "request_data"})
    trusted["student"] = None
    trusted["errors"] = ["Fixture runtime failure"]
    assert evidence_view(trusted, purpose, {"op": "request_data"})["errors"] == trusted["errors"]


def test_pointer_pages_preserve_escaped_names_types_and_the_final_element():
    data = {"a/b~": {"text": "教学α"*9000, "values": [None, False, 1, 1.0, "last"]}}
    pointer = "/a~1b~0/text"
    text, start = "", 0
    while start is not None:
        page = evidence_page(data, {"pointer": pointer, "start": start, "length": 20000})
        assert page["canonical_sha256"] == digest(data["a/b~"]["text"])
        text += page["value"]
        start = page["next_start"]
    assert text == data["a/b~"]["text"]
    assert evidence_page(data, {"pointer": "/a~1b~0/values", "offset": 4, "limit": 1})["value"] == ["last"]
    assert evidence_page(data, {"pointer": "/a~1b~0", "fields": ["values"]})["value"] == {"values": data["a/b~"]["values"]}
    for invalid in ["x", "/bad~2escape", "/a~1b~0/values/-1", "/a~1b~0/values/01", "/absent", "/a~1b~0/values/99"]:
        with pytest.raises(ValueError):
            resolve_pointer(data, invalid)
    with pytest.raises(ValueError):
        evidence_page(data, {"pointer": pointer, "length": 20001})
    with pytest.raises(ValueError, match="fields are absent"):
        evidence_page(data, {"pointer": "", "fields": ["absent"]})
