from __future__ import annotations

import json

import pytest

from tools.codex_teacher import (
    author_payload, authored_is_current, batches, cmd_merge, cmd_sample, cmd_shard,
    gate_is_current, gate_payload, read_jsonl, validate_response, write_jsonl_atomic,
)


def test_batches_preserve_order():
    rows = [{"id": str(i)} for i in range(5)]
    assert [[row["id"] for row in batch] for batch in batches(rows, 2)] == [
        ["0", "1"], ["2", "3"], ["4"],
    ]


def test_jsonl_checkpoint_roundtrip_and_duplicate_guard(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"id": "a", "text": "å"}, {"id": "b", "text": "β"}]
    write_jsonl_atomic(path, rows)
    assert read_jsonl(path) == rows

    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(SystemExit, match="duplicate ids"):
        read_jsonl(path)


def test_validate_response_requires_exact_ids_and_order():
    payload = {"rows": [{"id": "a"}, {"id": "b"}]}
    assert validate_response(payload, ["a", "b"]) == payload["rows"]
    with pytest.raises(ValueError, match="ids/order mismatch"):
        validate_response(payload, ["b", "a"])


def test_author_and_gate_payloads_expose_only_grounding_inputs():
    draft = {
        "id": "x", "source_chunk": "The source says 7.",
        "draft": "The draft says 9.", "secret": "not forwarded",
    }
    assert author_payload(draft) == {
        "id": "x", "source_chunk": "The source says 7.",
        "student_draft": "The draft says 9.",
    }
    teacher = {"id": "x", "source_chunk": "Source.", "teacher_text": "Text."}
    assert gate_payload(teacher) == {
        "id": "x", "source_chunk": "Source.", "teacher_text": "Text.",
    }


def test_author_payload_treats_empty_student_draft_as_valid_failure():
    assert author_payload({"id": "x", "source_chunk": "Source.", "draft": ""}) == {
        "id": "x", "source_chunk": "Source.", "student_draft": "",
    }


def test_resume_requires_content_and_model_fingerprint_match():
    source = {"id": "x", "source_chunk": "S", "source_sha256": "h", "draft": "D"}
    authored = {
        "id": "x", "source_chunk": "S", "source_sha256": "h",
        "student_draft": "D", "teacher_text": "T",
        "teacher_model": "m", "teacher_reasoning": "low",
    }
    assert authored_is_current(authored, source, model="m", reasoning="low")
    assert not authored_is_current(authored, {**source, "draft": "changed"},
                                   model="m", reasoning="low")

    gated = {**authored, "gate": {"model": "g", "reasoning": "low"}}
    assert gate_is_current(gated, authored, model="g", reasoning="low")
    assert not gate_is_current(gated, {**authored, "teacher_text": "changed"},
                               model="g", reasoning="low")


def test_sample_is_deterministic_and_seeded(tmp_path):
    source = tmp_path / "in.jsonl"
    rows = [{"id": str(i)} for i in range(20)]
    write_jsonl_atomic(source, rows)

    class Args:
        input = source
        size = 5
        seed = 3407

    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    Args.output = first
    cmd_sample(Args)
    Args.output = second
    cmd_sample(Args)
    assert read_jsonl(first) == read_jsonl(second)
    assert len(read_jsonl(first)) == 5


def test_shards_merge_back_to_reference_order(tmp_path):
    source = tmp_path / "in.jsonl"
    rows = [{"id": str(i)} for i in range(11)]
    write_jsonl_atomic(source, rows)

    class ShardArgs:
        input = source
        shards = 3

    shard_paths = []
    for index in range(3):
        ShardArgs.shard_index = index
        ShardArgs.output = tmp_path / f"s{index}.jsonl"
        cmd_shard(ShardArgs)
        shard_paths.append(ShardArgs.output)

    class MergeArgs:
        reference = source
        inputs = list(reversed(shard_paths))
        output = tmp_path / "merged.jsonl"

    cmd_merge(MergeArgs)
    assert read_jsonl(MergeArgs.output) == rows
