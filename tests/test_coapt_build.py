"""CoAPT round-toolkit guardrails: the ratio-locked token ledger, the frozen QA text
format (and its exact match with the closed-book diagnosis template), chunk helpers, and
the explicit-gate contract. CPU-only; the tokenizer-dependent mixer path is exercised by
the smoke round, not here."""
from __future__ import annotations

import json

import pytest

from conftest import REPO, load_module
from experiments.coapt.build_data import (
    QA_TEMPLATE, chunk_chars, continuation_prefix, corpus_text_path, fill_by_cycling,
    gated_rows, plan_mix, source_chunk_raw_stream, stable_uniform, stratified_chunks,
)

MIX = {"raw": 0.40, "teacher_cpt": 0.25, "qa_text": 0.15, "anchor": 0.20}
TEACHER_ONLY_DOMAIN_MIX = {
    "raw": 0.0, "teacher_cpt": 0.80, "qa_text": 0.0, "anchor": 0.20,
}


def test_plan_mix_is_ratio_locked_on_teacher_volume():
    plan = plan_mix(250, MIX, cap=10_000)
    assert plan["total"] == 1000                      # 250 / 0.25
    assert plan["raw"] == 400 and plan["anchor"] == 200
    assert plan["teacher_cpt"] == 250 and plan["qa_text"] == 150

    teacher_plan = plan_mix(800, TEACHER_ONLY_DOMAIN_MIX, cap=1_000)
    assert teacher_plan == {
        "total": 1000, "raw": 0, "qa_text": 0, "anchor": 200,
        "teacher_cpt": 800,
    }


def test_plan_mix_refuses_cap_and_empty_and_bad_shares():
    with pytest.raises(SystemExit):
        plan_mix(250, MIX, cap=999)                   # planned volume over the target
    with pytest.raises(SystemExit):
        plan_mix(0, MIX, cap=10_000)                  # no gated supervision
    with pytest.raises(SystemExit):
        plan_mix(1, {**MIX, "raw": 0.5}, cap=10_000)  # shares must sum to 1.0


def test_fill_by_cycling_repeats_to_a_strict_bound():
    items = [({"text": "a", "meta": {"content_tokens": 30}}, 30),
             ({"text": "b", "meta": {"content_tokens": 40}}, 40)]
    rows, total, passes = fill_by_cycling(items, 150)
    assert total == 140 and passes == 2               # two productive passes; third adds none
    assert [r["text"] for r in rows] == ["a", "b", "a", "b"]
    assert rows[0]["meta"]["pass"] == 1 and rows[-1]["meta"]["pass"] == 2
    assert fill_by_cycling([], 100) == ([], 0, 0)     # never loops on empty input
    one_pass_rows, one_pass_total, one_passes = fill_by_cycling(
        items, 150, max_passes=1)
    assert [r["text"] for r in one_pass_rows] == ["a", "b"]
    assert one_pass_total == 70 and one_passes == 1


def test_qa_template_matches_the_closed_book_diagnosis_template():
    student = load_module(REPO / "tools" / "student.py")
    qa = QA_TEMPLATE.format(question="Q?", answer="A.")
    prompt = student.answer_prompt("Q?")
    assert qa.startswith(prompt), "training text must begin with the diagnosis prompt"
    assert qa == prompt + " A."


def test_chunk_chars_never_splits_words_and_prefix_is_a_prefix():
    text = "alpha beta gamma delta epsilon zeta " * 40
    chunks = chunk_chars(text, 100)
    assert chunks and all(chunks)
    for chunk in chunks:
        assert " ".join(chunk.split()) in " ".join(text.split())
    prefix = continuation_prefix(chunks[0])
    assert chunks[0].startswith(prefix) and 0 < len(prefix) < len(chunks[0])
    assert not prefix.endswith(" ")


def test_stratified_chunks_cover_the_document_not_only_the_prefix():
    text = "aaa bbb ccc ddd eee fff"
    all_chunks = chunk_chars(text, 4)
    selected = stratified_chunks(text, 4, 3)
    assert [index for index, _ in selected] == [0, 2, len(all_chunks) - 1]
    assert selected[0][1] == all_chunks[0] and selected[-1][1] == all_chunks[-1]


def test_corpus_text_path_prefers_declared_cleaned_text(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "text").mkdir()
    cleaned = tmp_path / "corpus" / "d.md"
    raw = tmp_path / "text" / "d.md"
    cleaned.write_text("clean")
    raw.write_text("raw")
    row = {"id": "d", "corpus_path": "corpus/d.md", "text_path": "text/d.md"}
    assert corpus_text_path(tmp_path, row) == cleaned

    bad_hash = {**row, "corpus_sha256": "0" * 64}
    with pytest.raises(SystemExit, match="cleaned corpus hash mismatch"):
        corpus_text_path(tmp_path, bad_hash)

    cleaned.unlink()
    with pytest.raises(SystemExit, match="cleaned corpus text missing"):
        corpus_text_path(tmp_path, row)

    legacy = {"id": "d", "text_path": "text/d.md"}
    assert corpus_text_path(tmp_path, legacy) == raw


def test_raw_control_uses_each_teacher_source_span_once():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    prompts = [
        {"doc_id": "d", "chunk_index": 0, "source_chunk": "one two",
         "source_sha256": "h", "probe_type": "continuation"},
        {"doc_id": "d", "chunk_index": 0, "source_chunk": "one two",
         "source_sha256": "h", "probe_type": "summary"},
        {"doc_id": "d", "chunk_index": 1, "source_chunk": "three four",
         "source_sha256": "h", "probe_type": "continuation"},
    ]
    rows, total, passes = source_chunk_raw_stream(
        prompts, Tokenizer(), target=4, rnd=0, max_passes=1)
    assert [row["text"] for row in rows] == ["one two", "three four"]
    assert total == 4 and passes == 1


def test_gated_rows_requires_explicit_verdicts(tmp_path):
    path = tmp_path / "cpt_teacher.jsonl"
    rows = [
        {"id": "a", "doc_id": "d1", "teacher_text": "x", "gate": {"passed": True}},
        {"id": "b", "doc_id": "d1", "teacher_text": "y", "gate": {"passed": False}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    passed, failed = gated_rows(path, ("teacher_text", "doc_id"))
    assert [r["id"] for r in passed] == ["a"] and failed == 1

    path.write_text(json.dumps({"id": "c", "doc_id": "d1", "teacher_text": "z"}) + "\n")
    with pytest.raises(SystemExit):                   # a row with no gate refuses
        gated_rows(path, ("teacher_text", "doc_id"))
    path.write_text(json.dumps(
        {"id": "d", "doc_id": "", "teacher_text": "z", "gate": {"passed": True}}) + "\n")
    with pytest.raises(SystemExit):                   # required fields must be non-empty
        gated_rows(path, ("teacher_text", "doc_id"))


def test_stable_uniform_is_deterministic_and_in_unit_interval():
    a, b = stable_uniform(3407, "doc-1"), stable_uniform(3407, "doc-1")
    assert a == b and 0.0 < a <= 1.0
    assert stable_uniform(3407, "doc-1") != stable_uniform(3407, "doc-2")
