"""CoAPT round-toolkit guardrails: the ratio-locked token ledger, the frozen QA text
format (and its exact match with the closed-book diagnosis template), chunk helpers, and
the explicit-gate contract. CPU-only; the tokenizer-dependent mixer path is exercised by
the smoke round, not here."""
from __future__ import annotations

import json

import pytest

from conftest import REPO, load_module
from experiments.coapt.build_data import (
    QA_TEMPLATE, chunk_chars, continuation_prefix, fill_by_cycling, gated_rows,
    plan_mix, stable_uniform,
)

MIX = {"raw": 0.40, "teacher_cpt": 0.25, "qa_text": 0.15, "anchor": 0.20}


def test_plan_mix_is_ratio_locked_on_teacher_volume():
    plan = plan_mix(250, MIX, cap=10_000)
    assert plan["total"] == 1000                      # 250 / 0.25
    assert plan["raw"] == 400 and plan["anchor"] == 200
    assert plan["teacher_cpt"] == 250 and plan["qa_text"] == 150


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
    assert total == 140 and passes == 3               # a,b,a,b then a would overflow
    assert [r["text"] for r in rows] == ["a", "b", "a", "b"]
    assert rows[0]["meta"]["pass"] == 1 and rows[-1]["meta"]["pass"] == 2
    assert fill_by_cycling([], 100) == ([], 0, 1)     # never loops on empty input


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
