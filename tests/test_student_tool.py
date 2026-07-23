"""tools/student.py contract — the CPU-pure parts: NLL aggregation over vLLM
prompt_logprobs, the frozen closed-book template, and JSONL row validation. The vLLM
engine itself runs only in the eval environment (smoke round)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from conftest import REPO, load_module

student = load_module(REPO / "tools" / "student.py")


def test_mean_nll_skips_first_token_and_averages():
    token_ids = [11, 22, 33]
    logprobs = [None, {22: SimpleNamespace(logprob=-1.0)}, {33: -3.0}]
    nll, count = student.mean_nll(token_ids, logprobs)
    assert nll == 2.0 and count == 2


def test_mean_nll_handles_missing_and_empty():
    assert student.mean_nll([1, 2], [None, {}]) == (None, 0)
    assert student.mean_nll([], None) == (None, 0)


def test_answer_prompt_is_the_frozen_closed_book_format():
    assert student.answer_prompt(" Which method runs on exit? ") == \
        "Question: Which method runs on exit?\nAnswer:"


def test_read_rows_validates_schema(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(json.dumps({"id": "a", "question": "Q?"}) + "\n")
    assert student.read_rows(path, "question")[0]["id"] == "a"

    path.write_text(json.dumps({"id": "a", "question": "  "}) + "\n")
    with pytest.raises(SystemExit):                   # required field empty
        student.read_rows(path, "question")
    path.write_text(json.dumps({"question": "Q?"}) + "\n")
    with pytest.raises(SystemExit):                   # id is mandatory
        student.read_rows(path, "question")
    path.write_text("\n")
    with pytest.raises(SystemExit):                   # empty input refuses
        student.read_rows(path, "question")
