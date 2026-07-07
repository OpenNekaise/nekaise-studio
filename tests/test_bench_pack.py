"""The bench pack — split canon + contract, tested against a STUB bench dir (no real clone).

The stub's grading mimics the bench harness's API shape; grading correctness itself is the
bench repo's responsibility. What must never silently change here is the SPLIT RULE — every
recorded score depends on it — so a fingerprint of known ids is pinned below.
"""
import json

import pytest

from conftest import REPO, load_module

STUB_HARNESS = '''
MCQ_SYSTEM = "mcq system"
OPEN_SYSTEM = "open system"
def mcq_prompt(q):
    lines = [q["question"], ""]
    lines += [f"{'ABCDEFGH'[i]}. {c}" for i, c in enumerate(q["choices"])]
    return "\\n".join(lines + ["", "Answer with the single letter only."])
def strip_think(reply):
    return reply.split("</think>", 1)[1].strip() if "</think>" in reply else reply.strip()
def extract_letter(reply, n, choices):
    for ch in reply:
        if ch in "ABCDEFGH":
            return ch
    return None
def grade_open(pred, gold, aliases):
    p = pred.lower()
    return any(c.lower() in p for c in [gold] + list(aliases or []))
'''

QUESTIONS = [
    {"id": "nb-0001", "track": "open", "topic": "t", "difficulty": "easy",
     "question": "Q open?", "answer": "42 kW", "aliases": ["42kW"]},
    {"id": "nb-0009", "track": "mcq", "topic": "t", "difficulty": "hard",
     "question": "Q mcq?", "choices": ["x", "y"], "answer": "B"},
]


@pytest.fixture()
def bench(tmp_path, monkeypatch):
    (tmp_path / "eval_ollama.py").write_text(STUB_HARNESS)
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in QUESTIONS))
    monkeypatch.setenv("NEKAISE_BENCH_DIR", str(tmp_path))
    return load_module(REPO / "packs" / "bench" / "scorer.py")


def test_split_fingerprint_is_pinned(bench):
    # Changing the hash rule silently breaks comparability of every recorded score.
    pinned = {"nb-0001": "dev", "nb-0002": "dev", "nb-0003": "dev", "nb-0004": "dev",
              "nb-0005": "dev", "nb-0006": "dev", "nb-0007": "dev", "nb-0008": "dev",
              "nb-0009": "test", "nb-0010": "dev", "nb-0011": "dev", "nb-0012": "dev"}
    for qid, want in pinned.items():
        assert bench.in_split(qid, want), f"{qid} moved out of {want} — split rule changed!"
        assert not bench.in_split(qid, "test" if want == "dev" else "dev")


def test_splits_are_disjoint_and_cover(bench):
    dev = {r["id"] for r in bench.load_split("dev")}
    test = {r["id"] for r in bench.load_split("test")}
    allq = {r["id"] for r in bench.load_split("all")}
    assert dev | test == allq and not dev & test


def test_rows_carry_prompt_ready_questions_and_json_gold(bench):
    rows = {r["id"]: r for r in bench.load_split("all")}
    mcq = rows["nb-0009"]
    assert "A. x" in mcq["question"] and "single letter" in mcq["question"]
    gold = json.loads(mcq["answer"])
    assert gold == {"track": "mcq", "answer": "B", "choices": ["x", "y"]}
    assert json.loads(rows["nb-0001"]["answer"])["aliases"] == ["42kW"]


def test_grading_contract(bench):
    rows = {r["id"]: r for r in bench.load_split("all")}
    assert bench.is_correct("B", rows["nb-0009"]["answer"])
    assert not bench.is_correct("A", rows["nb-0009"]["answer"])
    assert bench.reward("the answer is 42 kW", rows["nb-0001"]["answer"]) == 1.0
    assert bench.reward("no idea", rows["nb-0001"]["answer"]) == 0.0
    assert bench.extract_answer("<think>x</think> B") == "B"
