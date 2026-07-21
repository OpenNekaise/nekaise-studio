"""The holdout↔exam guard — the assertion that keeps the fitness function pointed at the
right building. Runs against synthetic data; never touches real nekaise_data."""
import json
from pathlib import Path

import pytest

from conftest import REPO, load_module


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Two synthetic buildings + a frozen exam that is clearly about 'bravo'."""
    data = tmp_path / "nekaise_data"
    for b in ("alpha", "bravo"):
        (data / b).mkdir(parents=True)
        (data / b / "model.ttl").write_text("ex:X a s223:Thing .")
    exam = tmp_path / "eval_open.jsonl"
    rows = [{"id": f"q{i}", "question": "?", "ground_truth": "see /data/bravo/trends.csv",
             "anchors": ["bravo_GT41"], "source": "bravo_model.ttl"} for i in range(4)]
    exam.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setenv("NEKAISE_DATA", str(data))
    monkeypatch.delenv("NEKAISE_ALLOW_HOLDOUT_MISMATCH", raising=False)
    prepare = load_module(REPO / "gym" / "tasks" / "building" / "prepare.py")
    return prepare, exam


def test_default_holdout_is_first_alphabetically(env):
    prepare, _ = env
    assert prepare.default_holdout() == "alpha"   # the very footgun the guard exists for


def test_exam_building_inferred_from_rows(env):
    prepare, exam = env
    assert prepare.exam_building(exam) == "bravo"


def test_mismatch_fails_loudly(env):
    prepare, exam = env
    with pytest.raises(SystemExit, match="holdout mismatch"):
        prepare.require_holdout_matches_exam("alpha", exam)


def test_match_passes(env):
    prepare, exam = env
    assert prepare.require_holdout_matches_exam("bravo", exam) == "bravo"


def test_no_exam_means_no_guard(env, tmp_path):
    prepare, _ = env
    missing = tmp_path / "nope.jsonl"
    assert prepare.require_holdout_matches_exam("alpha", missing) == "alpha"


def test_explicit_override_env(env, monkeypatch):
    prepare, exam = env
    monkeypatch.setenv("NEKAISE_ALLOW_HOLDOUT_MISMATCH", "1")
    assert prepare.require_holdout_matches_exam("alpha", exam) == "alpha"
