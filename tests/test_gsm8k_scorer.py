"""The gsm8k referee contract — the fitness function the loop trusts. Pure, no dataset download."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from pack import load  # noqa: E402

scorer = load("gsm8k")


def test_extract_answer_takes_final_marked_number():
    assert scorer.extract_answer("reasoning... #### 42") == "42"
    assert scorer.extract_answer("#### 1,234.") == "1234"
    assert scorer.extract_answer("a 3 then 7 at the end") == "7"
    assert scorer.extract_answer("no numbers here") is None


def test_is_correct_matches_final_numbers():
    gold = "Step 1... #### 18"
    assert scorer.is_correct("the answer is #### 18", gold)
    assert scorer.is_correct("so 18", gold)
    assert not scorer.is_correct("so 19", gold)
    assert not scorer.is_correct("", gold)


def test_reward_is_graded_and_bounded():
    gold = "#### 5"
    assert scorer.reward("#### 5", gold) == 1.0
    assert scorer.reward("#### 6", gold) == 0.1     # well-formed but wrong
    assert scorer.reward("gibberish", gold) == 0.0
    for pred in ("#### 5", "#### 6", "x"):
        assert 0.0 <= scorer.reward(pred, gold) <= 1.0


def test_pack_loader_verifies_contract():
    with pytest.raises(FileNotFoundError):
        load("no-such-pack")
