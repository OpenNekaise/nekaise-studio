"""corpus_probes pack — referee contract + frozen-split determinism guardrails."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
from pack import load  # noqa: E402

probes = load("corpus_probes")


def test_contract():
    for fn in ("load_split", "is_correct", "reward", "extract_answer"):
        assert callable(getattr(probes, fn))


def test_splits_disjoint_and_cover():
    dev = {r["id"] for r in probes.load_split("dev")}
    frozen = {r["id"] for r in probes.load_split("frozen")}
    allr = {r["id"] for r in probes.load_split("all")}
    assert dev and frozen
    assert not dev & frozen
    assert dev | frozen == allr


def test_split_rule_is_deterministic():
    # id-hash md5 % 5 == 0 -> frozen; pin the rule (not the minted ids, which are
    # corpus-dependent) so a silent rule change can't reshuffle recorded metrics
    import hashlib
    for r in probes.load_split("all", 50):
        frozen = int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 5 == 0
        assert probes.in_split(r["id"], "frozen") is frozen


def test_rows_are_completion_ready():
    rows = probes.load_split("dev", 5)
    for r in rows:
        gold = json.loads(r["answer"])
        assert r["question"] and not r["question"].endswith(tuple("0123456789"))
        assert r["track"] in ("absorption", "transfer")
        float(gold["value"])  # numeric gold


def test_grading():
    gold = json.dumps({"value": "45", "answer": "45 °C"})
    assert probes.is_correct("45 °C in the supply line", gold)
    assert probes.is_correct(" 45.0 degrees", gold)
    assert not probes.is_correct("44 °C", gold)
    assert not probes.is_correct("no number here", gold)
    assert probes.reward("45", gold) == 1.0
    assert probes.extract_answer("about 1,600 times") == "1600"
