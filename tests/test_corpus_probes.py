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


def test_split_is_explicit_and_source_documents_are_disjoint():
    raw = [json.loads(line) for line in
           (REPO / "gym/tasks/corpus_probes/probes.jsonl").read_text().splitlines()]
    dev_docs = {row["doc_id"] for row in raw if row["split"] == "dev"
                and row["kind"] == "transfer"}
    frozen_docs = {row["doc_id"] for row in raw if row["split"] == "frozen"}
    absorption_docs = {row["doc_id"] for row in raw if row["kind"] == "absorption"}
    assert dev_docs and frozen_docs and absorption_docs
    assert dev_docs.isdisjoint(frozen_docs)
    assert dev_docs.isdisjoint(absorption_docs)
    assert frozen_docs.isdisjoint(absorption_docs)
    for row in raw[:50]:
        assert probes.in_split(row["id"], row["split"])


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
