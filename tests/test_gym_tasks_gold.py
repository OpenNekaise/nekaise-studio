"""R1 intake standard: every task ships prompt + reference solution + verifier binding,
and THE GOLD MUST PASS ITS OWN VERIFIER. A referee its own gold can't satisfy is broken."""
from __future__ import annotations

from pathlib import Path

import pytest

from gym import tasks

REPO = Path(__file__).resolve().parents[1]


def test_corpus_probes_gold_passes():
    rows = tasks.load("corpus_probes", split="all")
    failures = tasks.intake_check(rows, sample=300)
    assert not failures, failures[:10]


def test_toy_agentic_gold_passes():
    rows = tasks.load("toy_agentic", split="dev", n=4)
    assert len(rows) == 4
    failures = tasks.intake_check(rows)
    assert not failures, failures


@pytest.mark.skipif(
    not (REPO.parent / "nekaise-bench" / "questions.jsonl").exists(),
    reason="external nekaise-bench clone not present")
def test_bench_gold_passes():
    rows = tasks.load("bench", split="all")
    failures = tasks.intake_check(rows, sample=150)
    assert not failures, failures[:10]


def test_splits_deterministic():
    dev = {t.id for t in tasks.load("corpus_probes", split="dev")}
    frozen = {t.id for t in tasks.load("corpus_probes", split="frozen")}
    allr = {t.id for t in tasks.load("corpus_probes", split="all")}
    assert dev and frozen and dev.isdisjoint(frozen) and dev | frozen == allr
