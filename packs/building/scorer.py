"""building pack — the FIXED referee for ontology/topology Q&A over nekaise_data. Do NOT edit.

Mints deterministically-verifiable questions from the parsed ontology index (prepare.py)
and grades answers against the graph. The gold is encoded with a kind prefix so the same
`is_correct(pred, gold)` contract supports several question shapes:

    type:<Class>[|<Class>...]   - the entity's ontology class(es); correct if any is named
    count:<int>                 - how many of a class exist; correct on exact integer
    conns:<name>[|<name>...]    - what an entity connects to (s223:cnx); graded by fraction

Cross-building eval: the held-out building (NEKAISE_HOLDOUT, default "rio10") is the test
split; the rest are train. This measures generalization to an UNSEEN building — the real
"approach Opus" bar. Editing this file would cheat the metric (see skills/run-experiment.md).

Contract: load_split / is_correct / reward / extract_answer.
"""
from __future__ import annotations

import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402  (fixed index builder)

# Which building is held out for evaluation (cross-building generalization).
HOLDOUT_BUILDING = os.environ.get("NEKAISE_HOLDOUT", "rio10")


# ------------------------------ task generation -------------------------------
def _tasks_for(building: str, ents: list[dict]) -> list[dict]:
    """Deterministic ontology/topology questions with graph-derived gold."""
    tasks: list[dict] = []

    # type_of — only entities with a human-readable comment (so the question is answerable
    # by someone reading the ontology, not by guessing an opaque URI).
    for e in ents:
        if e["types"] and e["comment"]:
            tasks.append({
                "question": f"In building '{building}', what ontology class (type) is the entity "
                            f"described as \"{e['comment']}\"? Answer with the class name.",
                "answer": "type:" + "|".join(e["types"]),
            })

    # count_class — one per class present in the building.
    for cls, n in Counter(t for e in ents for t in e["types"]).items():
        tasks.append({
            "question": f"In building '{building}', how many entities of ontology class '{cls}' are there?",
            "answer": f"count:{n}",
        })

    # connections_of — topology (s223:cnx).
    for e in ents:
        if e["connections"] and e["comment"]:
            tasks.append({
                "question": f"In building '{building}', what is the entity described as \"{e['comment']}\" "
                            f"connected to (s223:cnx)? List the connection-point names.",
                "answer": "conns:" + "|".join(e["connections"]),
            })

    return tasks


def _all_tasks() -> dict[str, list[dict]]:
    return {b: _tasks_for(b, ents) for b, ents in prepare.load_index().items()}


def load_split(split: str = "test", n: int | None = 200) -> list[dict]:
    """Cross-building split: test = held-out building, train = the rest. Rows {question, answer}."""
    by_b = _all_tasks()
    if split == "test":
        pool = list(by_b.get(HOLDOUT_BUILDING, []))
    else:
        pool = [t for b, ts in by_b.items() if b != HOLDOUT_BUILDING for t in ts]
    random.Random(3407).shuffle(pool)
    return pool[:n] if n else pool


# --------------------------------- grading ------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def extract_answer(text: str) -> str:
    return text.strip()


def reward(prediction: str, gold_answer: str) -> float:
    """Graded reward in [0,1]. Exact for counts; any-class for type; fraction for connections."""
    kind, _, body = gold_answer.partition(":")
    if kind == "count":
        nums = re.findall(r"-?\d+", prediction)
        return 1.0 if nums and nums[-1] == body else 0.0

    pred = f" {_norm(prediction)} "
    expected = [_norm(x) for x in body.split("|") if x.strip()]
    if not expected:
        return 0.0
    hits = sum(1 for x in expected if f" {x} " in pred)
    if kind == "type":
        return 1.0 if hits >= 1 else 0.0           # naming any correct class counts
    return hits / len(expected)                    # conns: fraction of connections named


def is_correct(prediction: str, gold_answer: str) -> bool:
    """Used as eval metric and rejection-sampling filter: fully correct only."""
    return reward(prediction, gold_answer) >= 0.999


def load_test(n: int | None = 200) -> list[dict]:
    """Back-compat alias."""
    return load_split("test", n)
