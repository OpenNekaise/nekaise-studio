"""building task set — deterministically-verifiable ontology/topology QA over nekaise_data.

Cross-building split: test = the HOLDOUT building (unseen), train = the rest.

R10 — holdout is CODE-ENFORCED here: NEKAISE_HOLDOUT must be set explicitly or every
building entry point refuses to run (no silent first-folder fallback — that historically
graded the wrong building and leaked training data into the exam). The frozen-exam match
guard (prepare.require_holdout_matches_exam) still applies on top.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from gym.tasks import Task

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402  (fixed index builder, same directory)


def holdout() -> str:
    h = os.environ.get("NEKAISE_HOLDOUT")
    if not h:
        raise SystemExit(
            "NEKAISE_HOLDOUT is not set — building-pack work refuses to run without an "
            "explicit holdout (a silent fallback grades the wrong building and leaks the "
            "exam building into training).\n"
            "Fix: set NEKAISE_HOLDOUT=<holdout-building-folder> in .env (see .env.example); "
            "folders live under nekaise_data/.")
    return prepare.require_holdout_matches_exam(h)


def _ref_for(gold: str) -> str:
    kind, _, body = gold.partition(":")
    if kind == "count":
        return body
    parts = [x for x in body.split("|") if x.strip()]
    return parts[0] if kind == "type" else ", ".join(parts)


def _tasks_for(building: str, ents: list[dict]) -> list[Task]:
    from collections import Counter
    out: list[Task] = []

    def add(qid: str, question: str, gold: str) -> None:
        out.append(Task(id=qid, prompt=question, verifier="ontology_qa",
                        meta={"gold": gold}, ref_solution=_ref_for(gold), mode="chat",
                        tags={"building": building, "kind": gold.partition(":")[0]}))

    for i, e in enumerate(ents):
        if e["types"] and e["comment"]:
            add(f"{building}-type-{i}",
                f"In building '{building}', what ontology class (type) is the entity "
                f"described as \"{e['comment']}\"? Answer with the class name.",
                "type:" + "|".join(e["types"]))
    for cls, cnt in Counter(t for e in ents for t in e["types"]).items():
        add(f"{building}-count-{cls}",
            f"In building '{building}', how many entities of ontology class '{cls}' are there?",
            f"count:{cnt}")
    for i, e in enumerate(ents):
        if e["connections"] and e["comment"]:
            add(f"{building}-conns-{i}",
                f"In building '{building}', what is the entity described as \"{e['comment']}\" "
                f"connected to (s223:cnx)? List the connection-point names.",
                "conns:" + "|".join(e["connections"]))
    return out


def load(split: str = "test", n: int = 0) -> list[Task]:
    hold = holdout()
    idx = prepare.load_index()
    if split == "test":
        pool = _tasks_for(hold, idx.get(hold, []))
    else:
        pool = [t for b, ents in idx.items() if b != hold for t in _tasks_for(b, ents)]
    random.Random(3407).shuffle(pool)
    return pool[:n] if n else pool
