"""ontology_qa — graph-derived building questions (type / count / connections).

meta: {"gold": "<kind>:<body>"} with the kind-prefixed gold from the building task minter:

    type:<Class>[|<Class>...]   entity's ontology class(es); 1.0 if ANY is named
    count:<int>                 exact final integer match
    conns:<name>[|<name>...]    fraction of connection-point names present

Ported verbatim from packs/building/scorer.py's grading half.
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def verify(prompt: str, response: str, meta: dict) -> float:
    gold = str(meta.get("gold", ""))
    kind, _, body = gold.partition(":")
    if kind == "count":
        nums = re.findall(r"-?\d+", str(response))
        return 1.0 if nums and nums[-1] == body else 0.0

    pred = f" {_norm(str(response))} "
    expected = [_norm(x) for x in body.split("|") if x.strip()]
    if not expected:
        return 0.0
    hits = sum(1 for x in expected if f" {x} " in pred)
    if kind == "type":
        return 1.0 if hits >= 1 else 0.0
    return hits / len(expected)
