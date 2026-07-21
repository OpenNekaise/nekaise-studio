"""numeric_cloze — base-model completion probes: first number in the continuation.

meta: {"value": <number>}   the gold value the continuation must lead with.

A response is correct iff the FIRST number inside the opening window (80 chars) equals
the gold value exactly. Ported from packs/corpus_probes/scorer.py (the pure-CPT loop
metric) with ONE fix the intake smoke test forced: the old alternation matched "120"
inside "1200" (comma-group branch tried first even with no comma), making 13 probes
unwinnable for every model. The comma branch now requires a comma. This re-baselines
probe scores vs pre-refactor numbers (see docs/REFACTOR-NOTES.md D9); the rule is pinned
by tests from here on.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_WINDOW = 80


def first_number(text: str) -> str | None:
    m = _NUM.search(str(text))
    return m.group(0).replace(",", "") if m else None


def verify(prompt: str, response: str, meta: dict) -> float:
    first = first_number(str(response)[:_WINDOW])
    if first is None:
        return 0.0
    try:
        return 1.0 if abs(float(first) - float(meta["value"])) < 1e-9 else 0.0
    except (ValueError, KeyError, TypeError):
        return 0.0
