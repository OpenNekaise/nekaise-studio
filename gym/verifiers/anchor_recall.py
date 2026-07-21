"""anchor_recall — fraction of required anchor facts present in the response.

meta: {"anchors": [<str>, ...]}   the must-match facts (values, vendor tags, file paths,
                                  component names, time windows, checklist items)

Score = matched anchors / total anchors, matching on normalized substrings (strict on the
fact, lenient on phrasing). This is the SOFT tier: it is gameable by anchor-stuffing, so
training use is paired with gym.tools.monitors (anchor-density + listiness watch, R7) —
defense is measurement, not algorithm patches. Prefer hard verifiers where possible.
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9./:_-]+", " ", str(s).lower()).strip()


def verify(prompt: str, response: str, meta: dict) -> float:
    anchors = [a for a in (meta.get("anchors") or []) if str(a).strip()]
    if not anchors:
        return 0.0
    resp = f" {_norm(response)} "
    hits = sum(1 for a in anchors if _norm(a) and _norm(a) in resp)
    return hits / len(anchors)
