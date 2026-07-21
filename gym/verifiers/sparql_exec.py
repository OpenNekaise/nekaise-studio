"""sparql_exec — executable-SPARQL verifier: result-set F1 against the gold query (R7).

meta: {"ttl": <path to .ttl file or literal turtle text>,
       "gold_query": <SPARQL whose result set is the gold>}     (or "gold_rows": [[...]])

The response is expected to contain a SPARQL query (```sparql fence, or the first
SELECT/ASK block). Both queries run against the same graph; score is row-set F1
(order-insensitive, string-normalized cells). Hard-verifiable: no string matching against
prose, no judge. Malformed/failing predicted queries score 0.0.
"""
from __future__ import annotations

import re
from pathlib import Path

_FENCE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.S | re.I)
_SELECT = re.compile(r"((?:PREFIX\s+[^\n]+\n)*\s*(?:SELECT|ASK)\b.*)", re.S | re.I)
_ROW_CAP = 10_000


def extract_query(text: str) -> str | None:
    for block in _FENCE.findall(str(text)):
        if re.search(r"\b(SELECT|ASK)\b", block, re.I):
            return block.strip()
    m = _SELECT.search(str(text))
    return m.group(1).strip() if m else None


def _graph(meta: dict):
    import rdflib
    g = rdflib.Graph()
    ttl = str(meta["ttl"])
    if "\n" in ttl or not Path(ttl).exists():
        g.parse(data=ttl, format="turtle")
    else:
        g.parse(ttl, format="turtle")
    return g


def _rows(g, query: str) -> set[tuple]:
    res = g.query(query)
    if getattr(res, "type", "") == "ASK":
        return {(str(bool(res.askAnswer)).lower(),)}
    rows = set()
    for i, row in enumerate(res):
        if i >= _ROW_CAP:
            break
        rows.add(tuple(str(c).strip().lower() if c is not None else "" for c in row))
    return rows


def verify(prompt: str, response: str, meta: dict) -> float:
    query = extract_query(response)
    if not query:
        return 0.0
    try:
        g = _graph(meta)
        gold = ({tuple(str(c).strip().lower() for c in r) for r in meta["gold_rows"]}
                if "gold_rows" in meta else _rows(g, meta["gold_query"]))
        pred = _rows(g, query)
    except Exception:
        return 0.0
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision, recall = tp / len(pred), tp / len(gold)
    return 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
