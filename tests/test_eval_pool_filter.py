"""The CoAPT pool view of the referee: corpus_probes tasks expose doc_id (so records can
be pooled per document) and eval_probes' doc-id loader accepts both pool formats. Probe
content itself is untouched — the pool is a view, never a regeneration."""
from __future__ import annotations

import json

import pytest

from gym import tasks as gym_tasks
from tools import eval_probes


def test_corpus_probe_tasks_carry_doc_id_tags():
    rows = gym_tasks.load("corpus_probes", split="dev", n=25)
    assert rows
    for row in rows:
        assert row.tags.get("doc_id"), f"probe {row.id} lost its doc_id tag"
        assert row.tags.get("track") in {"transfer", "absorption"}


def test_load_doc_ids_accepts_plain_and_jsonl(tmp_path):
    plain = tmp_path / "pool.txt"
    plain.write_text("doc-a\n\ndoc-b\n")
    assert eval_probes.load_doc_ids(plain) == {"doc-a", "doc-b"}

    jsonl = tmp_path / "pool.jsonl"
    jsonl.write_text("".join(json.dumps({"doc_id": d, "topic": "t"}) + "\n"
                             for d in ("doc-a", "doc-c")))
    assert eval_probes.load_doc_ids(jsonl) == {"doc-a", "doc-c"}

    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    with pytest.raises(SystemExit):
        eval_probes.load_doc_ids(empty)
