from __future__ import annotations

import json

from experiments.cpt.build_data import stream_rows


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        limit = min(len(text), kwargs["max_length"])
        return {"offset_mapping": [(i, i + 1) for i in range(limit)]}


def _row(doc_id, start, text):
    return {
        "text": text,
        "meta": {
            "doc_id": doc_id,
            "topic": "t",
            "source": "s",
            "license": "open",
            "round": 0,
            "document_token_start": start,
            "content_tokens": len(text),
        },
    }


def test_anchored_stream_copies_prefix_and_resumes_without_duplicates(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("abcd")
    second.write_text("efgh")
    prefix = [_row("a", 0, "ab"), _row("a", 2, "cd"), _row("b", 0, "ef")]
    prefix_path = tmp_path / "prefix.jsonl"
    prefix_path.write_text("".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in prefix))
    docs = [
        {"id": "a", "topic": "t", "source": "s", "_text_path": first},
        {"id": "b", "topic": "t", "source": "s", "_text_path": second},
    ]
    recipe = {
        "target_content_tokens": 8,
        "max_document_content_tokens": 4,
        "document_slice_tokens": 4,
        "max_content_tokens": 2,
        "min_clean_chars": 1,
    }
    stats = {}
    rows = list(stream_rows(
        docs, CharacterTokenizer(), recipe, stats,
        prefix_path=prefix_path, prefix_content_tokens=6,
    ))
    keys = [
        (row["meta"]["doc_id"], row["meta"]["document_token_start"]) for row in rows
    ]
    assert rows[:3] == prefix
    assert rows[3]["text"] == "gh"
    assert keys == [("a", 0), ("a", 2), ("b", 0), ("b", 2)]
    assert stats["content_tokens"] == 8
