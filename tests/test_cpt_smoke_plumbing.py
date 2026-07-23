from __future__ import annotations

import json

from experiments.cpt.build_data import stream_rows
from tools import eval_probes


class _OffsetTokenizer:
    def __call__(self, text, **kwargs):
        limit = kwargs["max_length"]
        return {"offset_mapping": [(i, i + 1) for i in range(min(limit, len(text)))]}


def test_cpt_scale_datasets_are_nested_complete_row_prefixes(tmp_path):
    docs = []
    for doc_id, text in (("a", "abcdefgh"), ("b", "ijklmnop")):
        path = tmp_path / f"{doc_id}.md"
        path.write_text(text)
        docs.append({"id": doc_id, "topic": "t", "source": "s",
                     "license": "x", "_text_path": path})
    recipe = {
        "max_document_content_tokens": 6, "document_slice_tokens": 4,
        "max_content_tokens": 3, "min_clean_chars": 1,
        "target_content_tokens": 7,
    }
    small_stats = {}
    small = list(stream_rows(docs, _OffsetTokenizer(), recipe, small_stats))
    recipe = {**recipe, "target_content_tokens": 10}
    large_stats = {}
    large = list(stream_rows(docs, _OffsetTokenizer(), recipe, large_stats))
    assert large[:len(small)] == small
    assert small_stats["content_tokens"] == 7
    assert large_stats["content_tokens"] == 10
    assert all(row["meta"]["content_tokens"] <= 3 for row in large)


def test_eval_infers_producer_run_from_checkpoint_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "meta.json").write_text(json.dumps({"run_id": "run-1"}))
    assert eval_probes.run_id_from_target(f"ckpt:{checkpoint}") == "run-1"
