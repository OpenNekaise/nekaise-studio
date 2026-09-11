"""Composite stream plans (campaign extension 2026-09-11): a sealed, authenticated
snapshot; segment A = the base plan's documents in their original order (pruned or
changed ones kept as ranked tombstones), segment B = new documents that share no
identity with the base plan; an anchor dataset whose re-emitted bytes must hash to its
recorded content hash; segment-aware rounds; registered exhaustion. CPU-only,
char-level tokenizer, tiny synthetic corpus."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from experiments.cpt import build_data
import datakit  # noqa: E402  (lib on sys.path via build_data)
import corpusprep  # noqa: E402


class _OffsetTokenizer:
    def __call__(self, text, **kwargs):
        limit = kwargs["max_length"]
        return {"offset_mapping": [(i, i + 1) for i in range(min(limit, len(text)))]}


RECIPE = {
    "max_document_content_tokens": 6, "document_slice_tokens": 4,
    "max_content_tokens": 3, "min_source_chars": 1, "min_clean_chars": 1,
    "selection_seed": 1,
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _manifest_row(doc_id: str, text: str) -> dict:
    return {"id": doc_id, "status": "ok", "text_chars": len(text), "sha256": _sha(text),
            "corpus_sha256": "clean-" + _sha(text), "topic": "t", "source": "s",
            "license": "x", "text_path": f"text/{doc_id}.md"}


def _write_manifest(root: Path, rows: list[dict]) -> None:
    (root / "manifest" / "m.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))


def make_corpus(tmp_path: Path, docs: dict[str, str]) -> Path:
    root = tmp_path / "corpus"
    for sub in ("manifest", "text", "registry"):
        (root / sub).mkdir(parents=True)
    for doc_id, text in docs.items():
        (root / "text" / f"{doc_id}.md").write_text(text)
    _write_manifest(root, [_manifest_row(d, t) for d, t in docs.items()])
    return root


@pytest.fixture
def env(tmp_path, monkeypatch):
    exp = tmp_path / "exp"
    monkeypatch.setattr(build_data, "EXP_DIR", exp)
    monkeypatch.setattr(build_data, "probe_holdout", lambda: (set(), "probe-fp"))
    monkeypatch.setattr(build_data, "corpus_commit", lambda root: "deadbeef")
    monkeypatch.setattr(build_data, "corpus_dirty_paths", lambda root: 0)
    monkeypatch.setattr(build_data, "_tokenizer_path", lambda: "/models/tok-rev1")
    monkeypatch.setattr(build_data, "WORKSPACE",
                        types.SimpleNamespace(resolve_input=lambda p: Path(p)))
    return exp


def _plan_order(exp: Path, name: str) -> list[str]:
    rows = (exp / "data" / "plans" / name / "documents.jsonl").read_text().splitlines()
    return [json.loads(r)["id"] for r in rows if r.strip()]


def _build_base_and_anchor(env, tmp_path):
    """Base plan v2 over four documents and a 7-token anchor dataset built from it."""
    texts = {"a1": "abcdefgh", "a2": "ijklmnop", "a3": "qrstuvwx", "a4": "yzabcdef"}
    root = make_corpus(tmp_path, texts)
    recipe = {**RECIPE, "stream_plan": "v2", "target_content_tokens": 7}
    docs, plan = build_data.load_or_create_plan(root, recipe)
    stats = {}
    rows = list(build_data.stream_rows(docs, _OffsetTokenizer(), recipe, stats))
    spec = build_data.build_spec(recipe, plan, "tok-rev1", None)
    obj = datakit.write(env, spec, iter(rows), stats=stats)
    prov = datakit.provenance(obj)
    anchor = {"dataset_id": prov["dataset_id"], "content_sha256": prov["content_sha256"],
              "content_tokens": stats["content_tokens"]}
    assert stats["content_tokens"] == 7
    return root, texts, rows, anchor


def _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, *, target,
                    allow_exhaustion=False):
    """The corpus moved on since the anchor: one base document (never the anchor's
    terminal one) was pruned, another's cleaned content changed, two new documents
    arrived, and one "new" id is a byte-duplicate of a base document."""
    order = _plan_order(env, "v2")
    terminal = anchor_rows[-1]["meta"]["doc_id"]
    candidates = [d for d in order if d != terminal]
    pruned, changed = candidates[-1], candidates[-2]
    hist = tmp_path / "hist"
    hist.mkdir()
    (hist / "m.jsonl").write_bytes((root / "manifest" / "m.jsonl").read_bytes())
    (root / "text" / f"{pruned}.md").unlink()
    (root / "registry" / "pruned-1.jsonl").write_text(json.dumps(
        {"id": pruned, "reason": "off-topic-title", "pruned_at": "2026-09-01T00:00:00Z",
         "sha256": _sha(texts[pruned])}) + "\n")
    new_texts = {"b1": "ABCDEFGH", "b2": "IJKLMNÖP\"\\", "dup": texts[order[0]]}
    for doc_id, text in new_texts.items():
        (root / "text" / f"{doc_id}.md").write_text(text)
    rows = [_manifest_row(d, t) for d, t in {**texts, **new_texts}.items()]
    for row in rows:
        if row["id"] == changed:
            row["corpus_sha256"] = "clean-changed"
    _write_manifest(root, rows)
    recipe = {**RECIPE, "stream_plan": "v3", "target_content_tokens": target,
              "composite": {"base_plan": "v2", "historical_manifest": str(hist)},
              "extension_anchor": anchor}
    if allow_exhaustion:
        recipe["allow_exhaustion"] = True
    return recipe, {"order": order, "terminal": terminal, "pruned": pruned,
                    "changed": changed,
                    "surviving": [d for d in order if d not in {pruned, changed}]}


def test_composite_plan_is_sealed_segmented_and_records_missing(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20)
    docs, plan = build_data.load_or_create_plan(root, recipe)
    assert plan["segment_documents"] == {"v2": 2, "new": 2}       # dup excluded by hash
    assert plan["missing_reasons"] == {"off-topic-title": 1, "content-changed": 1}
    assert plan["composite_base_excluded"] == 5                   # 4 base ids + the dup
    assert [d["id"] for d in docs][:2] == w["surviving"]           # base order preserved
    assert {d["segment"] for d in docs[2:]} == {"new"}
    listed = [json.loads(l) for l in
              (env / "data" / "plans" / "v3" / "documents.jsonl").read_text().splitlines()]
    assert [t["id"] for t in listed[:4]] == w["order"]             # ranks never shift
    assert {t["id"]: t.get("missing") for t in listed[:4]} == {
        **{d: None for d in w["surviving"]}, w["pruned"]: "off-topic-title",
        w["changed"]: "content-changed"}
    assert all(str(d["_text_path"]).startswith(str(env / "data" / "plans" / "v3" / "snapshot"))
               for d in docs)
    snap = json.loads((env / "data" / "plans" / "v3" / "snapshot" / "provenance.json").read_text())
    assert {m["id"] for m in snap["missing_documents"]} == {w["pruned"], w["changed"]}
    assert snap["files"] == 4
    # Re-loading verifies the snapshot; tampering with one sealed file is refused.
    build_data.load_or_create_plan(root, recipe)
    sealed = env / "data" / "plans" / "v3" / "snapshot" / "text" / f"{w['surviving'][0]}.md"
    sealed.write_text(sealed.read_text() + "!")
    with pytest.raises(SystemExit, match="failed authentication"):
        build_data.load_or_create_plan(root, recipe)


def test_anchor_bytes_then_segment_a_continuation_then_segment_b(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows,
                                target=100, allow_exhaustion=True)
    docs, _plan = build_data.load_or_create_plan(root, recipe)
    prefix = datakit.data_file(env / "data" / "objects" / anchor["dataset_id"])
    stats = {}
    rows = list(build_data.stream_rows(
        docs, _OffsetTokenizer(), recipe, stats, prefix_path=prefix,
        prefix_content_tokens=anchor["content_tokens"],
        prefix_content_sha256=anchor["content_sha256"], allow_exhaustion=True))
    n = len(anchor_rows)
    assert [build_data.serialize_row(r) for r in rows[:n]] == \
        [build_data.serialize_row(r) for r in anchor_rows]
    keys = [(r["meta"]["doc_id"], r["meta"]["round"], r["meta"]["document_token_start"])
            for r in rows]
    assert len(keys) == len(set(keys))                            # no duplicated span
    # The anchor ended inside its terminal document at round 0: continuation finishes
    # that document, walks the rest of round 0 over surviving base documents only,
    # completes round 1 for them, and only then starts the new documents at round 0.
    order, surviving, terminal = w["order"], w["surviving"], w["terminal"]
    last = anchor_rows[-1]["meta"]
    resume_at = last["document_token_start"] + last["content_tokens"]
    expected = [(terminal, 0, resume_at)] if resume_at < 4 else []
    for doc in order[order.index(terminal) + 1:]:
        if doc in surviving:
            expected += [(doc, 0, 0), (doc, 0, 3)]
    expected += [(doc, 1, 4) for doc in surviving]
    assert keys[n:n + len(expected)] == expected
    tail = keys[n + len(expected):]
    assert tail and tail[0][1] == 0 and tail[0][0] in {"b1", "b2"}
    assert not any(k[0] in {w["pruned"], w["changed"]} for k in keys[n:])
    assert stats["exhausted"] and stats["shortfall_tokens"] == 100 - stats["content_tokens"]
    assert set(stats["segment_content_tokens"]) == {"anchor", "v2", "new"}
    assert stats["segment_content_tokens"]["anchor"] == 7
    unicode_rows = [r for r in rows if r["meta"]["doc_id"] == "b2"]
    cleaned = corpusprep.clean_body("IJKLMNÖP\"\\")
    assert "".join(r["text"] for r in unicode_rows) == cleaned[:6]   # 6-token doc cap


def test_composite_targets_are_nested_prefixes_and_stop_below_target(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, _w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=14)
    docs, _plan = build_data.load_or_create_plan(root, recipe)
    prefix = datakit.data_file(env / "data" / "objects" / anchor["dataset_id"])

    def build(target):
        stats = {}
        rows = list(build_data.stream_rows(
            docs, _OffsetTokenizer(), {**recipe, "target_content_tokens": target}, stats,
            prefix_path=prefix, prefix_content_tokens=7,
            prefix_content_sha256=anchor["content_sha256"]))
        return rows, stats
    small, small_stats = build(14)
    large, large_stats = build(20)
    assert large[:len(small)] == small
    assert small_stats["content_tokens"] <= 14 and not small_stats["exhausted"]
    assert large_stats["content_tokens"] <= 20 and not large_stats["exhausted"]
    assert large_stats["content_tokens"] > small_stats["content_tokens"]
    with pytest.raises(RuntimeError, match="below target"):
        build(100)                                                 # no exhaustion flag


def test_anchor_hash_and_token_drift_are_refused(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, _w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20)
    docs, _plan = build_data.load_or_create_plan(root, recipe)
    prefix = datakit.data_file(env / "data" / "objects" / anchor["dataset_id"])
    with pytest.raises(RuntimeError, match="anchor bytes drift"):
        list(build_data.stream_rows(docs, _OffsetTokenizer(), recipe, {},
                                    prefix_path=prefix, prefix_content_tokens=7,
                                    prefix_content_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="token drift"):
        list(build_data.stream_rows(docs, _OffsetTokenizer(), recipe, {},
                                    prefix_path=prefix, prefix_content_tokens=8,
                                    prefix_content_sha256=anchor["content_sha256"]))


def test_spec_binds_composite_identity_and_anchor(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, _w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20,
                                 allow_exhaustion=True)
    _docs, plan = build_data.load_or_create_plan(root, recipe)
    _prefix, resolved = build_data.anchor_for(recipe)
    spec = build_data.build_spec(recipe, plan, "tok-rev1", resolved)
    assert spec["composite"]["base_plan"] == "v2" and spec["allow_exhaustion"] is True
    assert spec["extension_anchor_dataset_id"] == anchor["dataset_id"]
    assert spec["snapshot_missing_documents"] == 2
    bad = {**recipe, "extension_anchor": {**anchor, "content_sha256": "0" * 64}}
    with pytest.raises(SystemExit, match="content drift"):
        build_data.anchor_for(bad)


def test_sealed_identities_are_enforced(env, tmp_path, monkeypatch):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20)
    _docs, plan = build_data.load_or_create_plan(root, recipe)
    plan_dir = env / "data" / "plans" / "v3"
    # tokenizer revision and cleaner hash are pinned by the plan
    build_data.assert_plan_identities(recipe, plan, "/models/tok-rev1")
    with pytest.raises(SystemExit, match="tokenizer revision"):
        build_data.assert_plan_identities(recipe, plan, "/models/tok-rev2")
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setattr(build_data, "sha256_file",
                       lambda path: "changed" if path.name == "corpusprep.py" else "x")
        with pytest.raises(SystemExit, match="corpusprep.py changed"):
            build_data.assert_plan_identities(recipe, plan, "/models/tok-rev1")
    # the plan's recorded snapshot hash must equal the snapshot's own record
    prov_path = plan_dir / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["snapshot_content_manifest_sha256"] = "0" * 64
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(SystemExit, match="records snapshot manifest"):
        build_data.plan_provenance(recipe)
    # historical evidence was classified per base document and the unverified list saved
    snap = json.loads((plan_dir / "snapshot" / "provenance.json").read_text())
    evidence = snap["historical_text_evidence"]
    assert sum(evidence.values()) == 2 and evidence["verified"] == 0   # fixture hashes are synthetic
    unverified = json.loads((plan_dir / "snapshot" / "historical_unverified.json").read_text())
    assert set(unverified["ids"]) == set(w["surviving"])


def test_terminal_anchor_document_must_survive(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20)
    # prune the anchor's terminal document too: sealing must refuse
    (root / "text" / f"{w['terminal']}.md").unlink()
    with pytest.raises(SystemExit, match="terminal document"):
        build_data.load_or_create_plan(root, recipe)


def test_rewritten_anchor_file_is_refused(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, _w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20)
    docs, _plan = build_data.load_or_create_plan(root, recipe)
    prefix = datakit.data_file(env / "data" / "objects" / anchor["dataset_id"])
    rewritten = tmp_path / "anchor_rewritten.jsonl"
    rewritten.write_text("".join(json.dumps(r, indent=None, separators=(", ", ": ")) + "\n"
                                 for r in anchor_rows))
    with pytest.raises(RuntimeError, match="canonical serialization"):
        list(build_data.stream_rows(docs, _OffsetTokenizer(), recipe, {},
                                    prefix_path=rewritten, prefix_content_tokens=7,
                                    prefix_content_sha256=anchor["content_sha256"]))


def test_audit_anchor_rederives_every_anchor_row(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20)
    docs, plan = build_data.load_or_create_plan(root, recipe)
    build_data.audit_anchor(recipe, docs, plan, _OffsetTokenizer())
    record = json.loads((env / "data" / "plans" / "v3" / "anchor_audit.json").read_text())
    tombstoned = sum(1 for r in anchor_rows if r["meta"]["doc_id"] in {w["pruned"], w["changed"]})
    assert record["ok"] and record["rows_total"] == len(anchor_rows)
    assert record["rows_tombstoned_skipped"] == tombstoned
    assert record["rows_verified"] == len(anchor_rows) - tombstoned
    # a changed sealed byte is detected by the audit
    sealed = env / "data" / "plans" / "v3" / "snapshot" / "text" / f"{w['terminal']}.md"
    sealed.write_text("zzzzzzzz")
    with pytest.raises(SystemExit, match="anchor audit FAILED"):
        build_data.audit_anchor(recipe, docs, plan, _OffsetTokenizer())


def test_anchor_audit_rejects_truncated_or_blank_anchor_and_binds_to_spec(env, tmp_path):
    root, texts, anchor_rows, anchor = _build_base_and_anchor(env, tmp_path)
    recipe, w = _make_composite(env, tmp_path, root, texts, anchor, anchor_rows, target=20)
    docs, plan = build_data.load_or_create_plan(root, recipe)
    build_data.audit_anchor(recipe, docs, plan, _OffsetTokenizer())
    plan_dir = env / "data" / "plans" / "v3"
    record = json.loads((plan_dir / "anchor_audit.json").read_text())
    assert record["ok"] and record["anchor_bytes_verified"]
    assert record["rows_total"] == record["rows_expected"] == len(anchor_rows)
    _prefix, resolved = build_data.anchor_for(recipe)
    spec = build_data.build_spec(recipe, plan, "tok-rev1", resolved)
    assert build_data.anchor_audit_binds(spec, plan_dir)
    other = {**spec, "extension_anchor_content_sha256": "0" * 64}
    assert not build_data.anchor_audit_binds(other, plan_dir)
    # truncate the anchor file: totals and bytes no longer reconcile
    prefix = datakit.data_file(env / "data" / "objects" / anchor["dataset_id"])
    original = prefix.read_bytes()
    try:
        prefix.write_bytes(original[: original.rfind(b"\n", 0, -1) + 1])
        with pytest.raises(SystemExit, match="anchor audit FAILED"):
            build_data.audit_anchor(recipe, docs, plan, _OffsetTokenizer())
        assert not build_data.anchor_audit_binds(spec, plan_dir)
        prefix.write_bytes(original.replace(b"\n", b"\n\n", 1))     # a blank line
        with pytest.raises(RuntimeError, match="blank line"):
            list(build_data.stream_rows(docs, _OffsetTokenizer(), recipe, {},
                                        prefix_path=prefix, prefix_content_tokens=7,
                                        prefix_content_sha256=anchor["content_sha256"]))
    finally:
        prefix.write_bytes(original)
