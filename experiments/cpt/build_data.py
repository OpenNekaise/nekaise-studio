#!/usr/bin/env python3
"""Build a deterministic, nested full-parameter CPT dataset from nekaise-corpus.

The scale ladder changes only ``target_content_tokens``.  Every larger artifact starts
with exactly the complete rows of every smaller artifact: documents have one fixed
weighted order and are exposed in fixed token slices (all documents' first slice,
then all documents' second slice, and so on).  Licenses are recorded as provenance,
never used as a filter.

    python experiments/cpt/build_data.py [--config <cpt.yaml>] [--plan-only]

Composite plans (campaign extension, 2026-09-11).  A ``composite:`` recipe block turns a
new stream plan into two ordered segments: segment A is a frozen base plan's documents in
their original order (documents the corpus has since pruned or changed are recorded as
missing and never re-read), segment B is every currently eligible document that shares
no identity with the base plan (id, raw hash, cleaned hash), in the same weighted order
computed over segment B alone.  A composite plan is sealed at creation: every document's
text is copied into ``plans/<name>/snapshot/`` with a per-file hash manifest that is
re-verified before every build, so a growing sibling corpus can never change the bytes
the plan streams.  With an ``extension_anchor`` the first rows of the new dataset are
read from an existing immutable dataset; the emitted bytes are hashed and must equal
that dataset's recorded content hash, so the new dataset is a byte-level extension of
the old one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CORE_EXP_DIR = Path(__file__).resolve().parent
REPO = CORE_EXP_DIR.parents[1]
sys.path.insert(0, str(REPO / "lib"))

import corpusprep  # noqa: E402
import datakit  # noqa: E402
from workspace import Workspace  # noqa: E402

WORKSPACE = Workspace.resolve(REPO).apply_environment()
EXP_DIR = WORKSPACE.experiment_dir("cpt")

CONFIG = REPO / "configs" / "cpt.yaml"
TOKENIZER = "openbmb/MiniCPM5-1B-Base"
PROBE_PROVENANCE = REPO / "gym" / "tasks" / "corpus_probes" / "provenance.json"
ORDERING_VERSION = "weighted-exponential-topic-source-v1"
STREAM_VERSION = "document-rounds-v1"
EXTENSION_VERSION = "anchored-prefix-snapshot-v1"
COMPOSITE_VERSION = "composite-segments-v1"
PLAN_ROW_KEYS = ("id", "topic", "source", "license", "sha256", "text_chars")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return digest.hexdigest()


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def serialize_row(row: dict) -> bytes:
    """Exactly the bytes datakit.write emits for one row (kept in lock step)."""
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def corpus_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def corpus_dirty_paths(root: Path) -> int:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        capture_output=True, text=True, check=True,
    ).stdout
    return len([line for line in status.splitlines() if line.strip()])


def manifest_files(root: Path) -> list[Path]:
    return sorted((root / "manifest").glob("*.jsonl"))


def registry_files(root: Path) -> list[Path]:
    return sorted((root / "registry").glob("pruned-*.jsonl"))


def manifest_rows(root: Path, hashes: dict[str, str] | None = None):
    """Rows of every manifest file, each file read ONCE as a byte buffer that is also
    hashed into ``hashes`` when given, so the recorded manifest hash is the hash of
    the bytes the rows were parsed from even while corpus writers are active."""
    for path in manifest_files(root):
        payload = path.read_bytes()
        if hashes is not None:
            hashes[path.name] = sha256_bytes(payload)
        for line in payload.decode(errors="replace").splitlines():
            if line.strip():
                yield json.loads(line)


def pruned_rows(root: Path) -> dict[str, dict]:
    """id -> the corpus pipeline's prune record (reason, time, hashes)."""
    out: dict[str, dict] = {}
    for path in registry_files(root):
        with path.open(errors="replace") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("id"):
                        out[row["id"]] = row
    return out


def probe_holdout() -> tuple[set[str], str]:
    provenance = json.loads(PROBE_PROVENANCE.read_text())
    splits = provenance["doc_splits"]
    # Absorption documents intentionally remain train eligible. Transfer documents
    # and exact-content duplicates of them are never visible to CPT.
    ids = set(splits["dev"]) | set(splits["frozen"])
    return ids, provenance["fingerprint"]


def weighted_order(docs: list[dict], seed: int) -> list[dict]:
    """Topic and source exposure grows with sqrt(raw mass), while documents within a
    source share that source's mass. This softens dominant crawls without flattening
    the real corpus distribution. Deterministic in (seed, id)."""
    topic_mass: dict[str, int] = defaultdict(int)
    source_mass: dict[tuple[str, str], int] = defaultdict(int)
    source_count: dict[tuple[str, str], int] = defaultdict(int)
    for row in docs:
        topic = row.get("topic") or "unknown"
        source = row.get("source") or "unknown"
        chars = int(row.get("text_chars") or 0)
        topic_mass[topic] += chars
        source_mass[(topic, source)] += chars
        source_count[(topic, source)] += 1
    topic_normalizer = sum(math.sqrt(mass) for mass in topic_mass.values())
    source_normalizer: dict[str, float] = defaultdict(float)
    for (topic, _source), mass in source_mass.items():
        source_normalizer[topic] += math.sqrt(mass)

    def priority(row: dict) -> tuple[float, str]:
        topic = row.get("topic") or "unknown"
        source = row.get("source") or "unknown"
        bucket = (topic, source)
        weight = (math.sqrt(topic_mass[topic]) / topic_normalizer
                  * math.sqrt(source_mass[bucket]) / source_normalizer[topic]
                  / source_count[bucket])
        digest = hashlib.sha256(
            f"{ORDERING_VERSION}:{seed}:{row['id']}".encode()).digest()
        uniform = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)
        return -math.log(uniform) / weight, row["id"]

    return sorted(docs, key=priority), {"topics": len(topic_mass), "sources": len(source_mass)}


def select_documents(root: Path, recipe: dict, *, rows: list[dict] | None = None,
                     exclude: dict | None = None) -> tuple[list[dict], dict]:
    """Eligible corpus documents in the frozen weighted order.

    ``exclude`` (composite plans) removes every document whose id, raw ``sha256`` or
    cleaned ``corpus_sha256`` belongs to the base plan, so segment B never repeats a
    base-plan document under a new id."""
    holdout_ids, probe_fingerprint = probe_holdout()
    rows = list(manifest_rows(root)) if rows is None else rows
    exclude = exclude or {}
    excluded_ids_set = set(exclude.get("ids") or ())
    excluded_hashes = set(exclude.get("sha256") or ())
    excluded_clean = set(exclude.get("corpus_sha256") or ())
    manifest_snapshot = hashlib.sha256()
    for row in rows:
        manifest_snapshot.update(json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode())
        manifest_snapshot.update(b"\n")
    holdout_hashes = {
        row.get("sha256") for row in rows
        if row.get("id") in holdout_ids and row.get("sha256")
    }
    seen_hashes: set[str] = set()
    docs: list[dict] = []
    excluded_ids = excluded_duplicates = excluded_composite = 0
    min_chars = int(recipe["min_source_chars"])

    for row in rows:
        doc_id = row.get("id") or ""
        source_hash = row.get("sha256") or ""
        if doc_id in holdout_ids:
            excluded_ids += 1
            continue
        if source_hash and source_hash in holdout_hashes:
            excluded_duplicates += 1
            continue
        if (doc_id in excluded_ids_set or (source_hash and source_hash in excluded_hashes)
                or (row.get("corpus_sha256") and row["corpus_sha256"] in excluded_clean)):
            excluded_composite += 1
            continue
        if row.get("status") != "ok" or int(row.get("text_chars") or 0) < min_chars:
            continue
        if source_hash and source_hash in seen_hashes:
            continue
        rel = Path(row.get("text_path") or f"text/{doc_id}.md")
        if not doc_id or not rel.parts or rel.parts[0] != "text":
            continue
        text_path = root / rel
        if not text_path.is_file():
            continue
        if source_hash:
            seen_hashes.add(source_hash)
        docs.append({**row, "_text_path": text_path})

    if not docs:
        raise SystemExit("no eligible local corpus documents")

    docs, shape = weighted_order(docs, int(recipe["selection_seed"]))
    ordered_fingerprint = hashlib.sha256("\n".join(
        f"{row['id']}:{row.get('sha256') or ''}" for row in docs).encode()).hexdigest()
    return docs, {
        "probe_fingerprint": probe_fingerprint,
        "ordered_documents_sha256": ordered_fingerprint,
        "eligible_documents": len(docs),
        **shape,
        "holdout_ids_excluded": excluded_ids,
        "holdout_duplicates_excluded": excluded_duplicates,
        "composite_base_excluded": excluded_composite,
        "manifest_snapshot_sha256": manifest_snapshot.hexdigest(),
    }


def plan_expectations(recipe: dict) -> dict:
    expected = {
        "schema_version": 1,
        "name": str(recipe["stream_plan"]),
        "ordering": ORDERING_VERSION,
        "selection_seed": int(recipe["selection_seed"]),
        "min_source_chars": int(recipe["min_source_chars"]),
        "probe_fingerprint": probe_holdout()[1],
    }
    composite = recipe.get("composite")
    if composite:
        expected["composite"] = COMPOSITE_VERSION
        expected["composite_base_plan"] = str(composite["base_plan"])
    return expected


def plan_payload(plan_rows: list[dict]) -> bytes:
    return "".join(json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n" for row in plan_rows).encode()


def _publish_plan(plan_dir: Path, build) -> None:
    """Build a plan into a uniquely named temp dir, then publish it atomically."""
    plan_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_dir.with_name(
        f".{plan_dir.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        build(temporary)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    os.replace(temporary, plan_dir)


def create_simple_plan(root: Path, recipe: dict, plan_dir: Path) -> None:
    selected, selection = select_documents(root, recipe)
    plan_rows = [{key: row.get(key) for key in PLAN_ROW_KEYS}
                 | {"text_path": str(row["_text_path"].relative_to(root))}
                 for row in selected]
    payload = plan_payload(plan_rows)
    provenance = {
        **plan_expectations(recipe), **selection,
        "corpus_commit_at_creation": corpus_commit(root),
        "documents_sha256": sha256_bytes(payload),
    }

    def build(temporary: Path) -> None:
        (temporary / "documents.jsonl").write_bytes(payload)
        (temporary / "provenance.json").write_text(json.dumps(
            provenance, ensure_ascii=False, indent=2, sort_keys=True))

    _publish_plan(plan_dir, build)


def _historical_clean_hashes(recipe_composite: dict) -> dict[str, str] | None:
    """id -> corpus_sha256 from a frozen earlier manifest copy, when the recipe names
    one (workspace-relative directory of manifest *.jsonl files)."""
    directory = recipe_composite.get("historical_manifest")
    if not directory:
        return None
    path = WORKSPACE.resolve_input(directory)
    out: dict[str, str] = {}
    for file in sorted(Path(path).glob("*.jsonl")):
        with file.open(errors="replace") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("id") and row.get("corpus_sha256"):
                        out[row["id"]] = row["corpus_sha256"]
    return out


def _historical_manifest_hashes(recipe_composite: dict) -> dict[str, str]:
    directory = recipe_composite.get("historical_manifest")
    if not directory:
        return {}
    path = Path(WORKSPACE.resolve_input(directory))
    return {file.name: sha256_file(file) for file in sorted(path.glob("*.jsonl"))}


def _last_row_doc_id(path: Path) -> str:
    """doc_id of the last row of a dataset file without reading the whole file."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - (4 << 20)))
        tail = handle.read()
    last = [line for line in tail.split(b"\n") if line.strip()][-1]
    return json.loads(last)["meta"]["doc_id"]


def audit_anchor(recipe: dict, docs: list[dict], plan: dict, tokenizer) -> None:
    """Anchored-span reconstruction: re-derive every anchor row from the SEALED
    segment-A text and require byte equality with the anchor file, while hashing the
    anchor file's raw bytes and reconciling its row and token totals with its
    provenance (an empty or truncated anchor can never pass). Rows of tombstoned
    documents are unverifiable and counted separately. The record is published
    atomically as plans/<name>/anchor_audit.json with every identity a later
    consumer must match (anchor id/hash, plan bytes, snapshot bytes, tokenizer,
    cleaner, stream settings); the runner refuses to train on an anchored dataset
    whose spec does not match a passed audit. This establishes continuity of the
    spans the anchor consumed; it says nothing about document tails the anchor never
    reached (those rest on the sealed snapshot and the historical hash evidence)."""
    prefix_path, anchor = anchor_for(recipe)
    if anchor is None:
        raise SystemExit("recipe has no extension_anchor to audit")
    anchor_dir = datakit.data_root(EXP_DIR) / "objects" / anchor["dataset_id"]
    anchor_provenance = datakit.provenance(anchor_dir)
    expected_rows = int(anchor_provenance["n"])
    tombstones = {row["id"] for row in plan_provenance(recipe)[0] if row.get("missing")}
    first_segment = _segments(docs)[0][1]
    rebuilt = stream_rows(first_segment, tokenizer, {
        **recipe, "target_content_tokens": 1 << 62}, {}, allow_exhaustion=True)
    total = verified = skipped = tokens = 0
    mismatch = None
    digest = hashlib.sha256()
    with prefix_path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                mismatch = {"row": total + 1, "reason": "blank line in the anchor file"}
                break
            total += 1
            row = json.loads(raw)
            tokens += int(row["meta"]["content_tokens"])
            if row["meta"]["doc_id"] in tombstones:
                skipped += 1
                continue
            try:
                candidate = next(rebuilt)
            except StopIteration:
                mismatch = {"row": total, "reason": "rebuilt stream ended early"}
                break
            if serialize_row(candidate) != raw:
                mismatch = {"row": total, "anchor": row["meta"],
                            "rebuilt": candidate["meta"], "reason": "bytes differ"}
                break
            verified += 1
            if verified % 200000 == 0:
                print(f"[audit-anchor] {verified:,} rows verified", flush=True)
    bytes_ok = digest.hexdigest() == anchor["content_sha256"]
    totals_ok = (total == expected_rows and tokens == anchor["content_tokens"]
                 and verified + skipped == total and total > 0)
    ok = mismatch is None and bytes_ok and totals_ok
    record = {
        "ok": ok, "mismatch": mismatch,
        "anchor_dataset_id": anchor["dataset_id"],
        "anchor_content_sha256": anchor["content_sha256"],
        "anchor_bytes_verified": bytes_ok,
        "anchor_content_tokens": anchor["content_tokens"], "tokens_seen": tokens,
        "rows_expected": expected_rows, "rows_total": total, "rows_verified": verified,
        "rows_tombstoned_skipped": skipped,
        "plan": plan["name"], "documents_sha256": plan["documents_sha256"],
        "snapshot_content_manifest_sha256": plan["snapshot_content_manifest_sha256"],
        "snapshot_missing_documents_sha256": plan["snapshot_missing_documents_sha256"],
        "tokenizer_revision": plan.get("tokenizer_revision"),
        "cleaner_sha256": plan.get("cleaner_sha256"),
        "stream_settings": {key: recipe[key] for key in (
            "max_document_content_tokens", "document_slice_tokens", "max_content_tokens",
            "min_clean_chars")},
        "scope": "continuity of anchored spans only; unseen tails rest on the sealed "
                 "snapshot and historical hash evidence",
        "finished_at": time.time(),
    }
    out = _plan_dir(recipe) / "anchor_audit.json"
    temporary = out.with_name(f".{out.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True))
    os.replace(temporary, out)
    print(f"[audit-anchor] ok={ok} bytes_verified={bytes_ok} verified={verified:,} "
          f"skipped={skipped:,} of {total:,}/{expected_rows:,} rows -> {out}")
    if not ok:
        raise SystemExit(f"anchor audit FAILED: {mismatch or 'bytes/totals mismatch'}")


def anchor_audit_binds(spec: dict, plan_dir: Path) -> bool:
    """True only if a PASSED anchor audit exists whose recorded identities equal the
    dataset spec's (anchor id + hash, plan bytes, snapshot bytes, missing set,
    tokenizer revision, stream settings)."""
    path = plan_dir / "anchor_audit.json"
    if not path.is_file():
        return False
    audit = json.loads(path.read_text())
    selection = spec.get("selection") or {}
    return bool(audit.get("ok")) and all((
        audit.get("anchor_dataset_id") == spec.get("extension_anchor_dataset_id"),
        audit.get("anchor_content_sha256") == spec.get("extension_anchor_content_sha256"),
        audit.get("documents_sha256") == spec.get("ordered_documents_sha256"),
        audit.get("snapshot_content_manifest_sha256") == spec.get("snapshot_content_manifest_sha256"),
        audit.get("snapshot_missing_documents_sha256") == spec.get("snapshot_missing_documents_sha256"),
        audit.get("tokenizer_revision") == spec.get("tokenizer_revision"),
        audit.get("cleaner_sha256") == sha256_file(REPO / "lib" / "corpusprep.py"),
        all(audit.get("stream_settings", {}).get(key) == selection.get(key) for key in (
            "max_document_content_tokens", "document_slice_tokens", "max_content_tokens",
            "min_clean_chars")),
    ))


def seal_snapshot(temporary: Path, root: Path, docs: list[dict],
                  missing: list[dict], extra: dict,
                  *, historical: dict[str, str] | None = None) -> dict:
    """Copy every plan document's text under <plan>/snapshot/, hashing while copying,
    then re-read every copy and verify it before the manifest is written. When a
    historical manifest is given, each base-segment copy is classified: ``verified``
    (its bytes hash to the document's historical cleaned hash), ``uncovered`` (no
    historical entry) or ``indirect`` (an entry exists but hashes a different cleaned
    file); the unverified ids are written to snapshot/historical_unverified.json for
    the anchor re-derivation audit. Returns the snapshot provenance record."""
    snapshot_root = temporary / "snapshot"
    entries = []
    size_bytes = 0
    evidence = {"verified": 0, "uncovered": 0, "indirect": 0}
    unverified: list[str] = []
    started = time.time()
    for doc in docs:
        if doc.get("missing"):
            continue
        rel = doc["text_path"]
        source = root / rel
        target = snapshot_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_mtime > started:
            raise SystemExit(f"source changed while sealing (corpus writer active): {rel}")
        digest = hashlib.sha256()
        with source.open("rb") as src, target.open("wb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                digest.update(block)
                dst.write(block)
            dst.flush()
            for handle in (src, dst):          # keep a 70 GB copy out of the page cache
                try:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
        if source.stat().st_mtime > started:
            raise SystemExit(f"source changed while sealing (corpus writer active): {rel}")
        size = target.stat().st_size
        size_bytes += size
        entries.append({"id": doc["id"], "text_path": rel,
                        "sha256": digest.hexdigest(), "size": size})
        if historical is not None and doc.get("segment") == "v2":
            expected = historical.get(doc["id"])
            if expected is None:
                evidence["uncovered"] += 1
                unverified.append(doc["id"])
            elif expected == digest.hexdigest():
                evidence["verified"] += 1
            else:
                evidence["indirect"] += 1
                unverified.append(doc["id"])
    # Verify: every copy re-hashes to what was written (torn or racing writes fail).
    for entry in entries:
        if sha256_file(snapshot_root / entry["text_path"]) != entry["sha256"]:
            raise SystemExit(f"snapshot copy failed verification: {entry['text_path']}")
    manifest_payload = "".join(json.dumps(
        entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n" for entry in entries).encode()
    (snapshot_root / "manifest.jsonl").write_bytes(manifest_payload)
    (snapshot_root / "historical_unverified.json").write_text(json.dumps(
        {"count": len(unverified), "ids": unverified}, indent=0))
    record = {
        "extension": EXTENSION_VERSION,
        "content_manifest_sha256": sha256_bytes(manifest_payload),
        "files": len(entries),
        "size_bytes": size_bytes,
        "missing_documents": missing,
        "historical_text_evidence": evidence,
        "sealed_at": time.time(),
        **extra,
    }
    (snapshot_root / "provenance.json").write_text(json.dumps(
        record, ensure_ascii=False, indent=2, sort_keys=True))
    return record


def verify_snapshot(plan_dir: Path, plan_rows: list[dict]) -> dict:
    """Authenticate a sealed snapshot: manifest bytes hash to the recorded value, every
    listed file re-hashes to its entry, and every plan document is either listed or
    recorded missing. Returns the snapshot provenance."""
    snapshot_root = plan_dir / "snapshot"
    record = json.loads((snapshot_root / "provenance.json").read_text())
    manifest_path = snapshot_root / "manifest.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    if sha256_bytes(manifest_bytes) != record["content_manifest_sha256"]:
        raise SystemExit(f"snapshot manifest bytes changed: {manifest_path}")
    entries = {}
    for line in manifest_bytes.decode().splitlines():
        if line.strip():
            entry = json.loads(line)
            entries[entry["id"]] = entry
    missing = {item["id"] for item in record.get("missing_documents", [])}
    tombstones = {row["id"] for row in plan_rows if row.get("missing")}
    if missing != tombstones:
        raise SystemExit(
            f"snapshot missing set ({len(missing)}) differs from the plan's tombstones "
            f"({len(tombstones)})")
    for row in plan_rows:
        if row["id"] in missing:
            continue
        entry = entries.get(row["id"])
        if entry is None or entry["text_path"] != row["text_path"]:
            raise SystemExit(f"snapshot is incomplete for plan document {row['id']}")
    if len(entries) != record["files"]:
        raise SystemExit("snapshot file count differs from its provenance")
    for entry in entries.values():
        path = snapshot_root / entry["text_path"]
        if not path.is_file() or path.stat().st_size != entry["size"] \
                or sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"snapshot file failed authentication: {entry['text_path']}")
    return record


def create_composite_plan(root: Path, recipe: dict, plan_dir: Path) -> None:
    composite = recipe["composite"]
    base_dir = datakit.data_root(EXP_DIR) / "plans" / str(composite["base_plan"])
    base_documents = base_dir / "documents.jsonl"
    base_provenance = json.loads((base_dir / "provenance.json").read_text())
    base_bytes = base_documents.read_bytes()
    if sha256_bytes(base_bytes) != base_provenance["documents_sha256"]:
        raise SystemExit(f"base plan bytes changed: {base_documents}")
    base_rows = [json.loads(line) for line in base_bytes.decode().splitlines()
                 if line.strip()]
    if int(base_provenance.get("selection_seed", -1)) != int(recipe["selection_seed"]) \
            or int(base_provenance.get("min_source_chars", -1)) != int(recipe["min_source_chars"]):
        raise SystemExit("composite recipe must keep the base plan's selection_seed "
                         "and min_source_chars")

    manifest_hashes: dict[str, str] = {}
    rows = list(manifest_rows(root, manifest_hashes))
    by_id = {row.get("id"): row for row in rows if row.get("id")}
    pruned = pruned_rows(root)
    historical = _historical_clean_hashes(composite)

    # Segment A keeps EVERY base document at its original rank; documents the corpus
    # has pruned or changed stay in the list as tombstones (``missing``) so ranks never
    # shift and the missing set is reconcilable against the sealed snapshot.
    segment_a: list[dict] = []
    surviving = 0
    missing: list[dict] = []
    for rank, row in enumerate(base_rows):
        entry = {**row, "rank": rank, "segment": "v2"}
        current = by_id.get(row["id"]) or {}
        if not (root / row["text_path"]).is_file():
            prune = pruned.get(row["id"]) or {}
            reason = prune.get("reason") or "absent"
            missing.append({"id": row["id"], "rank": rank, "text_path": row["text_path"],
                            "sha256": row.get("sha256"), "reason": reason,
                            "pruned_at": prune.get("pruned_at")})
            segment_a.append({**entry, "missing": reason})
            continue
        if historical is not None and row["id"] in historical \
                and current.get("corpus_sha256") \
                and current["corpus_sha256"] != historical[row["id"]]:
            missing.append({"id": row["id"], "rank": rank, "text_path": row["text_path"],
                            "sha256": row.get("sha256"), "reason": "content-changed",
                            "historical_corpus_sha256": historical[row["id"]],
                            "current_corpus_sha256": current["corpus_sha256"]})
            segment_a.append({**entry, "missing": "content-changed"})
            continue
        surviving += 1
        segment_a.append(entry)

    anchor = recipe.get("extension_anchor")
    if anchor is not None:
        terminal = _last_row_doc_id(
            datakit.data_file(datakit.data_root(EXP_DIR) / "objects" / str(anchor["dataset_id"])))
        if terminal in {item["id"] for item in missing}:
            raise SystemExit(
                f"the anchor's terminal document {terminal} is missing or changed; the "
                "stream cannot be continued from its recorded position")
    exclude = {
        "ids": {row["id"] for row in base_rows},
        "sha256": {row["sha256"] for row in base_rows if row.get("sha256")},
        "corpus_sha256": (
            {by_id[row["id"]]["corpus_sha256"] for row in base_rows
             if row["id"] in by_id and by_id[row["id"]].get("corpus_sha256")}
            | {pruned[row["id"]]["corpus_sha256"] for row in base_rows
               if row["id"] in pruned and pruned[row["id"]].get("corpus_sha256")}
            | ({historical[row["id"]] for row in base_rows if row["id"] in historical}
               if historical else set())),
    }
    new_docs, selection = select_documents(root, recipe, rows=rows, exclude=exclude)
    segment_b = [{key: row.get(key) for key in PLAN_ROW_KEYS}
                 | {"text_path": str(row["_text_path"].relative_to(root)),
                    "rank": len(base_rows) + index, "segment": "new"}
                 for index, row in enumerate(new_docs)]
    plan_rows = segment_a + segment_b
    payload = plan_payload(plan_rows)
    tokenizer_path = _tokenizer_path()
    provenance = {
        **plan_expectations(recipe),
        **{key: selection[key] for key in (
            "probe_fingerprint", "ordered_documents_sha256", "manifest_snapshot_sha256",
            "holdout_ids_excluded", "holdout_duplicates_excluded", "composite_base_excluded",
            "topics", "sources")},
        "eligible_documents": len(plan_rows),
        "segment_documents": {"v2": surviving, "new": len(segment_b)},
        "composite_base_documents_sha256": base_provenance["documents_sha256"],
        "composite_base_documents": len(base_rows),
        "missing_documents": len(missing),
        "missing_reasons": dict(Counter(item["reason"] for item in missing)),
        "manifest_files_sha256": manifest_hashes,
        "registry_files_sha256": {path.name: sha256_file(path) for path in registry_files(root)},
        "historical_manifest": composite.get("historical_manifest"),
        "historical_manifest_files_sha256": _historical_manifest_hashes(composite),
        "historical_documents_compared": (
            sum(1 for row in base_rows if row["id"] in historical) if historical else 0),
        "cleaner_sha256": sha256_file(REPO / "lib" / "corpusprep.py"),
        "tokenizer": TOKENIZER,
        "tokenizer_revision": Path(tokenizer_path).name,
        "corpus_commit_at_creation": corpus_commit(root),
        "corpus_dirty_paths_at_creation": corpus_dirty_paths(root),
        "documents_sha256": sha256_bytes(payload),
    }

    def build(temporary: Path) -> None:
        (temporary / "documents.jsonl").write_bytes(payload)
        record = seal_snapshot(temporary, root, plan_rows, missing, {
            "plan": provenance["name"], "documents_sha256": provenance["documents_sha256"],
            "corpus_commit": provenance["corpus_commit_at_creation"],
        }, historical=historical)
        provenance["snapshot_content_manifest_sha256"] = record["content_manifest_sha256"]
        provenance["snapshot_size_bytes"] = record["size_bytes"]
        provenance["historical_text_evidence"] = record["historical_text_evidence"]
        (temporary / "provenance.json").write_text(json.dumps(
            provenance, ensure_ascii=False, indent=2, sort_keys=True))
        verify_snapshot(temporary, plan_rows)

    _publish_plan(plan_dir, build)


def _plan_dir(recipe: dict) -> Path:
    name = str(recipe["stream_plan"])
    if not name or any(part in {"", ".", ".."} for part in Path(name).parts):
        raise SystemExit(f"invalid stream_plan name: {name!r}")
    return datakit.data_root(EXP_DIR) / "plans" / name


def plan_provenance(recipe: dict) -> tuple[list[dict], dict]:
    """Read an existing plan's rows and provenance (drift and byte checks, snapshot
    identity fields merged) WITHOUT touching any document text. Enough to compute the
    dataset spec; building additionally resolves and verifies the text."""
    plan_dir = _plan_dir(recipe)
    documents_path = plan_dir / "documents.jsonl"
    provenance_path = plan_dir / "provenance.json"
    if not (documents_path.is_file() and provenance_path.is_file()):
        if plan_dir.exists():
            raise SystemExit(f"incomplete stream plan at {plan_dir}; inspect it manually")
        raise FileNotFoundError(plan_dir)
    provenance = json.loads(provenance_path.read_text())
    expected = plan_expectations(recipe)
    drift = {key: (provenance.get(key), value) for key, value in expected.items()
             if provenance.get(key) != value}
    if drift:
        raise SystemExit(
            f"stream plan {provenance.get('name')!r} conflicts with the active recipe: "
            f"{drift}; choose a new stream_plan name for a new campaign")
    payload = documents_path.read_bytes()
    if sha256_bytes(payload) != provenance["documents_sha256"]:
        raise SystemExit(f"stream plan bytes changed: {documents_path}")
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    snapshot_provenance_path = plan_dir / "snapshot" / "provenance.json"
    if recipe.get("composite") and not snapshot_provenance_path.is_file():
        raise SystemExit(f"composite plan {provenance.get('name')!r} has no sealed snapshot")
    if snapshot_provenance_path.is_file():
        snapshot = json.loads(snapshot_provenance_path.read_text())
        recorded = provenance.get("snapshot_content_manifest_sha256")
        if recipe.get("composite") and not recorded:
            raise SystemExit(
                f"composite plan {provenance.get('name')!r} records no snapshot manifest hash")
        if recorded and recorded != snapshot["content_manifest_sha256"]:
            raise SystemExit(
                f"plan {provenance.get('name')!r} records snapshot manifest {recorded[:12]} "
                f"but the snapshot says {snapshot['content_manifest_sha256'][:12]}")
        missing = sorted({item["id"] for item in snapshot.get("missing_documents", [])})
        tombstones = sorted({row["id"] for row in rows if row.get("missing")})
        if missing != tombstones:
            raise SystemExit("snapshot missing set differs from the plan's tombstones")
        provenance = {
            **provenance,
            "snapshot_content_manifest_sha256": snapshot["content_manifest_sha256"],
            "snapshot_size_bytes": snapshot["size_bytes"],
            "snapshot_documents": sum(1 for row in rows if row["id"] not in set(missing)),
            "snapshot_missing_documents": len(missing),
            "snapshot_missing_documents_sha256": hashlib.sha256(
                "\n".join(missing).encode()).hexdigest(),
        }
    return rows, provenance


def load_or_create_plan(root: Path, recipe: dict, *, verify: bool = True,
                        create: bool = True) -> tuple[list[dict], dict]:
    """Freeze document membership/order once so a growing sibling corpus cannot drift.
    Composite plans additionally freeze the bytes (sealed snapshot, verified here).
    Returns the streamable documents (text paths resolved, missing ones dropped) and
    the plan provenance."""
    plan_dir = _plan_dir(recipe)
    try:
        rows, provenance = plan_provenance(recipe)
    except FileNotFoundError:
        if not create:
            raise
        if recipe.get("composite"):
            create_composite_plan(root, recipe, plan_dir)
        else:
            create_simple_plan(root, recipe, plan_dir)
        rows, provenance = plan_provenance(recipe)

    snapshot_root = plan_dir / "snapshot"
    snapshot_provenance_path = snapshot_root / "provenance.json"
    snapshot_provenance = None
    if snapshot_provenance_path.is_file():
        snapshot_provenance = (verify_snapshot(plan_dir, rows) if verify
                               else json.loads(snapshot_provenance_path.read_text()))
    expected_missing = (
        {item["id"] for item in snapshot_provenance.get("missing_documents", [])}
        if snapshot_provenance is not None else set()
    )
    docs = []
    missing = []
    for row in rows:
        if row["id"] in expected_missing:
            missing.append(row)
            continue
        path = ((snapshot_root / row["text_path"])
                if snapshot_provenance is not None else (root / row["text_path"]))
        if snapshot_provenance is None and not path.is_file():
            missing.append(row)
            continue
        docs.append({**row, "_text_path": path})
    if missing and snapshot_provenance is None:
        raise SystemExit(f"stream plan source disappeared: {root / missing[0]['text_path']}")
    if snapshot_provenance is not None:
        actual_missing = {row["id"] for row in missing}
        if actual_missing != expected_missing:
            raise SystemExit(
                "stream snapshot membership changed: "
                f"expected {len(expected_missing)} missing, found {len(actual_missing)}")
    return docs, provenance


def _segments(docs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Consecutive runs of the plan's ``segment`` label, in plan order (a plan without
    labels is one segment)."""
    segments: list[tuple[str, list[dict]]] = []
    for doc in docs:
        label = doc.get("segment") or "all"
        if not segments or segments[-1][0] != label:
            segments.append((label, []))
        segments[-1][1].append(doc)
    return segments


def stream_rows(
    docs: list[dict], tokenizer, recipe: dict, stats: dict,
    *, prefix_path: Path | None = None, prefix_content_tokens: int | None = None,
    prefix_content_sha256: str | None = None, allow_exhaustion: bool = False,
):
    """Yield dataset rows: the anchor prefix (verbatim, byte-hashed), then the first
    segment's rounds continued from the anchor's position, then every later segment
    from round 0. The stream stops the moment the next row would exceed the target.
    Exhausting every segment below the target is an error unless ``allow_exhaustion``
    is set, in which case the shortfall is registered in ``stats``."""
    target = int(recipe["target_content_tokens"])
    document_limit = int(recipe["max_document_content_tokens"])
    slice_limit = int(recipe["document_slice_tokens"])
    chunk_limit = int(recipe["max_content_tokens"])
    total = chunks = characters = 0
    used_docs: set[str] = set()
    prefix_keys: set[tuple[str, int, int]] = set()
    resume_doc_id = None
    resume_round = None
    resume_token_start = None
    topic_tokens: dict[str, int] = defaultdict(int)
    source_tokens: dict[str, int] = defaultdict(int)
    segment_tokens: dict[str, int] = defaultdict(int)

    def account(row: dict, segment: str) -> None:
        nonlocal total, chunks, characters
        meta = row["meta"]
        n_tokens = int(meta["content_tokens"])
        used_docs.add(meta["doc_id"])
        total += n_tokens
        chunks += 1
        characters += len(row["text"])
        topic_tokens[meta.get("topic") or "unknown"] += n_tokens
        source_tokens[meta.get("source") or "unknown"] += n_tokens
        segment_tokens[segment] += n_tokens

    def finish(*, exhausted: bool, rounds_entered: int) -> None:
        stats.update({
            "documents": len(used_docs), "chunks": chunks, "content_tokens": total,
            "target_content_tokens": target, "characters": characters,
            "rounds_entered": rounds_entered,
            "topic_content_tokens": dict(sorted(topic_tokens.items())),
            "source_content_tokens": dict(sorted(source_tokens.items())),
            "segment_content_tokens": dict(sorted(segment_tokens.items())),
            "exhausted": exhausted,
            "shortfall_tokens": max(0, target - total) if exhausted else 0,
        })

    if prefix_path is not None:
        digest = hashlib.sha256()
        with prefix_path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                if not raw.strip():
                    raise RuntimeError("anchor bytes drift: blank line in the anchor file")
                row = json.loads(raw)
                if raw != serialize_row(row):
                    raise RuntimeError(
                        "anchor bytes drift: an anchor line is not in the canonical "
                        "serialization (the anchor file was rewritten)")
                meta = row["meta"]
                n_tokens = int(meta["content_tokens"])
                if total + n_tokens > target:
                    raise RuntimeError(
                        f"anchor prefix exceeds target {target:,} content tokens")
                prefix_keys.add((
                    meta["doc_id"], int(meta["round"]),
                    int(meta["document_token_start"]),
                ))
                resume_doc_id = meta["doc_id"]
                resume_round = int(meta["round"])
                resume_token_start = (
                    int(meta["document_token_start"]) + int(meta["content_tokens"]))
                account(row, "anchor")
                yield row
        if total != prefix_content_tokens:
            raise RuntimeError(
                f"anchor content token drift: {total:,} != {prefix_content_tokens:,}")
        if prefix_content_sha256 and digest.hexdigest() != prefix_content_sha256:
            raise RuntimeError(
                "anchor bytes drift: the re-emitted prefix does not hash to "
                f"{prefix_content_sha256}")

    rounds_entered = 0
    for segment_index, (segment, segment_docs) in enumerate(_segments(docs)):
        seg_resume_round = seg_resume_doc_index = seg_resume_token_start = None
        if segment_index == 0 and resume_doc_id is not None:
            seg_resume_doc_index = next(
                (i for i, doc in enumerate(segment_docs) if doc["id"] == resume_doc_id), None)
            if seg_resume_doc_index is None:
                raise RuntimeError(
                    f"anchor resume document is absent from the first segment: {resume_doc_id}")
            seg_resume_round = resume_round
            seg_resume_token_start = resume_token_start
        for round_index, start in enumerate(range(0, document_limit, slice_limit)):
            if seg_resume_round is not None and round_index < seg_resume_round:
                continue
            rounds_entered = max(rounds_entered, round_index + 1)
            stop = min(start + slice_limit, document_limit)
            for doc_index, doc in enumerate(segment_docs):
                if (seg_resume_round is not None and round_index == seg_resume_round
                        and doc_index < seg_resume_doc_index):
                    continue
                text = corpusprep.clean_body(doc["_text_path"].read_text(errors="replace"))
                if len(text) < int(recipe["min_clean_chars"]):
                    continue
                encoded = tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=True, max_length=stop,
                )
                offsets = encoded["offset_mapping"]
                if len(offsets) <= start:
                    continue
                end = min(stop, len(offsets))
                first_token = start
                if (seg_resume_round is not None and round_index == seg_resume_round
                        and doc_index == seg_resume_doc_index):
                    first_token = max(first_token, seg_resume_token_start)
                for token_start in range(first_token, end, chunk_limit):
                    token_end = min(token_start + chunk_limit, end)
                    n_tokens = token_end - token_start
                    key = (doc["id"], round_index, token_start)
                    if key in prefix_keys:
                        continue
                    # Keep row boundaries stable across the entire scale ladder. The
                    # target is therefore a strict upper bound, normally short by < one
                    # chunk.
                    if total + n_tokens > target:
                        finish(exhausted=False, rounds_entered=rounds_entered)
                        return
                    lo = offsets[token_start][0]
                    hi = offsets[token_end][0] if token_end < len(offsets) else offsets[-1][1]
                    piece = text[lo:hi]
                    if not piece.strip():
                        continue
                    topic = doc.get("topic") or "unknown"
                    source = doc.get("source") or "unknown"
                    row = {
                        "text": piece,
                        "meta": {
                            "doc_id": doc["id"], "topic": topic, "source": source,
                            "license": doc.get("license"), "round": round_index,
                            "document_token_start": token_start,
                            "content_tokens": n_tokens,
                        },
                    }
                    yield row
                    account(row, segment)

    if total < target and not allow_exhaustion:
        finish(exhausted=True, rounds_entered=rounds_entered)
        raise RuntimeError(
            f"corpus capacity {total:,} tokens is below target {target:,}; "
            "lower the target or increase max_document_content_tokens")
    finish(exhausted=total < target, rounds_entered=rounds_entered)


def _tokenizer_path(revision: str | None = None) -> str:
    """Local snapshot of the tokenizer; composite plans pin the exact revision they
    were created with so a later cache update cannot change continuation bytes."""
    from huggingface_hub import snapshot_download
    return snapshot_download(TOKENIZER, revision=revision, local_files_only=True)


def assert_plan_identities(recipe: dict, plan: dict, tokenizer_path: str) -> None:
    """A composite plan streams with exactly the tokenizer revision and cleaner
    implementation recorded at its creation."""
    if not recipe.get("composite"):
        return
    if Path(tokenizer_path).name != plan["tokenizer_revision"]:
        raise SystemExit(
            f"tokenizer revision {Path(tokenizer_path).name} != plan's "
            f"{plan['tokenizer_revision']}")
    cleaner = sha256_file(REPO / "lib" / "corpusprep.py")
    if cleaner != plan["cleaner_sha256"]:
        raise SystemExit("lib/corpusprep.py changed since the composite plan was sealed")


def anchor_for(recipe: dict) -> tuple[Path | None, dict | None]:
    """Resolve and authenticate the extension anchor dataset named by the recipe."""
    anchor = recipe.get("extension_anchor")
    target = int(recipe["target_content_tokens"])
    if anchor is None or target <= int(anchor["content_tokens"]):
        return None, None
    anchor_dataset_id = str(anchor["dataset_id"])
    anchor_content_sha256 = str(anchor["content_sha256"])
    anchor_dir = datakit.data_root(EXP_DIR) / "objects" / anchor_dataset_id
    prefix_path = datakit.data_file(anchor_dir)
    anchor_provenance = datakit.provenance(anchor_dir)
    if anchor_provenance.get("dataset_id") != anchor_dataset_id:
        raise SystemExit(f"anchor dataset identity drift at {anchor_dir}")
    if anchor_provenance.get("content_sha256") != anchor_content_sha256:
        raise SystemExit(
            f"anchor dataset content drift: "
            f"{anchor_provenance.get('content_sha256')} != {anchor_content_sha256}")
    anchor_tokens = int((anchor_provenance.get("stats") or {}).get("content_tokens") or 0)
    if anchor_tokens != int(anchor["content_tokens"]):
        raise SystemExit(
            f"anchor dataset token drift: {anchor_tokens:,} != {int(anchor['content_tokens']):,}")
    return prefix_path, {
        "dataset_id": anchor_dataset_id, "content_sha256": anchor_content_sha256,
        "content_tokens": int(anchor["content_tokens"]),
    }


def build_spec(recipe: dict, plan: dict, tokenizer_revision: str, anchor: dict | None) -> dict:
    """The dataset identity (cache key). Everything that changes the bytes is here."""
    spec = {
        "kind": "cpt_scale",
        "source": "nekaise-corpus",
        "corpus_commit": plan["corpus_commit_at_creation"],
        "manifest_sha256": plan["manifest_snapshot_sha256"],
        "tokenizer": TOKENIZER,
        "tokenizer_revision": tokenizer_revision,
        "probe_fingerprint": plan["probe_fingerprint"],
        "stream_plan": plan["name"],
        "ordered_documents_sha256": plan["documents_sha256"],
        "ordering": ORDERING_VERSION,
        "stream": STREAM_VERSION,
        "selection": {
            key: recipe[key] for key in (
                "target_content_tokens", "max_document_content_tokens",
                "document_slice_tokens", "max_content_tokens", "min_source_chars",
                "min_clean_chars", "selection_seed", "stream_plan",
            )
        },
        "license_policy": "provenance_only_no_filter",
        "transfer_documents_excluded": True,
    }
    if recipe.get("composite"):
        spec["composite"] = {
            "version": COMPOSITE_VERSION,
            "base_plan": plan["composite_base_plan"],
            "base_documents_sha256": plan["composite_base_documents_sha256"],
            "snapshot_content_manifest_sha256": plan["snapshot_content_manifest_sha256"],
            "snapshot_missing_documents_sha256": plan["snapshot_missing_documents_sha256"],
        }
    if recipe.get("allow_exhaustion"):
        spec["allow_exhaustion"] = True
    if anchor is not None:
        if "snapshot_content_manifest_sha256" not in plan:
            raise SystemExit("anchored extension requires a sealed stream-plan snapshot")
        spec.update({
            "extension": EXTENSION_VERSION,
            "extension_anchor_dataset_id": anchor["dataset_id"],
            "extension_anchor_content_sha256": anchor["content_sha256"],
            "extension_anchor_content_tokens": anchor["content_tokens"],
            "snapshot_content_manifest_sha256": plan["snapshot_content_manifest_sha256"],
            "snapshot_documents": plan["snapshot_documents"],
            "snapshot_missing_documents": plan["snapshot_missing_documents"],
            "snapshot_missing_documents_sha256": plan["snapshot_missing_documents_sha256"],
        })
    return spec


def spec_for_config(config_path: Path) -> dict | None:
    """The exact dataset spec a config resolves to, without building or verifying the
    snapshot bytes (for manifest tools). None when the plan does not exist yet."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    recipe = cfg["data"]["corpus"]
    corpus_root = WORKSPACE.resolve_input(recipe["path"])
    try:
        _rows, plan = plan_provenance(recipe)
    except FileNotFoundError:
        return None
    _prefix, anchor = anchor_for(recipe)
    revision = (plan["tokenizer_revision"] if recipe.get("composite")
                else Path(_tokenizer_path()).name)
    return build_spec(recipe, plan, revision, anchor)


def main(argv=None) -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=CONFIG,
        help="CPT config whose data.corpus recipe should be built",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="create (and seal) the stream plan, verify it, and stop without building",
    )
    parser.add_argument(
        "--audit-anchor", action="store_true",
        help="re-derive every extension_anchor row from the sealed plan and require "
             "byte equality (writes plans/<name>/anchor_audit.json)",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    recipe = cfg["data"]["corpus"]
    corpus_root = WORKSPACE.resolve_input(recipe["path"])
    if not manifest_files(corpus_root) or not (corpus_root / "text").is_dir():
        raise SystemExit(
            f"corpus is incomplete at {corpus_root}: need manifest/ and local text/")

    docs, plan = load_or_create_plan(corpus_root, recipe)
    if args.plan_only:
        print(f"[plan] {plan['name']}: {len(docs):,} documents"
              f"{' (sealed snapshot verified)' if 'snapshot_content_manifest_sha256' in plan else ''}")
        return
    tokenizer_path = _tokenizer_path(plan.get("tokenizer_revision")
                                     if recipe.get("composite") else None)
    assert_plan_identities(recipe, plan, tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if args.audit_anchor:
        audit_anchor(recipe, docs, plan, tokenizer)
        return
    stats = {
        "purpose": "full_parameter_cpt_data_scaling",
        **{key: plan[key] for key in (
            "eligible_documents", "topics", "sources", "holdout_ids_excluded",
            "holdout_duplicates_excluded", "manifest_snapshot_sha256",
        )},
        "stream_plan": plan["name"],
        "stream_plan_sha256": plan["documents_sha256"],
    }
    if recipe.get("composite"):
        stats.update({
            "composite_base_plan": plan["composite_base_plan"],
            "segment_documents": plan.get("segment_documents"),
            "snapshot_missing_documents": plan["snapshot_missing_documents"],
        })
    prefix_path, anchor = anchor_for(recipe)
    if anchor is not None:
        stats.update({
            "extension_anchor_dataset_id": anchor["dataset_id"],
            "extension_anchor_content_sha256": anchor["content_sha256"],
            "extension_anchor_content_tokens": anchor["content_tokens"],
            "snapshot_documents": plan["snapshot_documents"],
            "snapshot_missing_documents": plan["snapshot_missing_documents"],
            "snapshot_content_manifest_sha256":
                plan["snapshot_content_manifest_sha256"],
        })
    spec = build_spec(recipe, plan, Path(tokenizer_path).name, anchor)
    if datakit.exists(EXP_DIR, spec):
        artifact = datakit.activate(EXP_DIR, spec)
        print(f"[build] cache hit -> {artifact}")
        return

    artifact = datakit.write(
        EXP_DIR, spec,
        stream_rows(
            docs, tokenizer, recipe, stats, prefix_path=prefix_path,
            prefix_content_tokens=anchor["content_tokens"] if anchor else None,
            prefix_content_sha256=anchor["content_sha256"] if anchor else None,
            allow_exhaustion=bool(recipe.get("allow_exhaustion")),
        ),
        stats=stats,
        recipe_path=__file__,
    )
    print(
        f"[build] {stats['documents']:,} docs -> {stats['chunks']:,} chunks, "
        f"{stats['content_tokens']:,}/{stats['target_content_tokens']:,} content tokens "
        f"across {len(stats['topic_content_tokens'])} topics"
        f"{' (EXHAUSTED, shortfall %s)' % format(stats['shortfall_tokens'], ',') if stats.get('exhausted') else ''}"
        f" -> {artifact}")


if __name__ == "__main__":
    main()
