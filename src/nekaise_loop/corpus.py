"""Read-only corpus discovery and snapshots; the teacher chooses the curriculum."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager, closing
from pathlib import Path



@contextmanager
def settled_corpus(root: Path):
    # Readers take verified historical document snapshots, not a global corpus
    # checkpoint. Atomic manifests + mandatory corpus hashes tolerate growth.
    # Eligibility must remain identical for the entire selection operation.
    path = root / "registry/eligibility.json"
    before = path.read_bytes()
    yield
    if path.read_bytes() != before:
        raise RuntimeError("Corpus eligibility changed during selection; retry to take a consistent snapshot.")


def eligible(row: dict, policy: dict) -> bool:
    if row.get("status") != "ok" or not row.get("license") or row["license"] in {"proprietary-internal", "restricted", "pointer-only"}:
        return False
    for rule in policy.values():
        match = rule["match"]
        if all((str(row.get("id", "")).startswith(value) if key == "id_prefix" else row.get(key) == value) for key, value in match.items()):
            return False
    return True


def policy_at(root):
    data = json.loads((root/"registry/eligibility.json").read_text())
    if data.get("version") != 1 or not isinstance(data.get("restrictions"), dict):
        raise ValueError("Unsupported corpus eligibility policy")
    for rule in data["restrictions"].values():
        if rule.get("status") != "restricted" or not rule.get("match") or set(rule["match"]) - {"source", "id_prefix"}:
            raise ValueError("Invalid corpus eligibility selector")
    return data["restrictions"]


def manifest_rows(root, shard=None):
    if shard is not None and not re.fullmatch(r"[\w-]+", shard):
        raise ValueError("Invalid manifest shard")
    paths = [root/"manifest"/f"{shard}.jsonl"] if shard else sorted((root/"manifest").glob("*.jsonl"))
    for path in paths:
        with path.open() as handle:
            for line in handle:
                yield json.loads(line)


def search_sources(root, query="", prefix="", offset=0, limit=20):
    """Candidate discovery across every page, without the old 600-document frontier."""
    policy = policy_at(root)
    index = root/"workspace/corpus-index.sqlite3"
    if index.exists():
        with closing(sqlite3.connect(f"file:{index}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            where, args = ["status='ok'", "id>=?", "id<?"], [prefix, prefix+"\uffff"]
            for word in query.split():
                where.append("(title LIKE ? OR id LIKE ? OR topic LIKE ?)")
                args.extend(["%"+word+"%"]*3)
            rows = [dict(r) for r in db.execute(f"SELECT id,title,topic,source,license,status FROM documents WHERE {' AND '.join(where)} ORDER BY id LIMIT ? OFFSET ?", (*args, limit+1, offset))]
        # The index is only discovery. Admission always reads the authoritative manifest.
        return {"rows": [{**r, "eligible_candidate": eligible(r, policy)} for r in rows[:limit]], "next_offset": offset+limit if len(rows)>limit else None}
    found, skipped = [], 0
    words = query.lower().split()
    for row in manifest_rows(root):
        haystack = " ".join(str(row.get(k, "")) for k in ("id", "title", "topic")).lower()
        if not row["id"].startswith(prefix) or not all(w in haystack for w in words) or not eligible(row, policy):
            continue
        if skipped < offset:
            skipped += 1
            continue
        found.append({k:row.get(k) for k in ("id", "title", "topic", "license", "source")})
        if len(found)>limit:
            break
    return {"rows": found[:limit], "next_offset": offset+limit if len(found)>limit else None}


def read_source(root, document_id, start=0, length=0):
    if not re.fullmatch(r"[\w.-]+", document_id) or document_id in {".", ".."} or start<0 or length<0:
        raise ValueError("Invalid source reference")
    with settled_corpus(root):
        policy = policy_at(root)
        shard = None
        index = root/"workspace/corpus-index.sqlite3"
        if index.exists():
            with closing(sqlite3.connect(f"file:{index}?mode=ro", uri=True)) as db:
                indexed = db.execute("SELECT manifest_shard FROM documents WHERE id=?", (document_id,)).fetchone()
            if indexed:
                shard = indexed[0]
        row = next((r for r in manifest_rows(root, shard) if r["id"] == document_id), None)
        if not row or not eligible(row, policy):
            raise ValueError(f"Source is missing or ineligible: {document_id}")
        raw = (root/"corpus"/f"{document_id}.md").read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != row.get("corpus_sha256"):
            raise ValueError(f"Source hash mismatch: {document_id}")
        text = raw.decode("utf-8")
        excerpt = text[start:start+length] if length else text[start:]
        if not excerpt.strip():
            raise ValueError("Selected source span is empty")
        return {"id": document_id, "title": row.get("title", document_id), "url": row.get("url", ""), "license": row["license"], "topic": row.get("topic", ""), "source_sha256": source_hash, "manifest_sha256": row.get("sha256"), "text": excerpt, "span_start": start, "span_length": len(excerpt), "document_chars": len(text), "selection_reason": "Selected by the teacher", "replay": False}
