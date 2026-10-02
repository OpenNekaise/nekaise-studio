"""Immutable, read-only corpus inventories; bytes are pinned when a document is reached.

An inventory fixes eligible IDs for one pass. New arrivals/revisions enter the next
pass. Missing or newly restricted documents are operational failures, never coverage.
"""
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .corpus import eligible, policy_at, read_source, settled_corpus
from .storage import now


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_inventory(root, workspace, cancelled=lambda: False):
    from .domain_corpus import is_domain, DomainSource
    if is_domain(root):
        return DomainSource(root).inventory(workspace, cancelled)
    directory = workspace / "curriculum/inventories"
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".sqlite3")
    os.close(fd)
    path = Path(temporary)
    admitted, excluded, manifests = 0, 0, {}
    try:
        with settled_corpus(root), closing(sqlite3.connect(path)) as out:
            policy = policy_at(root)
            policy_hash = file_hash(root / "registry/eligibility.json")
            # Bulk sequential insertion then index construction avoids millions
            # of random B-tree writes. Store only authoritative source metadata.
            out.execute("CREATE TABLE documents (position TEXT NOT NULL, document_id TEXT NOT NULL, metadata TEXT NOT NULL)")
            fields = ("id", "title", "url", "source", "license", "topic", "status", "sha256", "corpus_sha256")
            for manifest in sorted((root / "manifest").glob("*.jsonl")):
                batch = []
                before = manifest.stat()
                h = hashlib.sha256()
                with manifest.open("rb") as handle:
                    for i, line in enumerate(handle):
                        if i % 1000 == 0 and cancelled():
                            from .processes import Cancelled
                            raise Cancelled("Corpus inventory cancelled")
                        h.update(line)
                        row = json.loads(line)
                        if not eligible(row, policy):
                            excluded += 1
                            continue
                        metadata = {k: row.get(k) for k in fields}
                        metadata["manifest_name"] = manifest.name
                        batch.append((hashlib.sha256(row["id"].encode()).hexdigest(), row["id"], json.dumps(metadata)))
                        admitted += 1
                        if len(batch) == 1000:
                            out.executemany("INSERT INTO documents VALUES (?,?,?)", batch)
                            batch.clear()
                after = manifest.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise RuntimeError("Corpus manifest changed while inventorying; retry: " + manifest.name)
                manifests[manifest.name] = {"sha256": h.hexdigest(), "size": after.st_size, "mtime_ns": after.st_mtime_ns}
                out.executemany("INSERT INTO documents VALUES (?,?,?)", batch)
            if not admitted:
                raise ValueError("No eligible downloaded corpus documents; cannot start a coverage pass")
            out.execute("CREATE UNIQUE INDEX positions ON documents(position)")
            out.execute("CREATE UNIQUE INDEX identifiers ON documents(document_id)")
            out.execute("CREATE TABLE manifests (name TEXT PRIMARY KEY, stamp TEXT NOT NULL)")
            out.executemany("INSERT INTO manifests VALUES (?,?)", [(name, json.dumps(stamp)) for name, stamp in manifests.items()])
            out.commit()
        checksum = file_hash(path)
        target = directory / (checksum + ".sqlite3")
        os.replace(path, target)
        return {"sha256": checksum, "documents": admitted, "excluded_documents": excluded,
                "eligibility_sha256": policy_hash, "manifest_count": len(manifests), "created_at": now(),
                "revision_policy": "Pin verified bytes on first encounter; refresh IDs and revisions next pass"}
    finally:
        path.unlink(missing_ok=True)


def next_document(workspace, inventory, position):
    path = workspace / "curriculum/inventories" / (inventory["sha256"] + ".sqlite3")
    stat = path.stat()
    _verify_inventory(str(path), inventory["sha256"], stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as db:
        row = db.execute("SELECT position,document_id,metadata FROM documents WHERE position>? ORDER BY position LIMIT 1", (position,)).fetchone()
        if not row:
            return None
        metadata = json.loads(row[2])
        stamp = json.loads(db.execute("SELECT stamp FROM manifests WHERE name=?", (metadata["manifest_name"],)).fetchone()[0])
    return {"position": row[0], "document_id": row[1], "metadata": metadata, "manifest_stamp": stamp}


@lru_cache(maxsize=32)
def _verify_inventory(path, expected, inode, size, mtime):
    if file_hash(Path(path)) != expected:
        raise ValueError("Corpus inventory integrity failed")


def read_inventory_source(root, inventory, entry):
    from .domain_corpus import is_domain, DomainSource
    if is_domain(root):
        source = DomainSource(root)
        if (inventory.get('domain_source_hash') != source.identity
                or entry['metadata'].get('domain_source_hash') != source.identity):
            raise ValueError('Domain inventory belongs to a different binding')
        return source.read(entry['document_id'], forward=True)
    row = entry["metadata"]
    stamp = entry["manifest_stamp"]
    current = (root / "manifest" / row["manifest_name"]).stat()
    # A changed publisher shard needs fresh authoritative admission. Unchanged
    # shards use the immutable inventory metadata, avoiding repeated huge scans.
    if (current.st_size, current.st_mtime_ns) != (stamp["size"], stamp["mtime_ns"]):
        return read_source(root, entry["document_id"])
    with settled_corpus(root):
        if not eligible(row, policy_at(root)):
            raise ValueError("Inventoried source became ineligible: " + entry["document_id"])
        path = root / "corpus" / (entry["document_id"] + ".md")
        if path.parent.resolve() != (root / "corpus").resolve():
            raise ValueError("Invalid inventory document path")
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != row.get("corpus_sha256"):
            # Corpus can publish a new revision before its manifest replacement.
            # The normal reader either verifies that revision or fails visibly.
            return read_source(root, entry["document_id"])
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValueError("Inventoried source is empty: " + entry["document_id"])
        return {"id": row["id"], "title": row.get("title") or row["id"], "url": row.get("url") or "",
                "source": row.get("source"), "license": row["license"], "topic": row.get("topic") or "",
                "source_sha256": source_hash, "manifest_sha256": row.get("sha256"), "text": text,
                "span_start": 0, "span_length": len(text), "document_chars": len(text),
                "selection_reason": "Forward corpus coverage", "replay": False}
