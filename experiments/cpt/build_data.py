#!/usr/bin/env python3
"""Build a deterministic, nested full-parameter CPT dataset from nekaise-corpus.

The scale ladder changes only ``target_content_tokens``.  Every larger artifact starts
with exactly the complete rows of every smaller artifact: documents have one fixed
weighted order and are exposed in fixed token slices (all documents' first slice,
then all documents' second slice, and so on).  Licenses are recorded as provenance,
never used as a filter.

    python experiments/cpt/build_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
from collections import defaultdict
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


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def corpus_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def manifest_rows(root: Path):
    for path in sorted((root / "manifest").glob("*.jsonl")):
        with path.open(errors="replace") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def probe_holdout() -> tuple[set[str], str]:
    provenance = json.loads(PROBE_PROVENANCE.read_text())
    splits = provenance["doc_splits"]
    # Absorption documents intentionally remain train eligible. Transfer documents
    # and exact-content duplicates of them are never visible to CPT.
    ids = set(splits["dev"]) | set(splits["frozen"])
    return ids, provenance["fingerprint"]


def select_documents(root: Path, recipe: dict) -> tuple[list[dict], dict]:
    holdout_ids, probe_fingerprint = probe_holdout()
    rows = list(manifest_rows(root))
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
    excluded_ids = excluded_duplicates = 0
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

    seed = int(recipe["selection_seed"])

    def priority(row: dict) -> tuple[float, str]:
        topic = row.get("topic") or "unknown"
        source = row.get("source") or "unknown"
        bucket = (topic, source)
        # Topic and source exposure grows with sqrt(raw mass), while documents within
        # a source share that source's mass. This softens dominant crawls without
        # flattening the real corpus distribution.
        weight = (math.sqrt(topic_mass[topic]) / topic_normalizer
                  * math.sqrt(source_mass[bucket]) / source_normalizer[topic]
                  / source_count[bucket])
        digest = hashlib.sha256(
            f"{ORDERING_VERSION}:{seed}:{row['id']}".encode()).digest()
        uniform = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)
        return -math.log(uniform) / weight, row["id"]

    docs.sort(key=priority)
    ordered_fingerprint = hashlib.sha256("\n".join(
        f"{row['id']}:{row.get('sha256') or ''}" for row in docs).encode()).hexdigest()
    return docs, {
        "probe_fingerprint": probe_fingerprint,
        "ordered_documents_sha256": ordered_fingerprint,
        "eligible_documents": len(docs),
        "topics": len(topic_mass),
        "sources": len(source_mass),
        "holdout_ids_excluded": excluded_ids,
        "holdout_duplicates_excluded": excluded_duplicates,
        "manifest_snapshot_sha256": manifest_snapshot.hexdigest(),
    }


def load_or_create_plan(root: Path, recipe: dict) -> tuple[list[dict], dict]:
    """Freeze document membership/order once so a growing sibling corpus cannot drift."""
    name = str(recipe["stream_plan"])
    if not name or any(part in {"", ".", ".."} for part in Path(name).parts):
        raise SystemExit(f"invalid stream_plan name: {name!r}")
    plan_dir = datakit.data_root(EXP_DIR) / "plans" / name
    documents_path = plan_dir / "documents.jsonl"
    provenance_path = plan_dir / "provenance.json"
    expected = {
        "schema_version": 1,
        "name": name,
        "ordering": ORDERING_VERSION,
        "selection_seed": int(recipe["selection_seed"]),
        "min_source_chars": int(recipe["min_source_chars"]),
        "probe_fingerprint": probe_holdout()[1],
    }
    if documents_path.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        drift = {key: (provenance.get(key), value) for key, value in expected.items()
                 if provenance.get(key) != value}
        if drift:
            raise SystemExit(
                f"stream plan {name!r} conflicts with the active recipe: {drift}; "
                "choose a new stream_plan name for a new campaign")
        rows = [json.loads(line) for line in documents_path.read_text().splitlines()
                if line.strip()]
    elif plan_dir.exists():
        raise SystemExit(f"incomplete stream plan at {plan_dir}; inspect it manually")
    else:
        selected, selection = select_documents(root, recipe)
        plan_rows = []
        for row in selected:
            plan_rows.append({
                key: row.get(key) for key in (
                    "id", "topic", "source", "license", "sha256", "text_chars")
            } | {"text_path": str(row["_text_path"].relative_to(root))})
        payload = "".join(json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n" for row in plan_rows).encode()
        provenance = {
            **expected, **selection,
            "corpus_commit_at_creation": corpus_commit(root),
            "documents_sha256": hashlib.sha256(payload).hexdigest(),
        }
        plan_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = plan_dir.with_name(
            f".{plan_dir.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        temporary.mkdir(parents=True, exist_ok=False)
        (temporary / "documents.jsonl").write_bytes(payload)
        (temporary / "provenance.json").write_text(json.dumps(
            provenance, ensure_ascii=False, indent=2, sort_keys=True))
        os.replace(temporary, plan_dir)
        rows = plan_rows

    snapshot_root = plan_dir / "snapshot"
    snapshot_provenance_path = snapshot_root / "provenance.json"
    snapshot_provenance = (
        json.loads(snapshot_provenance_path.read_text())
        if snapshot_provenance_path.is_file() else None
    )
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
    if hashlib.sha256(documents_path.read_bytes()).hexdigest() != provenance["documents_sha256"]:
        raise SystemExit(f"stream plan bytes changed: {documents_path}")
    if missing and snapshot_provenance is None:
        raise SystemExit(f"stream plan source disappeared: {root / missing[0]['text_path']}")
    if snapshot_provenance is not None:
        actual_missing = {row["id"] for row in missing}
        if actual_missing != expected_missing:
            raise SystemExit(
                "stream snapshot membership changed: "
                f"expected {len(expected_missing)} missing, found {len(actual_missing)}")
        provenance = {
            **provenance,
            "snapshot_content_manifest_sha256":
                snapshot_provenance["content_manifest_sha256"],
            "snapshot_size_bytes": snapshot_provenance["size_bytes"],
            "snapshot_documents": len(docs),
            "snapshot_missing_documents": len(missing),
            "snapshot_missing_documents_sha256": hashlib.sha256(
                "\n".join(sorted(actual_missing)).encode()).hexdigest(),
        }
    return docs, provenance


def stream_rows(
    docs: list[dict], tokenizer, recipe: dict, stats: dict,
    *, prefix_path: Path | None = None, prefix_content_tokens: int | None = None,
):
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

    def account(row: dict) -> None:
        nonlocal total, chunks, characters
        meta = row["meta"]
        n_tokens = int(meta["content_tokens"])
        doc_id = meta["doc_id"]
        topic = meta.get("topic") or "unknown"
        source = meta.get("source") or "unknown"
        used_docs.add(doc_id)
        total += n_tokens
        chunks += 1
        characters += len(row["text"])
        topic_tokens[topic] += n_tokens
        source_tokens[source] += n_tokens

    if prefix_path is not None:
        with prefix_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
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
                account(row)
                yield row
        if total != prefix_content_tokens:
            raise RuntimeError(
                f"anchor content token drift: {total:,} != {prefix_content_tokens:,}")

    resume_doc_index = None
    if resume_doc_id is not None:
        resume_doc_index = next(
            (i for i, doc in enumerate(docs) if doc["id"] == resume_doc_id), None)
        if resume_doc_index is None:
            raise RuntimeError(
                f"anchor resume document is absent from snapshot: {resume_doc_id}")

    for round_index, start in enumerate(range(0, document_limit, slice_limit)):
        if resume_round is not None and round_index < resume_round:
            continue
        stop = min(start + slice_limit, document_limit)
        for doc_index, doc in enumerate(docs):
            if (resume_round is not None and round_index == resume_round
                    and doc_index < resume_doc_index):
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
            if (resume_round is not None and round_index == resume_round
                    and doc_index == resume_doc_index):
                first_token = max(first_token, resume_token_start)
            for token_start in range(first_token, end, chunk_limit):
                token_end = min(token_start + chunk_limit, end)
                n_tokens = token_end - token_start
                key = (doc["id"], round_index, token_start)
                if key in prefix_keys:
                    continue
                # Keep row boundaries stable across the entire scale ladder. The target
                # is therefore a strict upper bound, normally short by < one chunk.
                if total + n_tokens > target:
                    stats.update({
                        "documents": len(used_docs), "chunks": chunks,
                        "content_tokens": total, "target_content_tokens": target,
                        "characters": characters, "rounds_entered": round_index + 1,
                        "topic_content_tokens": dict(sorted(topic_tokens.items())),
                        "source_content_tokens": dict(sorted(source_tokens.items())),
                    })
                    return
                lo = offsets[token_start][0]
                hi = offsets[token_end][0] if token_end < len(offsets) else offsets[-1][1]
                piece = text[lo:hi]
                if not piece.strip():
                    continue
                topic = doc.get("topic") or "unknown"
                source = doc.get("source") or "unknown"
                yield {
                    "text": piece,
                    "meta": {
                        "doc_id": doc["id"], "topic": topic, "source": source,
                        "license": doc.get("license"), "round": round_index,
                        "document_token_start": token_start,
                        "content_tokens": n_tokens,
                    },
                }
                account({
                    "text": piece,
                    "meta": {
                        "doc_id": doc["id"], "topic": topic, "source": source,
                        "content_tokens": n_tokens,
                    },
                })

    stats.update({
        "documents": len(used_docs), "chunks": chunks, "content_tokens": total,
        "target_content_tokens": target, "characters": characters,
        "rounds_entered": math.ceil(document_limit / slice_limit),
        "topic_content_tokens": dict(sorted(topic_tokens.items())),
        "source_content_tokens": dict(sorted(source_tokens.items())),
    })
    if total < target:
        raise RuntimeError(
            f"corpus capacity {total:,} tokens is below target {target:,}; "
            "lower the target or increase max_document_content_tokens")


def main(argv=None) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=CONFIG,
        help="CPT config whose data.corpus recipe should be built",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    recipe = cfg["data"]["corpus"]
    corpus_root = WORKSPACE.resolve_input(recipe["path"])
    manifest_paths = sorted((corpus_root / "manifest").glob("*.jsonl"))
    if not manifest_paths or not (corpus_root / "text").is_dir():
        raise SystemExit(
            f"corpus is incomplete at {corpus_root}: need manifest/ and local text/")

    docs, plan = load_or_create_plan(corpus_root, recipe)
    tokenizer_path = snapshot_download(TOKENIZER, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    stats = {
        "purpose": "full_parameter_cpt_data_scaling",
        **{key: plan[key] for key in (
            "eligible_documents", "topics", "sources", "holdout_ids_excluded",
            "holdout_duplicates_excluded", "manifest_snapshot_sha256",
        )},
        "stream_plan": plan["name"],
        "stream_plan_sha256": plan["documents_sha256"],
    }
    target = int(recipe["target_content_tokens"])
    prefix_path = None
    prefix_content_tokens = None
    anchor = recipe.get("extension_anchor")
    if anchor is not None and target > int(anchor["content_tokens"]):
        anchor_dataset_id = str(anchor["dataset_id"])
        anchor_content_sha256 = str(anchor["content_sha256"])
        prefix_content_tokens = int(anchor["content_tokens"])
        if "snapshot_content_manifest_sha256" not in plan:
            raise SystemExit(
                "anchored extension requires a sealed stream-plan snapshot")
        anchor_dir = datakit.data_root(EXP_DIR) / "objects" / anchor_dataset_id
        prefix_path = datakit.data_file(anchor_dir)
        anchor_provenance = datakit.provenance(anchor_dir)
        if anchor_provenance.get("dataset_id") != anchor_dataset_id:
            raise SystemExit(
                f"anchor dataset identity drift at {anchor_dir}")
        if anchor_provenance.get("content_sha256") != anchor_content_sha256:
            raise SystemExit(
                f"anchor dataset content drift: "
                f"{anchor_provenance.get('content_sha256')} != {anchor_content_sha256}")
        anchor_tokens = int(
            (anchor_provenance.get("stats") or {}).get("content_tokens") or 0)
        if anchor_tokens != prefix_content_tokens:
            raise SystemExit(
                f"anchor dataset token drift: {anchor_tokens:,} "
                f"!= {prefix_content_tokens:,}")
        stats.update({
            "extension_anchor_dataset_id": anchor_dataset_id,
            "extension_anchor_content_sha256": anchor_content_sha256,
            "extension_anchor_content_tokens": prefix_content_tokens,
            "snapshot_documents": plan["snapshot_documents"],
            "snapshot_missing_documents": plan["snapshot_missing_documents"],
            "snapshot_content_manifest_sha256":
                plan["snapshot_content_manifest_sha256"],
        })
    spec = {
        "kind": "cpt_scale",
        "source": "nekaise-corpus",
        "corpus_commit": plan["corpus_commit_at_creation"],
        "manifest_sha256": plan["manifest_snapshot_sha256"],
        "tokenizer": TOKENIZER,
        "tokenizer_revision": Path(tokenizer_path).name,
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
    if prefix_path is not None:
        spec.update({
            "extension": EXTENSION_VERSION,
            "extension_anchor_dataset_id": anchor_dataset_id,
            "extension_anchor_content_sha256": anchor_content_sha256,
            "extension_anchor_content_tokens": prefix_content_tokens,
            "snapshot_content_manifest_sha256":
                plan["snapshot_content_manifest_sha256"],
            "snapshot_documents": plan["snapshot_documents"],
            "snapshot_missing_documents": plan["snapshot_missing_documents"],
            "snapshot_missing_documents_sha256":
                plan["snapshot_missing_documents_sha256"],
        })
    if datakit.exists(EXP_DIR, spec):
        artifact = datakit.activate(EXP_DIR, spec)
        print(f"[build] cache hit -> {artifact}")
        return

    artifact = datakit.write(
        EXP_DIR, spec,
        stream_rows(
            docs, tokenizer, recipe, stats, prefix_path=prefix_path,
            prefix_content_tokens=prefix_content_tokens,
        ),
        stats=stats,
        recipe_path=__file__,
    )
    print(
        f"[build] {stats['documents']:,} docs -> {stats['chunks']:,} chunks, "
        f"{stats['content_tokens']:,}/{stats['target_content_tokens']:,} content tokens "
        f"across {len(stats['topic_content_tokens'])} topics -> {artifact}")


if __name__ == "__main__":
    main()
