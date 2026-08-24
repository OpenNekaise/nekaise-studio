#!/usr/bin/env python3
"""CoAPT round toolkit — the deterministic half of one nekaise-coapt round.

The agent (skill `coapt-round`) is the orchestrator and teacher; this script owns
everything that must be reproducible byte-for-byte. All subcommands read the same
config (default `configs/coapt.yaml`) and write under the active workspace:

    python experiments/coapt/build_data.py init-pool          # R0, once: freeze the pool
    python experiments/coapt/build_data.py emit-docs          # cleaned pool docs for the round
    python experiments/coapt/build_data.py select-frontier \
        --nll <nll.jsonl> --probe-records <records.jsonl> [--previous <frontier.jsonl>]
    python experiments/coapt/build_data.py make-drafts        # frozen student draft prompts
    python experiments/coapt/build_data.py build              # mix the immutable round dataset

The mix is token-ledgered: gated teacher/QA volume sets the round total so achieved
shares match `data.mix` exactly; raw and anchor streams fill their shares from the
corpus. Transfer/frozen probe documents are never visible to any stream. Corpus text is
the source of truth end to end — this script never generates or edits prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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
EXP_DIR = WORKSPACE.experiment_dir("coapt")

CONFIG = REPO / "configs" / "coapt.yaml"
TOKENIZER = "openbmb/MiniCPM5-1B-Base"
PROBE_PROVENANCE = REPO / "gym" / "tasks" / "corpus_probes" / "provenance.json"
PROBES_JSONL = REPO / "gym" / "tasks" / "corpus_probes" / "probes.jsonl"

# Frozen across rounds. The QA text the student trains on uses the same words as the
# closed-book template in tools/student.py (`Question: ...\nAnswer:`).
QA_TEMPLATE = "Question: {question}\nAnswer: {answer}"
DRAFT_TEMPLATES_VERSION = "coapt-draft-v1"
SUMMARY_CUE = "\n\nIn short,"
STREAMS = ("raw", "teacher_cpt", "qa_text", "anchor")
# Part of every dataset spec: ANY edit to this recipe invalidates the spec cache, so a
# changed builder can never silently cache-hit data built by older code. Content
# addressing still dedupes identical bytes.
RECIPE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ---------------------------------------------------------------- shared helpers

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"empty input: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_uniform(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)


def scratch_path(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else WORKSPACE.scratch_dir / path


def round_dir(cfg: dict) -> Path:
    return scratch_path(cfg["data"]["round_dir"])


def corpus_root(cfg: dict) -> Path:
    return WORKSPACE.resolve_input(cfg["data"]["corpus"]["path"])


def corpus_text_path(root: Path, row: dict) -> Path:
    """Resolve the cleaned corpus text, with a legacy-only raw-text fallback.

    A manifest that declares ``corpus_path`` has completed the cleaning stage.  Falling
    back to ``text_path`` when that declared artifact is missing would silently train on
    a different representation, so that case is a hard error.  Older manifests without
    ``corpus_path`` remain readable through their extracted-text path.
    """
    doc_id = str(row.get("id") or "")
    cleaned = row.get("corpus_path")
    if cleaned:
        path = root / str(cleaned)
        if not path.is_file():
            raise SystemExit(f"cleaned corpus text missing for {doc_id}: {path}")
        expected = row.get("corpus_sha256")
        if expected and sha256_file(path) != expected:
            raise SystemExit(f"cleaned corpus hash mismatch for {doc_id}: {path}")
        return path
    path = root / str(row.get("text_path") or f"text/{doc_id}.md")
    if not path.is_file():
        raise SystemExit(f"corpus text missing for {doc_id}: {path}")
    expected = row.get("text_sha256")
    if expected and sha256_file(path) != expected:
        raise SystemExit(f"corpus text hash mismatch for {doc_id}: {path}")
    return path


def probe_holdout() -> tuple[set[str], str]:
    """Transfer documents (dev ∪ frozen) — never visible to any training stream."""
    provenance = json.loads(PROBE_PROVENANCE.read_text())
    splits = provenance["doc_splits"]
    return set(splits["dev"]) | set(splits["frozen"]), provenance["fingerprint"]


def dev_absorption_probes() -> dict[str, dict]:
    """doc_id -> {topic, dev_probes} for absorption-kind dev-split probes."""
    out: dict[str, dict] = {}
    with PROBES_JSONL.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            probe = json.loads(line)
            if probe["kind"] != "absorption" or probe["split"] != "dev":
                continue
            entry = out.setdefault(
                probe["doc_id"], {"topic": probe["topic"], "dev_probes": 0})
            entry["dev_probes"] += 1
    return out


def manifest_index(root: Path) -> dict[str, dict]:
    return {row["id"]: row for row in corpusprep.load_manifest(root) if row.get("id")}


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def chunk_doc_tokens(tokenizer, text: str, *, chunk_limit: int,
                     doc_limit: int) -> list[tuple[str, int]]:
    """Offset-exact token chunks of a cleaned document (first doc_limit tokens)."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=True, max_length=doc_limit)
    offsets = encoded["offset_mapping"]
    chunks: list[tuple[str, int]] = []
    for start in range(0, len(offsets), chunk_limit):
        end = min(start + chunk_limit, len(offsets))
        lo = offsets[start][0]
        hi = offsets[end][0] if end < len(offsets) else offsets[-1][1]
        piece = text[lo:hi]
        if piece.strip():
            chunks.append((piece, end - start))
    return chunks


def load_tokenizer() -> tuple[object, str]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(TOKENIZER, local_files_only=True)
    return AutoTokenizer.from_pretrained(path, local_files_only=True), Path(path).name


# ---------------------------------------------------------------- init-pool

def cmd_init_pool(cfg: dict, args) -> None:
    pool_cfg = cfg["data"]["pool"]
    path = scratch_path(pool_cfg["path"])
    if path.exists():
        raise SystemExit(
            f"pool already frozen at {path} — the pool is fixed for the whole campaign; "
            "start a new campaign (new pool path) instead of regenerating it")
    holdout_ids, fingerprint = probe_holdout()
    probes = dev_absorption_probes()
    manifest = manifest_index(corpus_root(cfg))
    min_probes = int(pool_cfg.get("min_probes", 3))
    seed = int(pool_cfg.get("seed", 3407))
    size = int(pool_cfg.get("size", 150))

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for doc_id, info in sorted(probes.items()):
        if info["dev_probes"] < min_probes:
            continue
        if doc_id in holdout_ids:
            raise SystemExit(f"absorption doc {doc_id} is in the transfer holdout")
        row = manifest.get(doc_id)
        if not row or row.get("status") != "ok":
            continue
        corpus_text_path(corpus_root(cfg), row)
        by_topic[info["topic"]].append(
            {"doc_id": doc_id, "topic": info["topic"], "dev_probes": info["dev_probes"]})

    for topic in by_topic:
        by_topic[topic].sort(key=lambda r: stable_uniform(seed, r["doc_id"]))
    pool: list[dict] = []
    queues = {topic: list(rows) for topic, rows in sorted(by_topic.items())}
    while len(pool) < size and any(queues.values()):
        for topic in sorted(queues):
            if queues[topic] and len(pool) < size:
                pool.append(queues[topic].pop(0))
    if len(pool) < size:
        print(f"[pool] corpus supports only {len(pool)} eligible docs (< {size})")
    write_jsonl(path, pool)
    topics = sorted({row["topic"] for row in pool})
    print(f"[pool] froze {len(pool)} docs over {len(topics)} topics -> {path}")
    print("POOL_RESULT " + json.dumps({
        "path": str(path), "docs": len(pool), "topics": topics,
        "min_probes": min_probes, "seed": seed,
        "probe_fingerprint": fingerprint, "sha256": sha256_file(path),
    }, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------- emit-docs

def cmd_emit_docs(cfg: dict, args) -> None:
    pool = read_jsonl(scratch_path(cfg["data"]["pool"]["path"]))
    manifest = manifest_index(corpus_root(cfg))
    min_clean = int(cfg["data"].get("min_clean_chars", 1500))
    rows, thin = [], []
    for entry in pool:
        doc_id = entry["doc_id"]
        row = manifest.get(doc_id)
        if row is None:
            raise SystemExit(f"pool doc {doc_id} vanished from the corpus manifest")
        text_path = corpus_text_path(corpus_root(cfg), row)
        text = corpusprep.clean_body(text_path.read_text(errors="replace"))
        if len(text) < min_clean:
            thin.append(doc_id)
            continue
        rows.append({
            "id": doc_id, "topic": entry["topic"], "chars": len(text), "text": text,
            "text_source": "corpus_path" if row.get("corpus_path") else "text_path",
            "source_sha256": row.get("corpus_sha256") or sha256_file(text_path),
            "cleaner_version": row.get("cleaner_version"),
        })
    out = round_dir(cfg) / "docs.jsonl"
    write_jsonl(out, rows)
    print(f"[docs] {len(rows)} pool docs -> {out}" +
          (f" ({len(thin)} thin docs skipped: {thin[:5]}...)" if thin else ""))


# ---------------------------------------------------------------- select-frontier

def cmd_select_frontier(cfg: dict, args) -> None:
    pool = read_jsonl(scratch_path(cfg["data"]["pool"]["path"]))
    nll_by_doc = {row["id"]: row for row in read_jsonl(Path(args.nll))}
    accuracy: dict[str, list[int]] = defaultdict(list)
    for record in read_jsonl(Path(args.probe_records)):
        if record.get("track") == "absorption" and record.get("doc_id"):
            accuracy[record["doc_id"]].append(int(record["correct"]))

    scored = []
    for entry in pool:
        doc_id = entry["doc_id"]
        marks = accuracy.get(doc_id)
        if not marks:
            raise SystemExit(
                f"pool doc {doc_id} has no probe records in {args.probe_records}; "
                "run eval_probes with --doc-ids over the full pool first")
        nll_row = nll_by_doc.get(doc_id)
        if not nll_row or nll_row.get("nll") is None:
            raise SystemExit(f"pool doc {doc_id} has no NLL in {args.nll}")
        scored.append({
            "doc_id": doc_id, "topic": entry["topic"],
            "accuracy": round(sum(marks) / len(marks), 4), "probes": len(marks),
            "nll": float(nll_row["nll"]),
        })

    # The frontier: least-absorbed first (lowest probe accuracy), most-foreign first
    # within a tie (highest NLL). Deterministic; the rule is frozen across rounds.
    scored.sort(key=lambda r: (r["accuracy"], -r["nll"], r["doc_id"]))
    frontier = [{**row, "rank": i} for i, row in enumerate(
        scored[:int(cfg["data"].get("frontier_docs", 48))])]
    out = round_dir(cfg) / "frontier.jsonl"
    write_jsonl(out, frontier)

    turnover = None
    if args.previous:
        previous_ids = {row["doc_id"] for row in read_jsonl(Path(args.previous))}
        fresh = [row["doc_id"] for row in frontier if row["doc_id"] not in previous_ids]
        turnover = round(len(fresh) / len(frontier), 4) if frontier else None
    print(f"[frontier] {len(frontier)}/{len(scored)} pool docs -> {out}")
    print("FRONTIER_RESULT " + json.dumps({
        "path": str(out), "docs": len(frontier),
        "mean_accuracy": round(sum(r["accuracy"] for r in frontier)
                               / len(frontier), 4) if frontier else None,
        "mean_nll": round(sum(r["nll"] for r in frontier)
                          / len(frontier), 4) if frontier else None,
        "turnover_vs_previous": turnover, "sha256": sha256_file(out),
    }, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------- make-drafts

def chunk_chars(text: str, limit: int) -> list[str]:
    """Character chunks extended to the next whitespace so words never split."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        while end < len(text) and not text[end].isspace():
            end += 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end
    return chunks


def continuation_prefix(chunk: str) -> str:
    """First half of the chunk, cut at a whitespace boundary."""
    mid = len(chunk) // 2
    while mid < len(chunk) and not chunk[mid].isspace():
        mid += 1
    return chunk[:mid].rstrip()


def stratified_chunks(text: str, limit: int, count: int | None) -> list[tuple[int, str]]:
    """Choose deterministic positions across a document instead of only its beginning."""
    chunks = chunk_chars(text, limit)
    if not chunks:
        return []
    if count is None or count >= len(chunks):
        return list(enumerate(chunks))
    if count <= 0:
        return []
    if count == 1:
        index = len(chunks) // 2
        return [(index, chunks[index])]
    indices = [round(i * (len(chunks) - 1) / (count - 1)) for i in range(count)]
    return [(index, chunks[index]) for index in indices]


def cmd_make_drafts(cfg: dict, args) -> None:
    draft_cfg = cfg["data"].get("draft", {})
    limit = int(draft_cfg.get("chunk_chars", 2400))
    configured_per_doc = draft_cfg.get("chunks_per_doc", 2)
    per_doc = None if configured_per_doc is None else int(configured_per_doc)
    probe_types = tuple(draft_cfg.get("probe_types", ("continuation", "summary")))
    invalid = set(probe_types) - {"continuation", "summary"}
    if invalid or not probe_types:
        raise SystemExit(f"invalid data.draft.probe_types: {probe_types}")
    frontier = read_jsonl(round_dir(cfg) / "frontier.jsonl")
    docs = {row["id"]: row for row in read_jsonl(round_dir(cfg) / "docs.jsonl")}
    rows = []
    for entry in frontier:
        doc = docs.get(entry["doc_id"])
        if doc is None:
            raise SystemExit(f"frontier doc {entry['doc_id']} missing from docs.jsonl")
        for ci, chunk in stratified_chunks(doc["text"], limit, per_doc):
            base = {"doc_id": doc["id"], "topic": doc["topic"], "chunk_index": ci,
                    "source_chunk": chunk, "templates": DRAFT_TEMPLATES_VERSION,
                    "source_sha256": doc.get("source_sha256"),
                    "cleaner_version": doc.get("cleaner_version")}
            if "continuation" in probe_types:
                rows.append({**base, "id": f"{doc['id']}::c{ci}::continuation",
                             "probe_type": "continuation",
                             "prompt": continuation_prefix(chunk)})
            if "summary" in probe_types:
                rows.append({**base, "id": f"{doc['id']}::c{ci}::summary",
                             "probe_type": "summary", "prompt": chunk + SUMMARY_CUE})
    out = round_dir(cfg) / "cpt_draft_prompts.jsonl"
    write_jsonl(out, rows)
    print(f"[drafts] {len(rows)} prompts from {len(frontier)} frontier docs -> {out}")


# ---------------------------------------------------------------- build

def gated_rows(path: Path, required: tuple[str, ...]) -> tuple[list[dict], int]:
    """Rows whose gate.passed is True; every row must carry an explicit gate verdict."""
    passed, failed = [], 0
    for i, row in enumerate(read_jsonl(path), 1):
        gate = row.get("gate")
        if not isinstance(gate, dict) or "passed" not in gate:
            raise SystemExit(f"{path}:{i}: row has no explicit gate verdict")
        for field in required:
            if not str(row.get(field) or "").strip():
                raise SystemExit(f"{path}:{i}: row missing required field {field!r}")
        if gate["passed"]:
            passed.append(row)
        else:
            failed += 1
    if not passed:
        raise SystemExit(f"no gate-passed rows in {path}")
    return passed, failed


def plan_mix(teacher_tokens: int, mix: dict, cap: int) -> dict:
    """Ratio-locked round volume: the gated teacher CPT volume sets the total.

    total = teacher / share_teacher; raw and anchor fill their shares from the corpus and
    the short QA pairs fill theirs by cycling (repetition is reported in the ledger).
    """
    shares = {name: float(mix[name]) for name in STREAMS}
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise SystemExit(f"data.mix shares must sum to 1.0, got {shares}")
    if teacher_tokens <= 0:
        raise SystemExit("no gated teacher CPT content tokens — run the branches first")
    total = teacher_tokens / shares["teacher_cpt"]
    if total > cap:
        raise SystemExit(
            f"planned round volume {total:,.0f} content tokens exceeds "
            f"target_content_tokens {cap:,} — raise the target or trim round artifacts")
    return {"total": int(round(total)),
            "raw": int(round(total * shares["raw"])),
            "qa_text": int(round(total * shares["qa_text"])),
            "anchor": int(round(total * shares["anchor"])),
            "teacher_cpt": teacher_tokens}


def fill_by_cycling(items: list[tuple[dict, int]], target: int,
                    max_passes: int | None = None) -> tuple[list[dict], int, int]:
    """Repeat (row, n_tokens) items in order until the strict token target is reached."""
    rows, total, passes, attempts = [], 0, 0, 0
    while total < target and (max_passes is None or attempts < max_passes):
        attempts += 1
        progressed = False
        for row, n_tokens in items:
            if total + n_tokens > target:
                continue
            rows.append({**row, "meta": {**row["meta"], "pass": attempts}})
            total += n_tokens
            progressed = True
        if not progressed:
            break
        passes += 1
    return rows, total, passes


def require_near_target(actual: int, target: int, label: str,
                        tolerance: float = 0.005) -> None:
    """Refuse a ledger whose strict row boundary undershoots a stream materially."""
    if target <= 0:
        return
    shortfall = target - actual
    if shortfall < 0 or shortfall / target > tolerance:
        raise SystemExit(
            f"{label} stream reached {actual:,}/{target:,} tokens; allowed shortfall is "
            f"{tolerance:.1%}")


def raw_stream(frontier_docs: list[dict], tokenizer, *, chunk_limit: int,
               doc_limit: int, target: int, rnd: int,
               max_passes: int | None = None) -> tuple[list[dict], int, int]:
    items: list[tuple[dict, int]] = []
    for doc in frontier_docs:
        for piece, n_tokens in chunk_doc_tokens(
                tokenizer, doc["text"], chunk_limit=chunk_limit, doc_limit=doc_limit):
            items.append(({"text": piece, "meta": {
                "stream": "raw", "doc_id": doc["id"], "round": rnd,
                "content_tokens": n_tokens,
                "source_sha256": doc.get("source_sha256")}}, n_tokens))
    if not items:
        raise SystemExit("frontier documents produced no raw chunks")
    if max_passes is not None and sum(n for _, n in items) * max_passes < target:
        raise SystemExit(
            f"raw stream has capacity for only "
            f"{sum(n for _, n in items) * max_passes:,}/{target:,} tokens with "
            f"max_stream_passes={max_passes}; enlarge the frozen frontier")
    return fill_by_cycling(items, target, max_passes=max_passes)


def source_chunk_raw_stream(prompt_rows: list[dict], tokenizer, *, target: int,
                            rnd: int, max_passes: int | None = None
                            ) -> tuple[list[dict], int, int]:
    """Raw control from the exact unique source spans used to author teacher text."""
    items: list[tuple[dict, int]] = []
    seen: set[tuple[str, object]] = set()
    for prompt in prompt_rows:
        chunk_index = prompt.get("chunk_index")
        key = (prompt["doc_id"], chunk_index if chunk_index is not None else prompt["id"])
        if key in seen:
            continue
        seen.add(key)
        text = prompt["source_chunk"]
        n_tokens = count_tokens(tokenizer, text)
        items.append(({"text": text, "meta": {
            "stream": "raw", "doc_id": prompt["doc_id"], "round": rnd,
            "chunk_index": chunk_index, "content_tokens": n_tokens,
            "source_sha256": prompt.get("source_sha256")}}, n_tokens))
    capacity = sum(n for _, n in items) * (max_passes or 1)
    if not items or (max_passes is not None and capacity < target):
        raise SystemExit(
            f"same-source raw stream has capacity for only {capacity:,}/{target:,} tokens; "
            "increase the frozen source-span coverage")
    return fill_by_cycling(items, target, max_passes=max_passes)


def anchor_stream(cfg: dict, tokenizer, pool_ids: set[str], holdout_ids: set[str],
                  *, chunk_limit: int, doc_limit: int, target: int,
                  rnd: int) -> tuple[list[dict], int, int]:
    root = corpus_root(cfg)
    seed = int(cfg["data"].get("anchor_seed", 3407))
    min_chars = int(cfg["data"].get("min_source_chars", 1500))
    min_clean = int(cfg["data"].get("min_clean_chars", 1500))
    manifest = corpusprep.load_manifest(root)
    holdout_hashes = {row.get("sha256") for row in manifest
                      if row.get("id") in holdout_ids and row.get("sha256")}
    eligible, seen_hashes = [], set()
    for row in manifest:
        doc_id = row.get("id") or ""
        if (not doc_id or doc_id in pool_ids or doc_id in holdout_ids
                or row.get("status") != "ok"
                or int(row.get("text_chars") or 0) < min_chars):
            continue
        source_hash = row.get("sha256") or ""
        if source_hash and (source_hash in holdout_hashes or source_hash in seen_hashes):
            continue
        if source_hash:
            seen_hashes.add(source_hash)
        eligible.append(row)
    eligible.sort(key=lambda row: stable_uniform(seed, row["id"]))

    rows, total, used = [], 0, 0
    for row in eligible:
        if total >= target:
            break
        text_path = corpus_text_path(root, row)
        text = corpusprep.clean_body(text_path.read_text(errors="replace"))
        if len(text) < min_clean:
            continue
        used += 1
        for piece, n_tokens in chunk_doc_tokens(
                tokenizer, text, chunk_limit=chunk_limit, doc_limit=doc_limit):
            if total + n_tokens > target:
                return rows, total, used
            rows.append({"text": piece, "meta": {
                "stream": "anchor", "doc_id": row["id"],
                "topic": row.get("topic") or "unknown", "round": rnd,
                "content_tokens": n_tokens,
                "source_sha256": row.get("corpus_sha256") or sha256_file(text_path)}})
            total += n_tokens
    if total < target:
        raise SystemExit(
            f"anchor capacity {total:,} tokens is below its share target {target:,}")
    return rows, total, used


def cmd_build(cfg: dict, args) -> None:
    data = cfg["data"]
    rnd = int(data["round"])
    rdir = round_dir(cfg)
    pool_path = scratch_path(data["pool"]["path"])
    pool_ids = {row["doc_id"] for row in read_jsonl(pool_path)}
    frontier_path = rdir / "frontier.jsonl"
    frontier_ids = {row["doc_id"] for row in read_jsonl(frontier_path)}
    holdout_ids, probe_fingerprint = probe_holdout()
    if pool_ids & holdout_ids:
        raise SystemExit("pool intersects the transfer holdout — refuse to build")

    tokenizer, tokenizer_revision = load_tokenizer()
    chunk_limit = int(data.get("max_content_tokens", 1800))
    doc_limit = int(data.get("raw_doc_tokens", 8192))

    control = data.get("control")
    if control:
        # SPEC §3 null arm: the same frontier documents re-read raw, at the exact token
        # volume the CoAPT arm spent — no teacher text, no QA, same anchor share.
        matched_ledger_path = None
        if control.get("match_ledger"):
            matched_ledger_path = scratch_path(control["match_ledger"])
            matched_ledger = json.loads(matched_ledger_path.read_text())
            total = int(matched_ledger["content_tokens"])
        else:
            total = int(control["total_content_tokens"])
        max_passes = data.get("max_stream_passes")
        max_passes = None if max_passes is None else int(max_passes)
        shares = {name: float(data["mix"][name]) for name in STREAMS}
        raw_target = int(round(total * (shares["raw"] + shares["teacher_cpt"]
                                        + shares["qa_text"])))
        anchor_target = total - raw_target
        prompt_path = rdir / "cpt_draft_prompts.jsonl"
        teacher_control_path = rdir / "cpt_teacher.jsonl"
        control_source_rows, _ = gated_rows(
            teacher_control_path, ("source_chunk", "doc_id"))
        raw_rows, raw_tokens, raw_passes = source_chunk_raw_stream(
            control_source_rows, tokenizer, target=raw_target, rnd=rnd,
            max_passes=max_passes)
        require_near_target(raw_tokens, raw_target, "raw control")
        anchor_rows, anchor_tokens, anchor_docs = anchor_stream(
            cfg, tokenizer, pool_ids, holdout_ids, chunk_limit=chunk_limit,
            doc_limit=doc_limit, target=anchor_target, rnd=rnd)
        require_near_target(anchor_tokens, anchor_target, "control anchor")
        all_rows = raw_rows + anchor_rows
        for row in all_rows:
            if row["meta"]["doc_id"] in holdout_ids:
                raise SystemExit(
                    f"transfer doc leaked into the mix: {row['meta']['doc_id']}")
        random.Random(int(data.get("shuffle_seed", 3407))).shuffle(all_rows)
        got = raw_tokens + anchor_tokens
        stats = {
            "purpose": "coapt_control_mix", "round": rnd, "content_tokens": got,
            "matched_total_content_tokens": total, "chunks": len(all_rows),
            "streams": {
                "raw": {"rows": len(raw_rows), "content_tokens": raw_tokens,
                        "target_content_tokens": raw_target, "passes": raw_passes},
                "anchor": {"rows": len(anchor_rows), "content_tokens": anchor_tokens,
                           "target_content_tokens": anchor_target,
                           "documents": anchor_docs},
            },
            "pool_docs": len(pool_ids), "frontier_docs": len(frontier_ids),
        }
        spec = {
            "kind": "coapt_control", "round": rnd, "source": "nekaise-corpus",
            "mixer": "equal-token-raw-repetition-v1",
            "recipe_sha256": RECIPE_SHA256,
            "tokenizer": TOKENIZER, "tokenizer_revision": tokenizer_revision,
            "probe_fingerprint": probe_fingerprint,
            "matched_total_content_tokens": total,
            "max_content_tokens": chunk_limit, "raw_doc_tokens": doc_limit,
            "anchor_seed": int(data.get("anchor_seed", 3407)),
            "shuffle_seed": int(data.get("shuffle_seed", 3407)),
            "pool_sha256": sha256_file(pool_path),
            "frontier_sha256": sha256_file(frontier_path),
            "docs_sha256": sha256_file(rdir / "docs.jsonl"),
            "draft_prompts_sha256": sha256_file(prompt_path),
            "cpt_teacher_sha256": sha256_file(teacher_control_path),
            **({"matched_ledger_sha256": sha256_file(matched_ledger_path)}
               if matched_ledger_path else {}),
            "transfer_documents_excluded": True,
        }
        if datakit.exists(EXP_DIR, spec):
            artifact = datakit.activate(EXP_DIR, spec)
            print(f"[build] cache hit -> {artifact}")
            print(f"DATASET_ID {artifact.name}")
            return
        artifact = datakit.write(EXP_DIR, spec, iter(all_rows), stats=stats,
                                 recipe_path=__file__)
        ledger_text = json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True)
        (rdir / f"token_ledger_{artifact.name}.json").write_text(ledger_text)
        (rdir / "token_ledger_control.json").write_text(ledger_text)
        print(f"[build] control round {rnd}: {len(all_rows):,} rows, {got:,} content "
              f"tokens (matched to {total:,}) -> {artifact}")
        print(f"DATASET_ID {artifact.name}")
        return

    teacher_path = rdir / "cpt_teacher.jsonl"
    sft_path = rdir / "sft_final.jsonl"
    teacher_rows, teacher_failed = gated_rows(teacher_path, ("teacher_text", "doc_id"))
    qa_share = float(data["mix"]["qa_text"])
    if qa_share > 0:
        sft_rows, sft_failed = gated_rows(sft_path, ("question", "answer", "doc_id"))
    else:
        sft_rows, sft_failed = [], 0
    for label, rows in (("cpt_teacher", teacher_rows), ("sft_final", sft_rows)):
        stray = {row["doc_id"] for row in rows} - frontier_ids
        if stray:
            raise SystemExit(f"{label} references docs outside this round's frontier: "
                             f"{sorted(stray)[:5]}")

    teacher_items: list[tuple[dict, int]] = []
    for row in teacher_rows:
        n_tokens = count_tokens(tokenizer, row["teacher_text"])
        teacher_items.append(({"text": row["teacher_text"], "meta": {
            "stream": "teacher_cpt", "doc_id": row["doc_id"], "round": rnd,
            "probe_type": row.get("probe_type"), "source_id": row.get("id"),
            "content_tokens": n_tokens,
            "source_sha256": row.get("source_sha256")}}, n_tokens))
    qa_items: list[tuple[dict, int]] = []
    for row in sft_rows:
        text = QA_TEMPLATE.format(question=row["question"].strip(),
                                  answer=row["answer"].strip())
        n_tokens = count_tokens(tokenizer, text)
        qa_items.append(({"text": text, "meta": {
            "stream": "qa_text", "doc_id": row["doc_id"], "round": rnd,
            "qtype": row.get("qtype"), "source_id": row.get("id"),
            "content_tokens": n_tokens}}, n_tokens))

    teacher_unique_tokens = sum(n for _, n in teacher_items)
    qa_unique_tokens = sum(n for _, n in qa_items)
    fixed_total = data.get("fixed_total_content_tokens")
    max_passes = data.get("max_stream_passes")
    max_passes = None if max_passes is None else int(max_passes)
    if fixed_total:
        # Share-sweep mode: total is pinned; EVERY stream fills its share by cycling.
        # Higher teacher shares mean repetition of the same gated material, not new
        # teacher knowledge — the unique volume is recorded in the ledger.
        shares = {name: float(data["mix"][name]) for name in STREAMS}
        if abs(sum(shares.values()) - 1.0) > 1e-6:
            raise SystemExit(f"data.mix shares must sum to 1.0, got {shares}")
        plan = {"total": int(fixed_total),
                **{name: int(round(int(fixed_total) * shares[name]))
                   for name in STREAMS}}
        if (max_passes is not None
                and teacher_unique_tokens * max_passes < plan["teacher_cpt"]):
            raise SystemExit(
                f"teacher stream has capacity for only "
                f"{teacher_unique_tokens * max_passes:,}/{plan['teacher_cpt']:,} "
                f"tokens with max_stream_passes={max_passes}; generate more unique "
                "gate-passed teacher rows")
        teacher_stream, teacher_tokens, teacher_passes = fill_by_cycling(
            teacher_items, plan["teacher_cpt"], max_passes=max_passes)
        require_near_target(teacher_tokens, plan["teacher_cpt"], "teacher")
    else:
        plan = plan_mix(teacher_unique_tokens, data["mix"],
                        int(data["target_content_tokens"]))
        teacher_stream = [row for row, _ in teacher_items]
        teacher_tokens, teacher_passes = teacher_unique_tokens, 1
    qa_stream, qa_tokens, qa_passes = fill_by_cycling(qa_items, plan["qa_text"])
    require_near_target(qa_tokens, plan["qa_text"], "QA")

    frontier_docs = [doc for doc in read_jsonl(rdir / "docs.jsonl")
                     if doc["id"] in frontier_ids]
    if plan["raw"] > 0:
        raw_rows, raw_tokens, raw_passes = raw_stream(
            frontier_docs, tokenizer, chunk_limit=chunk_limit, doc_limit=doc_limit,
            target=plan["raw"], rnd=rnd, max_passes=max_passes)
    else:
        raw_rows, raw_tokens, raw_passes = [], 0, 0
    if plan["anchor"] > 0:
        anchor_rows, anchor_tokens, anchor_docs = anchor_stream(
            cfg, tokenizer, pool_ids, holdout_ids, chunk_limit=chunk_limit,
            doc_limit=doc_limit, target=plan["anchor"], rnd=rnd)
        require_near_target(anchor_tokens, plan["anchor"], "anchor")
    else:
        anchor_rows, anchor_tokens, anchor_docs = [], 0, 0

    all_rows = raw_rows + teacher_stream + qa_stream + anchor_rows
    for row in all_rows:
        if row["meta"]["doc_id"] in holdout_ids:
            raise SystemExit(f"transfer doc leaked into the mix: {row['meta']['doc_id']}")
    random.Random(int(data.get("shuffle_seed", 3407))).shuffle(all_rows)

    total = raw_tokens + teacher_tokens + qa_tokens + anchor_tokens
    ledger = {
        "raw": {"rows": len(raw_rows), "content_tokens": raw_tokens,
                "target_content_tokens": plan["raw"], "passes": raw_passes},
        "teacher_cpt": {"rows": len(teacher_stream), "content_tokens": teacher_tokens,
                        "target_content_tokens": plan["teacher_cpt"]
                        if fixed_total else teacher_tokens,
                        "passes": teacher_passes,
                        "unique_content_tokens": teacher_unique_tokens,
                        "gate_failed": teacher_failed},
        "qa_text": {"rows": len(qa_stream), "content_tokens": qa_tokens,
                    "target_content_tokens": plan["qa_text"], "passes": qa_passes,
                    "unique_pairs": len(qa_items),
                    "unique_content_tokens": qa_unique_tokens,
                    "gate_failed": sft_failed},
        "anchor": {"rows": len(anchor_rows), "content_tokens": anchor_tokens,
                   "target_content_tokens": plan["anchor"], "documents": anchor_docs},
    }
    for name in STREAMS:
        ledger[name]["share_planned"] = float(data["mix"][name])
        ledger[name]["share_achieved"] = round(
            ledger[name]["content_tokens"] / total, 4) if total else None

    stats = {
        "purpose": "coapt_round_mix", "round": rnd,
        "content_tokens": total, "planned_content_tokens": plan["total"],
        "target_content_tokens": int(data["target_content_tokens"]),
        "chunks": len(all_rows), "streams": ledger,
        "pool_docs": len(pool_ids), "frontier_docs": len(frontier_ids),
    }
    spec = {
        "kind": "coapt_share_sweep" if fixed_total else "coapt_round",
        "round": rnd, "source": "nekaise-corpus",
        "mixer": "fixed-total-share-v1" if fixed_total else "ratio-locked-teacher-v1",
        "recipe_sha256": RECIPE_SHA256,
        **({"fixed_total_content_tokens": int(fixed_total)} if fixed_total else {}),
        "tokenizer": TOKENIZER, "tokenizer_revision": tokenizer_revision,
        "probe_fingerprint": probe_fingerprint,
        "mix": {name: float(data["mix"][name]) for name in STREAMS},
        "target_content_tokens": int(data["target_content_tokens"]),
        "max_content_tokens": chunk_limit, "raw_doc_tokens": doc_limit,
        "anchor_seed": int(data.get("anchor_seed", 3407)),
        "shuffle_seed": int(data.get("shuffle_seed", 3407)),
        "qa_template": QA_TEMPLATE, "draft_templates": DRAFT_TEMPLATES_VERSION,
        "pool_sha256": sha256_file(pool_path),
        "frontier_sha256": sha256_file(frontier_path),
        "docs_sha256": sha256_file(rdir / "docs.jsonl"),
        "draft_prompts_sha256": sha256_file(rdir / "cpt_draft_prompts.jsonl"),
        "cpt_teacher_sha256": sha256_file(teacher_path),
        **({"sft_final_sha256": sha256_file(sft_path)} if qa_share > 0 else {}),
        "transfer_documents_excluded": True,
    }
    if datakit.exists(EXP_DIR, spec):
        artifact = datakit.activate(EXP_DIR, spec)
        print(f"[build] cache hit -> {artifact}")
        print(f"DATASET_ID {artifact.name}")
        return
    artifact = datakit.write(EXP_DIR, spec, iter(all_rows), stats=stats,
                             recipe_path=__file__)
    ledger_text = json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True)
    (rdir / "token_ledger.json").parent.mkdir(parents=True, exist_ok=True)
    (rdir / f"token_ledger_{artifact.name}.json").write_text(ledger_text)
    (rdir / "token_ledger.json").write_text(ledger_text)
    shares = {name: ledger[name]["share_achieved"] for name in STREAMS}
    print(f"[build] round {rnd}: {len(all_rows):,} rows, {total:,} content tokens "
          f"(shares {shares}) -> {artifact}")
    print(f"DATASET_ID {artifact.name}")


# ---------------------------------------------------------------- entry

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="build",
                        choices=("init-pool", "emit-docs", "select-frontier",
                                 "make-drafts", "build"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--nll", help="select-frontier: student.py score output JSONL")
    parser.add_argument("--probe-records",
                        help="select-frontier: eval_probes pool records JSONL")
    parser.add_argument("--previous",
                        help="select-frontier: previous round's frontier.jsonl "
                             "(reports turnover)")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.resolve().read_text())
    if args.command == "select-frontier" and not (args.nll and args.probe_records):
        raise SystemExit("select-frontier requires --nll and --probe-records")
    {"init-pool": cmd_init_pool, "emit-docs": cmd_emit_docs,
     "select-frontier": cmd_select_frontier, "make-drafts": cmd_make_drafts,
     "build": cmd_build}[args.command](cfg, args)


if __name__ == "__main__":
    main()
