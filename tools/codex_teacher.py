#!/usr/bin/env python3
"""Codex-CLI teacher for the CoAPT-CPT branch.

The author and gate are intentionally separate commands.  Both use ChatGPT-authenticated
``codex exec`` with a strict JSON schema, so no OpenAI API key is required.  Outputs are
checkpointed after every batch and resume by immutable row id.

    python tools/codex_teacher.py author --model gpt-5.6-terra \
        --in cpt_drafts.jsonl --out cpt_teacher_authored.jsonl
    python tools/codex_teacher.py gate --model gpt-5.6-terra \
        --in cpt_teacher_authored.jsonl --out cpt_teacher.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


AUTHOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rows"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "student_errors", "teacher_text"],
                "properties": {
                    "id": {"type": "string"},
                    "student_errors": {"type": "array", "items": {"type": "string"}},
                    "teacher_text": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

GATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rows"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "passed", "reason", "unsupported_claims"],
                "properties": {
                    "id": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "unsupported_claims": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
            },
        },
    },
}

AUTHOR_INSTRUCTIONS = """You are the CoAPT textbook editor. For every input row:
- Treat source_chunk as the ONLY source of truth. Never add outside facts.
- Use student_draft only to identify errors, omissions, and confused terminology.
- An empty student_draft is a valid student failure. Author from source_chunk and record
  that the student produced no draft among the student errors.
- Write continuous textbook prose, with no headings, bullets, or meta commentary.
- Preserve useful correct material, repair errors, and cover the important missed facts.
- Keep numbers, units, standards, and names exactly supported by source_chunk.
- Aim for 0.75x to 1.25x the source_chunk length so the result is substantive.
Return exactly one result for every id, in the same order, under the required schema.
"""

GATE_INSTRUCTIONS = """You are a strict source-grounding gate, separate from authorship.
For every input row, compare teacher_text to source_chunk claim by claim. A single number,
name, causal relation, definition, or factual assertion not supported by source_chunk makes
passed=false. Paraphrase is allowed; outside knowledge is not. List unsupported claims
concisely. Return exactly one verdict for every id, in the same order.
"""


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"empty input: {path}")
    ids = [str(row.get("id") or "") for row in rows]
    if any(not row_id for row_id in ids):
        raise SystemExit(f"input has a row without id: {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"input has duplicate ids: {path}")
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body)
    os.replace(temporary, path)


def batches(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def validate_response(payload: dict, expected_ids: list[str]) -> list[dict]:
    returned = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(returned, list):
        raise ValueError("Codex response has no rows array")
    returned_ids = [str(row.get("id") or "") for row in returned]
    if returned_ids != expected_ids:
        raise ValueError(
            f"Codex response ids/order mismatch: expected {expected_ids}, got {returned_ids}")
    return returned


def run_codex(*, model: str, reasoning: str, schema: dict, instructions: str,
              rows: list[dict], timeout: int) -> list[dict]:
    codex = shutil.which("codex")
    if not codex:
        raise SystemExit("codex CLI not found on PATH")
    prompt = instructions + "\nINPUT_ROWS:\n" + json.dumps(
        rows, ensure_ascii=False, separators=(",", ":"))
    with tempfile.TemporaryDirectory(prefix="nekaise-codex-teacher-") as temp:
        root = Path(temp)
        schema_path = root / "schema.json"
        output_path = root / "output.json"
        schema_path.write_text(json.dumps(schema, sort_keys=True))
        command = [
            codex, "exec", "--ephemeral", "--ignore-user-config",
            "--skip-git-repo-check", "-C", temp, "-s", "read-only",
            "-m", model, "-c", f'model_reasoning_effort="{reasoning}"',
            "--output-schema", str(schema_path), "--output-last-message", str(output_path),
            "--color", "never", "-",
        ]
        error = ""
        for attempt in range(2):
            output_path.unlink(missing_ok=True)
            completed = subprocess.run(
                command, input=prompt, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout)
            tail = "\n".join(completed.stdout.splitlines()[-30:])
            if completed.returncode:
                error = f"codex exec failed ({completed.returncode}):\n{tail}"
                continue
            try:
                payload = json.loads(output_path.read_text())
                return validate_response(payload, [row["id"] for row in rows])
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                error = f"invalid structured Codex output: {exc}\n{tail}"
        if len(rows) > 1:
            midpoint = len(rows) // 2
            return run_codex(
                model=model, reasoning=reasoning, schema=schema,
                instructions=instructions, rows=rows[:midpoint], timeout=timeout,
            ) + run_codex(
                model=model, reasoning=reasoning, schema=schema,
                instructions=instructions, rows=rows[midpoint:], timeout=timeout,
            )
        raise RuntimeError(f"Codex row failed after 2 attempts: {error}")


def author_payload(row: dict) -> dict:
    required = ("source_chunk",)
    missing = [field for field in required if not str(row.get(field) or "").strip()]
    if missing:
        raise SystemExit(f"draft row {row['id']} missing {missing}")
    return {
        "id": row["id"], "source_chunk": row["source_chunk"],
        "student_draft": str(row.get("draft") or ""),
    }


def gate_payload(row: dict) -> dict:
    required = ("source_chunk", "teacher_text")
    missing = [field for field in required if not str(row.get(field) or "").strip()]
    if missing:
        raise SystemExit(f"teacher row {row['id']} missing {missing}")
    return {
        "id": row["id"], "source_chunk": row["source_chunk"],
        "teacher_text": row["teacher_text"],
    }


def authored_is_current(output: dict, source: dict, *, model: str, reasoning: str) -> bool:
    return all((
        output.get("source_chunk") == source.get("source_chunk"),
        output.get("source_sha256") == source.get("source_sha256"),
        output.get("student_draft") == source.get("draft"),
        output.get("teacher_model") == model,
        output.get("teacher_reasoning") == reasoning,
    ))


def gate_is_current(output: dict, source: dict, *, model: str, reasoning: str) -> bool:
    gate = output.get("gate") or {}
    return all((
        output.get("source_chunk") == source.get("source_chunk"),
        output.get("source_sha256") == source.get("source_sha256"),
        output.get("teacher_text") == source.get("teacher_text"),
        gate.get("model") == model,
        gate.get("reasoning") == reasoning,
    ))


def cmd_sample(args) -> None:
    inputs = read_jsonl(args.input)
    ranked = sorted(inputs, key=lambda row: hashlib.sha256(
        f"{args.seed}:{row['id']}".encode()).digest())
    selected = ranked[:args.size]
    if len(selected) < args.size:
        raise SystemExit(f"requested {args.size} rows but input has only {len(inputs)}")
    write_jsonl_atomic(args.output, selected)
    print(f"[sample] {len(selected)}/{len(inputs)} seed={args.seed} -> {args.output}")


def cmd_shard(args) -> None:
    inputs = read_jsonl(args.input)
    if args.shards <= 0 or not 0 <= args.shard_index < args.shards:
        raise SystemExit("require --shards > 0 and 0 <= --shard-index < --shards")
    selected = inputs[args.shard_index::args.shards]
    write_jsonl_atomic(args.output, selected)
    print(f"[shard] {args.shard_index}/{args.shards}: {len(selected)} -> {args.output}")


def cmd_merge(args) -> None:
    reference = read_jsonl(args.reference)
    merged: dict[str, dict] = {}
    for path in args.inputs:
        for row in read_jsonl(path):
            if row["id"] in merged:
                raise SystemExit(f"duplicate merged id {row['id']} from {path}")
            merged[row["id"]] = row
    reference_ids = [row["id"] for row in reference]
    missing = [row_id for row_id in reference_ids if row_id not in merged]
    extra = sorted(set(merged) - set(reference_ids))
    if missing or extra:
        raise SystemExit(
            f"merge/reference mismatch: {len(missing)} missing, {len(extra)} extra")
    write_jsonl_atomic(args.output, [merged[row_id] for row_id in reference_ids])
    print(f"[merge] {len(reference_ids)} rows from {len(args.inputs)} files -> {args.output}")


def cmd_author(args) -> None:
    inputs = read_jsonl(args.input)
    existing = read_jsonl(args.output) if args.output.is_file() else []
    source_by_id = {row["id"]: row for row in inputs}
    current = {
        row["id"]: row for row in existing
        if row["id"] in source_by_id and authored_is_current(
            row, source_by_id[row["id"]], model=args.model, reasoning=args.reasoning)
    }
    done = set(current)
    pending = [row for row in inputs if row["id"] not in done]
    by_id = {row["id"]: row for row in inputs}
    authored = current
    for batch in batches(pending, args.batch_size):
        payloads = [author_payload(row) for row in batch]
        results = run_codex(
            model=args.model, reasoning=args.reasoning, schema=AUTHOR_SCHEMA,
            instructions=AUTHOR_INSTRUCTIONS, rows=payloads, timeout=args.timeout)
        for source, result in zip(batch, results):
            authored[source["id"]] = {
                "id": source["id"], "doc_id": source["doc_id"],
                "topic": source.get("topic"), "probe_type": source.get("probe_type"),
                "chunk_index": source.get("chunk_index"),
                "source_chunk": source["source_chunk"], "student_draft": source["draft"],
                "student_errors": result["student_errors"],
                "teacher_text": result["teacher_text"].strip(),
                "source_sha256": source.get("source_sha256"),
                "cleaner_version": source.get("cleaner_version"),
                "teacher_model": args.model, "teacher_reasoning": args.reasoning,
            }
        ordered = [authored[row_id] for row_id in by_id if row_id in authored]
        write_jsonl_atomic(args.output, ordered)
        print(f"[author] {len(ordered)}/{len(inputs)} -> {args.output}", flush=True)


def cmd_gate(args) -> None:
    inputs = read_jsonl(args.input)
    existing = read_jsonl(args.output) if args.output.is_file() else []
    source_by_id = {row["id"]: row for row in inputs}
    current = {
        row["id"]: row for row in existing
        if row["id"] in source_by_id and gate_is_current(
            row, source_by_id[row["id"]], model=args.model, reasoning=args.reasoning)
    }
    done = set(current)
    pending = []
    by_id = {row["id"]: row for row in inputs}
    gated = current
    for row in inputs:
        if row["id"] in done:
            continue
        if not str(row.get("teacher_text") or "").strip():
            gated[row["id"]] = {
                **row,
                "gate": {
                    "passed": False, "reason": "empty teacher_text",
                    "unsupported_claims": [], "model": args.model,
                    "reasoning": args.reasoning,
                },
            }
        else:
            pending.append(row)
    for batch in batches(pending, args.batch_size):
        payloads = [gate_payload(row) for row in batch]
        results = run_codex(
            model=args.model, reasoning=args.reasoning, schema=GATE_SCHEMA,
            instructions=GATE_INSTRUCTIONS, rows=payloads, timeout=args.timeout)
        for source, result in zip(batch, results):
            gated[source["id"]] = {
                **source,
                "gate": {
                    "passed": result["passed"], "reason": result["reason"],
                    "unsupported_claims": result["unsupported_claims"],
                    "model": args.model, "reasoning": args.reasoning,
                },
            }
        ordered = [gated[row_id] for row_id in by_id if row_id in gated]
        write_jsonl_atomic(args.output, ordered)
        print(f"[gate] {len(ordered)}/{len(inputs)} -> {args.output}", flush=True)
    if not pending and len(gated) == len(inputs):
        ordered = [gated[row_id] for row_id in by_id]
        write_jsonl_atomic(args.output, ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sample", "shard", "merge", "author", "gate"))
    parser.add_argument("--model")
    parser.add_argument("--reasoning",
                        choices=("none", "low", "medium", "high", "xhigh", "max"),
                        default="low")
    parser.add_argument("--in", dest="input", type=Path)
    parser.add_argument("--out", dest="output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.mode in {"sample", "shard", "author", "gate"} and not args.input:
        raise SystemExit(f"{args.mode} requires --in")
    if args.mode == "merge" and (not args.inputs or not args.reference):
        raise SystemExit("merge requires --inputs and --reference")
    if args.mode == "sample":
        cmd_sample(args)
    elif args.mode == "shard":
        cmd_shard(args)
    elif args.mode == "merge":
        cmd_merge(args)
    elif not args.model:
        raise SystemExit(f"{args.mode} requires --model")
    elif args.mode == "author":
        cmd_author(args)
    else:
        cmd_gate(args)


if __name__ == "__main__":
    main()
