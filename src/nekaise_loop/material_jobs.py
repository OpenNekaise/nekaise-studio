"""Durable bounded fan-out for material authors, owned by one loop worker."""
from __future__ import annotations

import asyncio
import json
import re
import time

import httpx
from pydantic import ValidationError

from .artifacts import digest
from .failures import TeacherUnavailable
from .material_types import CandidateBatch
from .material_allowance import allowance_totals
from .processes import Cancelled, stop_owned
from .providers.material import AUTHOR_TRANSPORTS, AuthorHTTPError
from .storage import encode, now


def initialize_jobs(store):
    # The normal workspace worker lock owns dispatch. No independent scheduler
    # or lease timeout may steal a still-running request from that worker.
    with store.connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS material_jobs (
            id TEXT PRIMARY KEY, round_id TEXT NOT NULL REFERENCES rounds(id),
            plan_id TEXT NOT NULL, author_id TEXT NOT NULL, input_artifact TEXT NOT NULL,
            status TEXT NOT NULL, artifact TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_material_jobs_round ON material_jobs(round_id);
        CREATE TABLE IF NOT EXISTS material_calls (
            id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES material_jobs(id),
            stage_id INTEGER NOT NULL REFERENCES stage_runs(id), status TEXT NOT NULL,
            reserved_tokens INTEGER NOT NULL, input_chars INTEGER NOT NULL,
            artifact TEXT, usage TEXT NOT NULL DEFAULT '{}', error TEXT,
            created_at TEXT NOT NULL, finished_at TEXT, process_pid INTEGER, process_start TEXT
        );
        """)
        db.execute("BEGIN IMMEDIATE")
        columns = {r["name"] for r in db.execute("PRAGMA table_info(material_calls)")}
        for name, kind in (("process_pid", "INTEGER"), ("process_start", "TEXT")):
            if name not in columns:
                db.execute(f"ALTER TABLE material_calls ADD COLUMN {name} {kind}")


def recover_author_processes(store):
    """Only the workspace worker reconciles children left by its predecessor."""
    if not store.one("SELECT name FROM sqlite_master WHERE type='table' AND name='material_calls'"):
        return
    initialize_jobs(store)
    for call in store.query("SELECT id,job_id,process_pid,process_start FROM material_calls WHERE process_pid IS NOT NULL"):
        stop_owned(call["process_pid"], call["process_start"])
        with store.connect(immediate=True) as db:
            db.execute("UPDATE material_calls SET status=CASE WHEN status='running' THEN 'uncertain' ELSE status END,process_pid=NULL,process_start=NULL,finished_at=COALESCE(finished_at,?) WHERE id=?", (now(), call["id"]))
            db.execute("UPDATE material_jobs SET status='uncertain',error='Worker exited during author execution; usage may be unknown',updated_at=? WHERE id=? AND status='running'", (now(), call["job_id"]))


def request_body(author, spec, schema):
    # Express the existing provenance allowlists in the generation schema too.
    # Long hashes can be mistyped even after path-only retry feedback. Never
    # repair returned citations or mutate the shared schema for sibling jobs.
    schema = json.loads(json.dumps(schema))
    properties = schema["$defs"]["Candidate"]["properties"]
    for field, identifiers in (("source_keys", spec["sources"]),
                               ("seed_ids", spec["job"]["seed_ids"])):
        allowed = sorted(set(identifiers))
        if allowed:
            properties[field]["items"]["enum"] = allowed
        else:
            properties[field]["maxItems"] = 0
    instruction = (
        "You are a Material Author working for the primary teacher. Produce candidate teaching material, not student observations or evaluation scores. "
        "Follow the teacher's expansion instructions and corrected seed demonstrations. Use only the supplied source keys for citations; "
        "an empty source list means authored material: use source_keys=[], never an empty-string placeholder. Do not invent source identifiers or student attempts. "
        "Return one JSON object matching the supplied schema. All explanatory text intended for training belongs in the specified material fields. "
        "Candidate rows forbid every property not declared in the schema, including annotations such as territory. "
        "If retry_validation is present, it contains host validation paths and error types for your previous rejected response; correct those structural errors. "
        "If retry_budget is present, the previous response exceeded its output reservation. Its reported output includes reasoning and the entire JSON, "
        "including prompts and metadata, not just training answers. Keep JSON formatting and incidental metadata concise within the unchanged reservation; "
        "Preserve the teacher's content and schema fields; follow any explicit teacher permission to return fewer rows to fit the budget. "
        "Without that permission, do not reduce the requested material. "
        "unprovided_source_key and unprovided_seed_id identify citations outside the supplied source keys or job seed_ids; use only those supplied identifiers. "
        "seed_feedback is input-only primary-teacher context keyed by seed ID, not candidate fields. "
        "Never emit teacher or seed_feedback keys in candidate rows; put the answer in training_response for chat_response or training_text for text modes. "
        "Do not insert model-specific role markers or a thinking prefill: the student tokenizer will serialize accepted content."
    )
    payload = {"task": spec, "output_schema": schema}
    if author.transport == "claude_code":
        payload["execution_limits"] = {"max_response_output_tokens": spec["job"]["max_output_tokens"] // 2,
            "max_model_requests": 1, "instruction": "Return complete schema-valid JSON within the response cap; no continuation is available."}
    elif author.transport == "codex_code":
        payload["execution_limits"] = {"reserved_output_tokens": spec["job"]["max_output_tokens"],
            "provider_output_token_cap": None,
            "instruction": "Return one complete JSON object without tools. Keep reasoning plus final output within the reservation; over-budget reported output is rejected. No author-level retry or continuation is available."}
    body = {**author.options, "model": author.model,
        "messages": [{"role": "system", "content": instruction},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "max_tokens": spec["job"]["max_output_tokens"], "stream": False}
    if author.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


class CandidateValidationError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("Material author response failed completion, schema or provenance validation: "
                         + json.dumps(diagnostics, ensure_ascii=True) + "; inspect saved call artifact")


def candidates(result, spec):
    try:
        if not isinstance(result.content, str):
            raise ValueError("Author response was incomplete or did not contain final content")
        batch = CandidateBatch.model_validate_json(result.content)
        if not result.complete:
            raise ValueError("Author response was incomplete")
        source_keys = set(spec["sources"])
        seed_ids = set(spec["job"]["seed_ids"])
        diagnostics = []
        for index, row in enumerate(batch.rows):
            for field, allowed, kind in (("source_keys", source_keys, "unprovided_source_key"),
                                         ("seed_ids", seed_ids, "unprovided_seed_id")):
                for citation_index, citation in enumerate(getattr(row, field)):
                    if citation not in allowed:
                        diagnostics.append({"path": ["rows", index, field, citation_index], "type": kind})
                        if len(diagnostics) == 8:
                            raise CandidateValidationError(diagnostics)
        if diagnostics:
            raise CandidateValidationError(diagnostics)
        return batch.model_dump()["rows"]
    except CandidateValidationError:
        # Keep bounded provenance paths, never the rejected citation values.
        raise
    except ValidationError as exc:
        # Never echo values, validator messages/context, or arbitrary field text.
        diagnostics = [{"path": [p if isinstance(p, int) or re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", str(p)) else "<field>"
                                 for p in e["loc"][:12]], "type": e["type"]}
                       for e in exc.errors(include_input=False, include_context=False, include_url=False)[:8]]
        raise CandidateValidationError(diagnostics) from None
    except (KeyError, TypeError, ValueError):
        raise CandidateValidationError([{"path": [], "type": "completion_or_provenance"}]) from None


async def _execute(ctx, specs, pool, client_factory):
    store, artifacts = ctx.store, ctx.artifacts
    initialize_jobs(store)
    if len(specs) > pool.max_calls_per_round or sum(s["job"]["max_output_tokens"] for s in specs) > pool.max_output_tokens_per_round:
        raise ValueError("Teacher expansion plan exceeds the declared per-round author allowance")
    authors = {a.id: a for a in pool.authors}
    schema = CandidateBatch.model_json_schema()
    requests = []
    for spec in specs:
        author = authors[spec["job"]["author_id"]]
        if author.transport not in AUTHOR_TRANSPORTS:
            raise ValueError(f"Unsupported material-author transport: {author.transport}")
        body = request_body(author, spec, schema)
        size = len(json.dumps(body, ensure_ascii=False))
        if size > pool.max_input_chars_per_call or body["max_tokens"] > author.max_output_tokens:
            raise ValueError(f"Material job {spec['job']['id']} exceeds its request execution limits")
        if author.transport == "claude_code" and body["max_tokens"] < 256:
            raise ValueError("Claude Code jobs need at least 256 reserved output tokens")
        payload = {"version": 1, "round_id": ctx.round["id"], "author": author.model_dump(), "spec": spec, "request": body}
        key = artifacts.put(payload)
        requests.append((key, author, payload, size))
    if len(requests) > pool.max_calls_per_round or sum(p[2]["request"]["max_tokens"] for p in requests) > pool.max_output_tokens_per_round:
        raise ValueError("Teacher expansion plan exceeds the declared per-round author allowance")
    for key, author, payload, _ in requests:
        store.execute("INSERT OR IGNORE INTO material_jobs(id,round_id,plan_id,author_id,input_artifact,status,created_at,updated_at) VALUES(?,?,?,?,?,'pending',?,?)",
                      (key, ctx.round["id"], payload["spec"]["job"]["id"], author.id, key, now(), now()))
    # A previous worker/stage has exited. Never pretend an interrupted remote
    # request was unbilled, and never permit a second scheduler to claim it by TTL.
    store.execute("UPDATE material_calls SET status='uncertain',finished_at=? WHERE status='running' AND job_id IN (SELECT id FROM material_jobs WHERE round_id=?)", (now(), ctx.round["id"]))
    pending, completed = [], {}
    for item in requests:
        saved = store.one("SELECT status,artifact FROM material_jobs WHERE id=?", (item[0],))
        if saved["status"] == "complete":
            completed[item[0]] = artifacts.get(saved["artifact"])
        else:
            pending.append(item)
    since = store.campaign(ctx.campaign["id"]).get("teacher_budget_since") or ctx.campaign["created_at"]
    with store.connect() as db:
        allowance = allowance_totals(db, ctx.round["id"], since, pool)
    needed = sum(item[2]["request"]["max_tokens"] for item in pending)
    if len(pending) > allowance["remaining_calls"] or needed > allowance["remaining_output_tokens"]:
        raise TeacherUnavailable(f"Material-author remaining allowance cannot fund all unfinished jobs: {max(0, allowance['remaining_output_tokens'])} output tokens remain; {needed} required. Orchestrator must investigate and explicitly fund the retry shortfall with material_allowance. Time does not renew it; completed jobs remain reusable.", "material_budget")
    deadline = time.monotonic() + ctx.config.max_stage_seconds
    client_options = {"timeout": httpx.Timeout(65, connect=15), "follow_redirects": False, "trust_env": False,
                      "limits": httpx.Limits(max_connections=pool.concurrency, max_keepalive_connections=pool.concurrency)}
    factory = client_factory or httpx.AsyncClient
    async with factory(**client_options) as client:
        adapters = {name: adapter(client) for name, adapter in AUTHOR_TRANSPORTS.items()}

        async def dispatch(item):
            key, author, payload, size = item
            previous = store.one("SELECT artifact,usage,reserved_tokens FROM material_calls WHERE job_id=? AND status='failed' ORDER BY id DESC LIMIT 1", (key,))
            if previous and previous["artifact"]:
                failure = artifacts.get(previous["artifact"])
                diagnostics = failure.get("validation_errors")
                previous_content = {}
                if failure.get("request_artifact"):
                    # A transport failure cannot establish that earlier structural
                    # or budget errors were corrected. Carry prior feedback forward.
                    previous_request = artifacts.get(failure["request_artifact"])
                    previous_content = json.loads(previous_request["request"]["messages"][1]["content"])
                    diagnostics = diagnostics or previous_content.get("retry_validation")
                budget = previous_content.get("retry_budget")
                usage = json.loads(previous["usage"])
                reported = usage.get("completion_tokens") if isinstance(usage, dict) else None
                reserved = previous["reserved_tokens"]
                if type(reported) is int and reported > reserved:
                    budget = {"reserved_output_tokens": reserved,
                              "reported_output_tokens": reported,
                              "overrun_tokens": reported - reserved}
                if diagnostics or budget:
                    payload = json.loads(json.dumps(payload))
                    content = json.loads(payload["request"]["messages"][1]["content"])
                    if diagnostics:
                        content["retry_validation"] = diagnostics
                    if budget:
                        content["retry_budget"] = budget
                    payload["request"]["messages"][1]["content"] = json.dumps(content, ensure_ascii=False)
                    size = len(json.dumps(payload["request"], ensure_ascii=False))
                    if size > pool.max_input_chars_per_call:
                        raise ValueError("Material retry diagnostics exceed the request input limit")
            request_artifact = artifacts.put(payload)
            pending_evidence = artifacts.put({"request_artifact": request_artifact, "response": None})
            with store.connect(immediate=True) as db:
                allowance = allowance_totals(db, ctx.round["id"], since, pool)
                reservation = payload["request"]["max_tokens"]
                if allowance["remaining_calls"] < 1 or reservation > allowance["remaining_output_tokens"]:
                    raise TeacherUnavailable("Material-author allowance exhausted, including uncertain calls. Orchestrator must inspect and explicitly fund the retry; time does not renew this allowance.", "material_budget")
                call_id = db.execute("INSERT INTO material_calls(job_id,stage_id,status,reserved_tokens,input_chars,artifact,created_at) VALUES(?,?,'running',?,?,?,?)", (key, ctx.stage_id, reservation, size, pending_evidence, now())).lastrowid
                db.execute("UPDATE material_jobs SET status='running',error=NULL,updated_at=? WHERE id=?", (now(), key))
            ctx.event("material_author", f"Material author {author.label} started {payload['spec']['job']['id']}", {"job_id": key, "call_id": call_id})
            raw = None
            try:
                authored = await adapters[author.transport].generate(author, payload["request"], env_file=ctx.engine.settings.root/".env", max_bytes=pool.max_response_bytes,
                    execution={"store": store, "call_id": call_id, "claude": ctx.engine.settings.claude, "codex": ctx.engine.settings.codex,
                               "directory": ctx.directory/f"author-{call_id}"})
                raw = artifacts.put({"request_artifact": request_artifact, "response": authored.raw})
                reported = authored.usage
                usage = {k: v for k, v in (reported or {}).items() if k in {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"} and isinstance(v, int) and not isinstance(v, bool) and v >= 0} if isinstance(reported, dict) else {}
                store.execute("UPDATE material_calls SET artifact=?,usage=? WHERE id=?", (raw, encode(usage), call_id))
                rows = candidates(authored, payload["spec"])
                result = {"job_id": key, "plan_id": payload["spec"]["job"]["id"], "author_id": author.id,
                          "model": authored.model or author.model, "requested_model": author.model,
                          "call_id": call_id, "response_artifact": raw, "rows": rows, "usage": usage,
                          "expected_items": payload["spec"]["job"]["expected_items"], "sources": payload["spec"]["sources"],
                          "seed_artifact": payload["spec"]["seed_artifact"]}
                output = artifacts.put(result)
                with store.connect(immediate=True) as db:
                    db.execute("UPDATE material_calls SET status='complete',finished_at=? WHERE id=?", (now(), call_id))
                    db.execute("UPDATE material_jobs SET status='complete',artifact=?,error=NULL,updated_at=? WHERE id=?", (output, now(), key))
                ctx.event("material_author", f"Material author {author.label} completed {len(rows)} candidates", {"job_id": key, "call_id": call_id, "artifact": output})
                return result
            except BaseException as exc:
                if isinstance(exc, CandidateValidationError) and raw:
                    evidence = artifacts.put({**artifacts.get(raw), "validation_errors": exc.diagnostics})
                    store.execute("UPDATE material_calls SET artifact=? WHERE id=?", (evidence, call_id))
                if getattr(exc, "evidence", None) is not None:
                    evidence = artifacts.put({"request_artifact": request_artifact, "response_error": exc.evidence})
                    store.execute("UPDATE material_calls SET artifact=? WHERE id=?", (evidence, call_id))
                if getattr(exc, "usage", None):
                    store.execute("UPDATE material_calls SET usage=? WHERE id=?", (encode(exc.usage), call_id))
                state = "cancelled" if isinstance(exc, (asyncio.CancelledError, Cancelled)) else "waiting" if isinstance(exc, TeacherUnavailable) else "failed"
                reason = str(exc) if isinstance(exc, (RuntimeError, ValueError, Cancelled)) else type(exc).__name__
                with store.connect(immediate=True) as db:
                    db.execute("UPDATE material_calls SET status=?,error=?,finished_at=? WHERE id=?", (state, reason[:1000], now(), call_id))
                    db.execute("UPDATE material_jobs SET status=?,error=?,updated_at=? WHERE id=?", (state, reason[:1000], now(), key))
                raise

        active, author_counts, resource_counts = {}, {}, {}
        failure = None
        try:
            while active or (pending and failure is None):
                if ctx.cancelled():
                    raise Cancelled("Material generation stopped by operator")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Material generation stage exceeded its execution deadline")
                for item in pending[:] if failure is None else []:
                    if len(active) >= pool.concurrency:
                        break
                    a = item[1]
                    if author_counts.get(a.id, 0) >= a.concurrency:
                        continue
                    if a.resource_pool and resource_counts.get(a.resource_pool, 0) >= pool.resource_limits[a.resource_pool]:
                        continue
                    pending.remove(item)
                    task = asyncio.create_task(dispatch(item))
                    active[task] = item
                    author_counts[a.id] = author_counts.get(a.id, 0) + 1
                    resource_counts[a.resource_pool] = resource_counts.get(a.resource_pool, 0) + 1
                done, _ = await asyncio.wait(active, timeout=min(.25, max(.001, deadline-time.monotonic())), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    item = active.pop(task)
                    author_counts[item[1].id] -= 1
                    resource_counts[item[1].resource_pool] -= 1
                    try:
                        completed[item[0]] = task.result()
                    except (CandidateValidationError, AuthorHTTPError) as exc:
                        # A rejected response must not cancel already-reserved
                        # siblings and turn their work into unknown usage. Stop
                        # dispatching new work and drain only in-flight calls.
                        # Operator cancellation, availability waits and the stage
                        # deadline still interrupt this bounded drain immediately.
                        failure = failure or exc
            if failure is not None:
                raise failure
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
    return [completed[item[0]] for item in requests]


def run_jobs(ctx, specs):
    return asyncio.run(_execute(ctx, specs, ctx.config.material_authors,
                                getattr(ctx.engine, "material_client_factory", None)))
