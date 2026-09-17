"""Durable bounded fan-out for material authors, owned by one loop worker."""
from __future__ import annotations

import asyncio
import json
import time

import httpx

from .artifacts import digest
from .failures import TeacherUnavailable
from .material_types import CandidateBatch
from .processes import Cancelled
from .providers.material import AUTHOR_TRANSPORTS
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
            created_at TEXT NOT NULL, finished_at TEXT
        );
        """)


def request_body(author, spec, schema):
    instruction = (
        "You are a Material Author working for the primary teacher. Produce candidate teaching material, not student observations or evaluation scores. "
        "Follow the teacher's expansion instructions and corrected seed demonstrations. Use only the supplied source keys for citations; "
        "an empty source list means authored material. Do not invent source identifiers or student attempts. "
        "Return one JSON object matching the supplied schema. All explanatory text intended for training belongs in the specified material fields. "
        "Do not insert model-specific role markers or a thinking prefill: the student tokenizer will serialize accepted content."
    )
    body = {**author.options, "model": author.model,
        "messages": [{"role": "system", "content": instruction},
                     {"role": "user", "content": json.dumps({"task": spec, "output_schema": schema}, ensure_ascii=False)}],
        "max_tokens": spec["job"]["max_output_tokens"], "stream": False}
    if author.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def candidates(result, spec):
    try:
        if not result.complete or not isinstance(result.content, str):
            raise ValueError("Author response was incomplete or did not contain final content")
        batch = CandidateBatch.model_validate_json(result.content)
        source_keys = set(spec["sources"])
        seed_ids = set(spec["job"]["seed_ids"])
        for row in batch.rows:
            if not set(row.source_keys) <= source_keys or not set(row.seed_ids) <= seed_ids:
                raise ValueError("Author candidate cited an unprovided source or seed")
        return batch.model_dump()["rows"]
    except (KeyError, TypeError, ValueError):
        # Raw evidence is already saved; avoid dumping complete data into errors.
        raise ValueError("Material author response failed completion, schema or provenance validation; inspect saved call artifact") from None


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
    deadline = time.monotonic() + ctx.config.max_stage_seconds
    client_options = {"timeout": httpx.Timeout(65, connect=15), "follow_redirects": False, "trust_env": False,
                      "limits": httpx.Limits(max_connections=pool.concurrency, max_keepalive_connections=pool.concurrency)}
    factory = client_factory or httpx.AsyncClient
    async with factory(**client_options) as client:
        adapters = {name: adapter(client) for name, adapter in AUTHOR_TRANSPORTS.items()}

        async def dispatch(item):
            key, author, payload, size = item
            with store.connect(immediate=True) as db:
                used = db.execute("SELECT COUNT(*) AS n,COALESCE(SUM(c.reserved_tokens),0) AS tokens FROM material_calls c JOIN material_jobs j ON j.id=c.job_id WHERE j.round_id=? AND c.created_at>=?", (ctx.round["id"], since)).fetchone()
                reservation = payload["request"]["max_tokens"]
                if used["n"] >= pool.max_calls_per_round or used["tokens"] + reservation > pool.max_output_tokens_per_round:
                    raise TeacherUnavailable("Material-author allowance exhausted, including reservations for uncertain calls. Explicit Resume renews this allowance; saved completed jobs remain reusable.", "budget")
                call_id = db.execute("INSERT INTO material_calls(job_id,stage_id,status,reserved_tokens,input_chars,created_at) VALUES(?,?,'running',?,?,?)", (key, ctx.stage_id, reservation, size, now())).lastrowid
                db.execute("UPDATE material_jobs SET status='running',error=NULL,updated_at=? WHERE id=?", (now(), key))
            ctx.event("material_author", f"Material author {author.label} started {payload['spec']['job']['id']}", {"job_id": key, "call_id": call_id})
            raw = None
            try:
                authored = await adapters[author.transport].generate(author, payload["request"], env_file=ctx.engine.settings.root/".env", max_bytes=pool.max_response_bytes)
                raw = artifacts.put({"request_artifact": key, "response": authored.raw})
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
                if getattr(exc, "evidence", None) is not None:
                    evidence = artifacts.put({"request_artifact": key, "response_error": exc.evidence})
                    store.execute("UPDATE material_calls SET artifact=? WHERE id=?", (evidence, call_id))
                state = "cancelled" if isinstance(exc, (asyncio.CancelledError, Cancelled)) else "waiting" if isinstance(exc, TeacherUnavailable) else "failed"
                reason = str(exc) if isinstance(exc, (RuntimeError, ValueError, Cancelled)) else type(exc).__name__
                with store.connect(immediate=True) as db:
                    db.execute("UPDATE material_calls SET status=?,error=?,finished_at=? WHERE id=?", (state, reason[:1000], now(), call_id))
                    db.execute("UPDATE material_jobs SET status=?,error=?,updated_at=? WHERE id=?", (state, reason[:1000], now(), key))
                raise

        active, author_counts, resource_counts = {}, {}, {}
        try:
            while pending or active:
                if ctx.cancelled():
                    raise Cancelled("Material generation stopped by operator")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Material generation stage exceeded its execution deadline")
                for item in pending[:]:
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
                    completed[item[0]] = task.result()
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
    return [completed[item[0]] for item in requests]


def run_jobs(ctx, specs):
    return asyncio.run(_execute(ctx, specs, ctx.config.material_authors,
                                getattr(ctx.engine, "material_client_factory", None)))
