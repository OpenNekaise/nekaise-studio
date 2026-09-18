# Material Authors

Material Authors expand the primary teacher's plan and corrected seed material.
They do not own teaching, assessment or recovery decisions. This is part of CoAPT
Mid-training, using the existing CPT/SFT material recipes and trainer.

## Configuration

The local registry is `workspace/material-authors.json`. New campaigns snapshot it
unless their recipe explicitly contains `material_authors`. Existing campaigns and
continuations retain their recorded registry; an orchestrator can explicitly apply a
new registry through a `material_authors` config update in a fresh continuation.
Changing the file does not silently change an active job or its retry.

Example for the user-selected DeepSeek endpoint:

```json
{
  "authors": [{
    "id": "deepseek-flash",
    "label": "DeepSeek Flash",
    "transport": "openai_chat",
    "location": "remote",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
    "api_key_env": "DEEPSEEK_API_KEY",
    "concurrency": 4,
    "resource_pool": "deepseek-account",
    "max_output_tokens": 8192,
    "timeout_seconds": 600,
    "json_mode": true,
    "options": {"thinking": {"type": "disabled"}}
  }],
  "resource_limits": {"deepseek-account": 4},
  "concurrency": 4,
  "max_calls_per_round": 16,
  "max_output_tokens_per_round": 32768,
  "max_input_chars_per_call": 120000,
  "max_response_bytes": 2000000
}
```

Set `DEEPSEEK_API_KEY=` in the ignored root `.env`, with file permissions `0600`, or
in the service environment. An environment value takes precedence. The file loader
supports simple assignments, quoted values, `export` and comments, without shell
expansion. Credentials are loaded only for author calls and are not exported into
teacher subprocesses. Do not put secrets in registry options or campaign JSON.

The first live high-thinking request consumed all 8192 output tokens as reasoning
and returned no final content. The bounded starter registry therefore disables
thinking; reasoning remains an explicit per-author option for suitable workloads.
Enabling it requires budgeting for both reasoning and complete structured output.
Avoid contradictory toggle and effort settings. Historical campaigns keep their
original snapshots and failure evidence.

Each author has independent model options. Add Kimi or another compatible endpoint
with its documented model, base URL and a separate credential variable. Do not copy
DeepSeek-specific thinking parameters to a backend that does not support them.
The endpoint must support `/chat/completions`; `json_mode=false` omits the response
format parameter, but the returned content still has to match the candidate schema.

For local models, register an operator-managed compatible serving endpoint with
`location=local`, the actual model name and optionally an empty `api_key_env`.
This implementation connects to the endpoint; it does not launch model servers,
load arbitrary checkpoints or manage their VRAM. If the server shares the student's
GPU, account for its resident memory before training. A shared resource pool limits
author requests, not server residency. Native local-process adapters can implement
`MaterialAuthor` and register explicitly in `AUTHOR_TRANSPORTS` later; none is claimed
by the HTTP implementation. Unsupported transports fail before dispatch.

## Parallelism and execution accounting

One workspace worker dispatches a bounded number of async requests, sharing one HTTP
client. It checks global, per-author and shared-resource limits before starting a job.
Blocked jobs do not occupy all worker slots ahead of independent authors. Registered
author count is separate from active concurrency; no process/thread is created for
every model or candidate. This is a single-host scheduler, not a distributed queue.

The teacher chooses independent expansion jobs and expected candidate counts.
Expected counts may miss; record the shortfall and let the teacher select the actual
package. A job output budget includes provider reasoning as well as visible content.
Only structured teaching fields, not `reasoning_content`, become candidate text.

Reserve one call and the requested maximum output tokens atomically before dispatch.
Reservations remain conservative even when reported output is smaller. All attempts,
including failed/unknown remote work, count toward the current round allowance.
Explicit operator Resume renews the allowance using `teacher_budget_since`; automatic
retries do not. Input-character, response-byte, per-author timeout and stage-deadline
bounds also apply. These are usage limits, not a monetary price guarantee. Missing
usage is unknown, not zero cost; no exactly-once billing guarantee is made.

## Durable stages and teacher authority

`revise → expand → material_select → gate → freeze` preserves original seed teaching.
For diagnostics or explicit legacy_optional policy, empty jobs make the added stages no-ops with no author or
selection call. `material_select` belongs to the primary teacher and can select exact
candidate IDs or whole immutable job batches, edit candidates, omit seeds, and choose
final shares/passes. Every unselected candidate stays in the expansion artifact.
The teacher describes its chosen inspection scope; logs do not prove comprehension.

Complete jobs are keyed by round, immutable plan/seeds/sources, provider configuration
and exact request content. Retry reuses only complete matching results. Responses are
saved before successful completion is recorded; crashes before completion can still
produce duplicate charges, which remain visible. The normal workspace worker lock
owns dispatch; there is no lease timeout that can steal a live request. Stale running
calls become uncertain when a new worker/stage takes ownership.

Cancellation closes local HTTP tasks and preserves completed work. A remote request
already accepted may still be billed. HTTP 402/429 becomes quota/rate waiting; other
transport, schema or provenance faults go to the existing orchestrator. No automatic
provider switch, JSON repair, top-up generation or reduced-package fallback occurs.
The teacher can choose a different material plan after an explicit recovery handoff.

## Provenance, history and interface

Candidates have separate typed records. Material rows explicitly have no student
attempt; they never create empty-answer metrics. Accepted material enters the teacher
stream with `material_origin`, while generated CPT prose stays distinct from raw
corpus text. Gate/freeze, assessment, reflection and replay see the full package.
Source and seed identifiers must refer to the job's supplied immutable context.
These structural checks do not judge semantic correctness or impose a teaching score.

The read-only archive exposes `material_candidates` with round, offset/limit and an
optional candidate ID. All content and provider response artifacts remain accessible.
The teacher request contains a bounded preview, not a teaching-history cutoff. Large
teacher contexts use the existing complete externalized recorded-data file.

The Report work panel separates author API usage/reservations from prepared targets
by origin and actual optimizer metrics. The iteration page shows paged authored
materials and teacher choices. Its read-only endpoint is
`GET /api/rounds/{round_id}/materials?offset=0&limit=20`.

## Extension and validation

Adapters implement async `MaterialAuthor.generate(author, request, env_file,
max_bytes)` and return `AuthorResult`: normalized content, completion, model and usage,
alongside the untouched provider envelope. The scheduler/selection code does not parse
vendor response paths. Transports may translate the common chat request to their own
protocol. Keep network code
inside providers, dispatch inside worker stages, ML imports outside the API, and
credentials outside persisted artifacts. Tests may inject an HTTP mock transport;
there are no fake live providers.

This extension aims to leave `workers/model.py`, `training.py` and `serialization.py`
unchanged. Actual compatibility hashes decide whether Adam can be inherited. Source
and prompt changes still require a fresh continuation. Measure accepted targets per
teacher minute and API usage, selection cost, source coverage, actual updates and
teacher-chosen online observations before claiming throughput or learning gains.

Official protocol references: [DeepSeek first call](https://api-docs.deepseek.com/),
[concurrency and waiting](https://api-docs.deepseek.com/quick_start/rate_limit/),
[thinking controls](https://api-docs.deepseek.com/guides/thinking_mode/),
and [HTTPX async client](https://www.python-httpx.org/async/).


## Required production workflow (2026-09-18)

New recipes default to `expansion_policy=required_v1`. Selection rejects positive training
plans without valid jobs before student inference. Freeze requires completed expansion,
teacher-selected candidates and positive prepared expanded targets. Train verifies the same
round-bound receipt before updates. These are execution/provenance requirements; semantic
quality belongs to the teacher, who may edit/reject candidates and explicitly choose zero
passes. API errors retain waiting/recovery behavior; the host invents no replacement recipe.
Primary-teacher efficiency is measured trained targets / primary Teacher input+output;
author usage stays separate. [The handbook](COAPT.md#training-production-efficiency) defines
missing data, partial rounds, repetition and chart semantics.
