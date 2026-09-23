# Material Authors

Material Authors expand the primary teacher's plan and corrected seed material.
They do not own teaching, assessment or recovery decisions. This is part of CoAPT
Mid-training, using the existing CPT/SFT material recipes and trainer.

The operator's ongoing-training decision is `material_review_policy=trusted_author_v1`:
teacher curriculum jobs authorize complete generated batches at the planned mix/passes.
There is no post-generation Teacher selection call or individual content review, including
in later evaluation/reflection. Structural/schema/provenance checks remain; exact duplicate
content is not silently removed. Artifacts explicitly distinguish this preauthorization
from individual inspection, and synthetic material never becomes a student observation.
The teacher still chooses sources, jobs, seed corrections, dose and online student assessment.
The compatibility default `teacher_review_v1` retains the older workflow described below;
historical artifacts and full exact author text remain readable under either policy.

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
author requests, not server residency. Unsupported transports fail before dispatch.

### Claude Code authors

The `claude_code` transport launches the configured Claude executable through the
workspace worker. Use an existing authenticated CLI installation and a pinned model
ID verified on that installation, for example:

```json
{
  "id": "claude-opus", "label": "Claude Code Opus 5",
  "transport": "claude_code", "model": "claude-opus-5",
  "concurrency": 2, "resource_pool": "claude-account",
  "max_output_tokens": 32768, "timeout_seconds": 600,
  "options": {"effort": "medium", "thinking": false}
}
```

Include `"claude-account": 2` in the registry's `resource_limits`. Omit `base_url`
and `api_key_env`: this adapter uses the CLI's existing authentication. It runs in
an isolated job directory with tools disabled, safe mode, empty settings sources,
strict MCP configuration and no session persistence. The worker records each
process PID and start identity; cancellation joins its runner, and a replacement
worker reconciles only recorded owned children.

The CLI emits structured candidates and a streamed result envelope. On the validated
CLI 2.1.278, `--max-turns 1` alone does **not** prevent internal truncation recovery.
The adapter therefore allows one observed model request and aborts on truncation or
another request. Half the job output reservation is the response cap; half reserves
room for an already accepted continuation during cancellation. CLI transport retries
are disabled and structured-output attempts are limited to one. Timeout, cancellation
and incomplete envelopes retain reservations and mark usage unknown; server-side
cancellation and exactly-once charging cannot be guaranteed. Revalidate this contract
when upgrading the CLI. No host JSON repair or automatic generation fallback occurs.

Reported input includes fresh input, cache creation and cache reads once. Opus author
usage is separate from primary Teacher usage and available in `by_author`; missing
usage remains unknown. CLI `costUSD` is preserved as provider evidence, not represented
as actual subscription spend. Teacher selection/editing and the ordinary freeze/train
provenance checks still determine what enters training.

### Codex authors

The `codex_code` transport uses the worker's existing authenticated Codex CLI. The
operator-selected role configuration on 2026-09-22 is Claude Opus 5.5
(`teacher_provider=claude`, `teacher_model=claude-opus-5-5`) for primary teaching and
online evaluation, and GPT-5.6 Terra for material generation. The recovery
orchestrator remains a separate role. An example Terra entry is:

```json
{
  "id": "gpt-terra", "label": "GPT-5.6 Terra",
  "transport": "codex_code", "model": "gpt-5.6-terra",
  "concurrency": 4, "resource_pool": "codex-account",
  "max_output_tokens": 32768, "timeout_seconds": 600,
  "options": {"effort": "low"}
}
```

Include `"codex-account": 4` in `resource_limits`; omit endpoint/key settings.
Each job is an ephemeral, read-only CLI process with a strict output schema,
ignored user configuration/rules and disabled project instructions, plugins,
hooks, shell, web, apps and delegation. Remaining tool/unsupported-item events
abort the owned process. No author-level fallback, JSON repair or retry is added.
Cancellation joins all owned processes and retains every reservation.

The validated CLI is Codex 0.155.1. Its JSONL stream exposes completed items and
turn usage, **not** individual model-request boundaries or live token usage. It
does not offer a verified provider output-token cap. `max_output_tokens` is an
up-front reservation and a reported-output acceptance ceiling for this transport;
the catalog explicitly returns `max_response_output_tokens=null`. Actual remote
usage may exceed reservations, including through CLI internal retries or work
continuing after cancellation. Timeout and output-byte caps bound local execution;
they are not hard remote token or monetary limits. Known output overruns are
rejected and additionally charged against the remaining round allowance without
rewriting the original reservation. Unknown usage remains unknown. Do not describe
the reservation as a guaranteed upper bound or substitute an unverified config key.

Acceptance requires exactly one observed clean turn and schema/provenance-valid
final JSON. Input already includes cached tokens, and output already includes
reasoning; neither is added twice. Author usage stays outside primary Teacher
efficiency. Artifacts retain the JSONL events, executable, requested model and
enforcement limitations. The CLI pins the requested model, but this JSONL version
does not attest the served model ID; artifacts state that limitation explicitly.
Revalidate the CLI contract after upgrades. Role changes and author-registry
changes take effect through a fresh continuation, preserving compatible Adam,
teaching history, required expansion and the existing budget epoch.

References: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

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
Reservations are not released when reported output is smaller; known output above
the reservation also consumes allowance. The Codex limitation above still applies. All attempts,
including failed/unknown remote work, count toward the current round allowance.
Explicit operator Resume renews the allowance using `teacher_budget_since`; automatic
retries and automatic continuations do not reset that timestamp. Before dispatch,
the scheduler checks that the remaining same-round allowance can fund all unfinished
jobs, avoiding partial spending on a package that cannot finish. Input-character,
response-byte, per-author timeout and stage-deadline
bounds also apply. These are usage limits, not a monetary price guarantee. Missing
usage is unknown, not zero cost; no exactly-once billing guarantee is made.

## Durable stages and teacher authority

`revise → expand → material_select → gate → freeze` preserves original seed teaching.
For diagnostics or explicit legacy_optional policy, empty jobs make the added stages no-ops with no author or
selection call. `material_select` belongs to the primary teacher and can select exact
candidate IDs or whole immutable job batches, edit candidates, omit seeds, and choose
final shares/passes. Every unselected candidate stays in the expansion artifact.
The teacher describes its chosen inspection scope; logs do not prove comprehension.

Author requests keep the original teacher explanation in an input-only
`seed_feedback` map keyed by seed ID. Seed examples contain material fields rather
than a `teacher` field that is forbidden in candidate output. The original revision,
exact training text and response, and source provenance remain intact. This avoids
demonstrating an invalid output field; it does not guarantee that a provider will
follow the schema or finish within the teacher's chosen output allowance.

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
Repeated JSON evidence of at least 2,048 characters may share one exact value in the
rendered Teacher request. A `$shared` reference resolves in one lookup; shared entries
contain the original value, not a summary. Original `input.json` inputs and complete
`recorded-data.json` remain available, including every candidate, source and token ID.
Literal marker collisions retain the ordinary representation. For already externalized
requests, an optional `shared-data.json` provides the same lossless view. This encoding
threshold does not limit history, curriculum, candidate counts or review scope.

The Report work panel separates author API usage/reservations from prepared targets
by origin and actual optimizer metrics. The iteration page shows paged authored
materials and teacher choices. Its read-only endpoint is
`GET /api/rounds/{round_id}/materials?offset=0&limit=20`.

## Extension and validation

Adapters implement async `MaterialAuthor.generate(author, request, env_file,
max_bytes, execution=None)` and return `AuthorResult`: normalized content, completion, model and usage,
alongside the untouched provider envelope. The scheduler/selection code does not parse
vendor response paths. Transports may translate the common chat request to their own
protocol. `execution` carries the owning store/call ID, configured CLI and job directory;
process transports require it, while HTTP transports ignore it. Keep network code
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
Claude references: [CLI flags](https://code.claude.com/docs/en/cli-reference),
[environment controls](https://code.claude.com/docs/en/env-vars),
and [model configuration](https://code.claude.com/docs/en/model-config).


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


### Orchestrator recovery allowances

Internal material reservation exhaustion wakes the orchestrator (`material_budget`); it is
not a provider availability wait. After inspecting the failed call, the orchestrator may
return an explicit `material_allowance` with an unchanged-stage retry. The grant names
the current round and budget epoch, expected call/reservation totals, the exact additional
call/token shortfalls, and a reason. The host checks the current ledger and quiescent
expansion, then atomically journals the grant and queues retry. Reapplying a resolved
decision cannot grant twice. User holds, stale epochs/totals, unrelated rounds and
continuations reject grants. All failed/cancelled reservations and unknown usage remain
counted; `teacher_budget_since` is unchanged. Ordinary retries grant nothing implicitly.

Cumulative supplementary calls/tokens per round and epoch cannot exceed the original
round allowance. Exhausting that envelope requires investigation and a supported repair,
not endless unchanged retries or a no-op continuation. Source repairs still require a
fresh continuation with verified checkpoint and optimizer compatibility; no old-round
grant transfers to it. External provider quota still waits for availability.

Schema failures save bounded paths and error types without rejected values, custom
validator messages or source text. The next attempt on the same job receives those
diagnostics; its exact request has a separate immutable artifact, while the job identity
and completed-job reuse remain stable. Candidate validation and teacher selection remain
mandatory. Unknown historical diagnostics may be absent and are never invented.

### Citation identifiers

Each Author request constrains candidate `source_keys` and `seed_ids` to the exact
identifiers supplied for that job in its output schema. An empty allowlist permits
only an empty array. Codex receives these constraints in its strict output schema;
other transports receive the same schema in the request. The host independently
checks returned citations even if a transport ignores schema constraints. It never
rewrites a mistyped citation or drops the affected row. These are provenance checks,
not a review of teaching content; Teacher instructions, material and dose remain
unchanged. Historical requests and rejected responses remain immutable.
