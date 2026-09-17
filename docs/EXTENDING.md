# Extending the loop

Keep an extension at the smallest boundary that owns its behavior.

The current path is **CoAPT Mid-training (CPT + SFT)**. **CoAPT Post-training (RL + OPD)**
is planned; concrete implementations are deferred. Use these collective names for
training paths, and individual method names for recipe or implementation details.
The maintained naming contract is in [COAPT.md](COAPT.md#project-terminology-and-current-scope).

## Another teacher

Implement `Teacher` in `providers/base.py`: curriculum, revision, assessment design,
grading and reflection. Contracts live in `teaching.py`. The teacher chooses task and
assessment cardinality; IDs only need to be unique and match the corresponding attempts.
It supplies exact `student_prompt` and `training_text`, inclusion decisions, sources,
historical review references, token shares, passes and durable notes/next actions.
Curriculum `train_epochs=0` explicitly skips weight updates while preserving prepared
material and assessment. Config `train_steps=0` means automatic update counting when
training is enabled; positive values override that count.
Diagnostic curricula may use all-zero token shares with `train_epochs=0`; enabled
training requires shares summing to 1. Preserve the teacher's choice through preparation.

Every call needs access to the full teaching archive, corpus discovery and the local
CoAPT handbook. Do not turn pagination into a recency limit or silently replace the
teacher's choices with configured defaults. Preserve actual observations, source snapshots,
usage, recorded decisions, execution timeouts and explicit operator budgets.

Teacher requests also carry `operations`: the current operator hold and latest command,
plus the latest applied recovery decision/report on the campaign's ancestor branch.
Pending proposals and sibling continuations are excluded from this default handoff.
The read-only archive exposes `reports` (cursor pagination via `before`/`next_before`)
and `report` (`recovery_id`) for all decision turns, checks and actual outcomes. This
context does not replace the teacher's saved reflection or force a diagnostic curriculum.
An orchestrator-requested generation reproduction can use teacher-selected prompts and
zero training passes through the ordinary worker stages. The orchestrator must inspect
the resulting evidence; a completed round with no prompts does not validate generation.

Register the provider explicitly in configuration and construction. Avoid arbitrary shell
commands from browser input. An OpenCode adapter can use a documented JSON/file transport;
it is not implemented by passing an unchecked command string through the UI.

## Another student or trainer

Implement `generate(checkpoint, rows, on_answer)` and
`prepare(checkpoint, rows)` and `train(checkpoint, dataset, dataset_hash, on_metric)`.
This is the unified CoAPT Mid-training contract. `cpt` and `sft` remain material labels
within a recipe and both feed the teacher stream; they do not select separate trainers.
Preparation returns exact token samples and a stream ledger; training receives that frozen
object. Rows default to whole-text tokenization (`training_tokenization=full_text`).
For the teacher's optional `prompt_prefix` mode, freeze records `training_prompt`, which
must be an exact nonempty prefix of row `text`. Encode it with generation's special-token
handling and encode the remaining suffix without extra special tokens; then apply the
usual EOS and chunking rules. Preserve this mode on historical replay. Neither mode
changes the exact text or full-sequence loss; unsupported modes must not silently fall
back. Frozen IDs and row metadata contribute to dataset provenance.
Teacher grading returns `needs_practice` and `priority`; reflection persists learning notes. The trainer returns a checkpoint
path and manifest containing parent, dataset identity, config, versions, and file hashes.
Each metric must describe an actual optimizer step. Never substitute generated curves or
success-shaped output for a failed process.

Students supporting historical comparisons implement
`compare(checkpoint, reference, rows, on_answer, *, reference_format=None)` returning separate `current` and
`reference` answer lists with the same IDs. Curriculum `comparison_round_id` defaults
to empty; a nonempty value selects one completed round's distinct checkpoint in this
workspace's runs. Selection freezes its training artifact identity; draft rechecks that
identity and all checkpoint hashes before execution. The local adapter uses sequential
owned FP32 inference processes and one shared `max_stage_seconds` timeout, with separate
`draft-N/current/` and `draft-N/reference/` inputs/logs. Current answers alone update the
student projection; each lesson's `comparison` contains reference provenance and the full
generation evidence for teacher revision/reflection and artifact inspection. Reference
answers are never automatically trained, graded or selected as the next checkpoint.
No automatic comparison or score gate is added. `comparison_checkpoint` defaults to
`output`; `input` selects the completed round's `model_before`, corroborated by the
content-verified training artifact's `manifest.parent`. Input comparison supports only
pinned local Hugging Face snapshots, with all files linked to hash-verified cache blobs
(Git blob SHA-1 or SHA-256), required configuration/tokenizer files and complete
safetensors shards. Selection freezes revision and file SHA-256 values; draft repeats
verification and rejects any identity change. No arbitrary teacher path, download,
cache write or checkpoint replacement is allowed. Historical output artifacts retain
their original paths even when the repository has moved; input comparison does not
load those output weights. These checks establish recorded lineage and current bytes,
not a retroactive input checksum or remote authentication of the revision's file list.

The first live output comparison completed with separate current/reference evidence
and audit-consistent but unreliable answers on both checkpoints. Input snapshot
comparison still requires live validation through the normal worker after host checks.

Keep heavy dependencies in a separate runtime. `workers/model.py` documents the current
JSON input and `LOOP` JSONL output protocol. A remote executor may implement the same
contract, but needs explicit cancellation, artifact transfer, and identity guarantees.
`ProcessRunner.run()` decodes `LOOP` JSONL events only when given an `on_message` callback.
Teacher and recovery transports use plain logs, where quoted protocol examples are text.

## CoAPT Post-training and trajectory support (planned)

CoAPT Post-training is the collective name for RL and OPD. The design notes below are
future extension contracts; the current development scope is Mid-training. Agentic
trajectory supervision is supporting future work beyond the implemented text QA material.
These extensions need real training paths beyond the current Mid-training trainer. Add typed
trajectory records containing tasks, assistant decisions, actual tool observations,
environment versions, and verified outcomes. Corrected actions must be executed again;
they cannot reuse observations from the original action. Trajectory SFT needs role/loss
masks for assistant targets and tests for those masks.

OPD fixes the task curriculum while refreshing student rollouts during training. Its
scorer must evaluate supplied student tokens with compatible tokenization. Record teacher
identity, rollout checkpoint, staleness policy, token-level feedback, and the actual
distillation objective. A text-only teacher CLI is insufficient for probability-based OPD.
Online task success checks remain separate from teacher-likelihood signals.

The outer loop and dashboard record transport can remain. Add a separate trainer contract
where the semantics differ; keep the static Mid-training dataset contract distinct from
dynamic Post-training rollout optimization. Extend UI inspectors for real action/observation records.

## Another stage or storage schema

Add a named stage function with explicit dependencies and serializable output, register
its order in `config.py`, and add failure/retry tests. Keep side effects behind artifacts
and provider interfaces. Record new provenance needed to explain the resulting checkpoint.

Database schema version 3 is explicit; migrations from v1/v2 preserve historical runs. Introduce a migration before changing persistent
columns; never silently repurpose old records. New UI-only presentation fields can be
derived by `service.py` without rewriting learning artifacts.

## Another recovery orchestrator

Keep this role separate from the teacher and the scheduler. The orchestrator owns recovery
actions, including whether to validate, repair further, retry, continue, wait or pause.
Implement the strict decision schema in `recovery.py`, owned-process cancellation, a noninteractive local repair transport,
and bounded logs. Only allow declared recipe fields, with Pydantic validation before creating
a continuation. Coding agents run under the exclusive source lock with no model worker live.
Agent output queues work after the agent exits; it cannot substitute for actual stage results.
Codex uses workspace-write and a JSON output schema; Claude Code uses local file/shell tools
with noninteractive permissions. OpenCode needs a real adapter and recovery integration tests.

Every response includes a concise `reason` and a `report` covering investigation, changes,
observed checks, the chosen action and outstanding work. `action=check` requests the host
Python integration suite. Source/prompt/handbook changes also trigger these checks.
Results return in the next incident's `feedback`, including the previous decision, source
hash, outcome and paths to complete logs. The agent may repair again or return its final
decision. Each response replaces the previous proposal, including its config updates.
The controller must not translate a successful check into continuation or a failed check
into pause. Existing operator controls and execution budgets remain effective.

Artifacts live under `workspace/recoveries/<id>/attempt-<n>/turn-<n>/`: `incident.json`,
transport logs and response, `decision.json`, `report.md`, and host `checks.json`/`checks.log`
when requested. The attempt root keeps the final decision/report and source backups.
Check requests per attempt use `max_repair_attempts`, followed by one final decision turn;
exhausted execution allowances and transport failures remain visible with preserved reports.

The orchestrator also receives `history-inventory.json` with all runs, lineage, current
retention decisions and raw log paths/hashes. Its final response includes `run_retention`
(keep/archive, display label and reason), `log_removals` (explicit path, hash, summary and
reason), and `history_review_after_rounds`. The host enforces file ownership and integrity;
the agent decides usefulness and the next review interval. Routine `history_review`
incidents run between completed rounds when `manage_history` and `auto_recover` are enabled.
They use the normal recovery/command queue and do not require a code change or a new campaign.

Cleanup only moves closed raw logs into `workspace/trash/history/<recovery_id>/`, with an
idempotent journal. It does not permanently purge files or remove learning evidence.
Archived runs stay available to the teacher and under Archived runs in the dashboard.
The dashboard also displays orchestrator reports. Inspect with `nekaise-loop history-inventory`
or `nekaise-loop history-reviews`; restore a cleaned log with `nekaise-loop restore-log <id>`.
The read-only teacher archive exposes `log_summaries` with journal entries and recovery IDs.

`Curriculum.scoring_round_ids` requests historical frozen-sample likelihood diagnostics.
`LocalModel.score_history` launches only worker-owned `workers/scoring.py` subprocesses
and shares one timeout with subsequent draft generation. The immutable draft stores
`historical_scoring`; reflection receives it directly. Each pair preserves artifact and
dataset identities, checkpoint file hashes, exact token IDs, runtime precision and
per-token target/EOS probabilities. Unaligned prompt boundaries remain unassigned;
no whole row is reconstructed from chunks. CPU records only FP32; CUDA additionally
records BF16 autocast over FP32 weights. Eval mode does not reproduce training dropout.
This path never optimizes, writes checkpoints or fabricates free-running answers.

## Native single-turn chat

`student_format=chat_template` uses one user message with `enable_thinking=false` and
`add_generation_prompt=true`. `raw_text` remains the backward-compatible default; row
`format` overrides the campaign only when explicitly set. Do not add special tokens twice
or truncate away a native assistant header. `chat_response` training rows carry raw user
`training_prompt` and assistant-only `training_response`. Serialize exact generation-prefix
IDs, response IDs and native closing IDs through the first effective EOS. Freeze the
rendered text, prefix IDs, template hash and closing evidence; no extra tokenizer EOS follows
chat EOS. All tokens carry loss. Reject unsupported native serialization rather than
silently altering content. Replay checks recorded template hashes but uses the current
tokenizer; raw replay across lineages is not guaranteed token-identical. `serialization.py`
is included in optimizer compatibility hashing. No multi-turn or tool trajectory support
is implied. Comparisons use the saved reference interface unless a row explicitly overrides
both sides; preserve separate prompt evidence and label checkpoint-and-interface comparisons.


## Optional same-round comparisons and work accounting

`Evaluation.compare_before` selects input-checkpoint inference for individual online
questions. `LocalModel.compare(..., reference_rows=...)` runs every current prompt but
only the selected reference prompts, preserving one shared timeout and separate logs.
The `evaluate` artifact freezes checkpoint identities; `answer` rechecks them before
execution. Never append references, criteria or checkpoint labels to student prompts.
Identical no-update checkpoints reuse one recorded answer explicitly, without a delta.

`Evaluation.dimensions` declares teacher-named `{name,rubric}` criteria before answers;
`Grade.dimensions` returns matching `{name,score,feedback}` entries. The host validates
names only, not educational correctness or a formula for the overall score. Paired
grades use one shuffled request with opaque IDs and no checkpoint labels. Restore IDs
and provenance on persistence. Match recorded prompt tokens, generation settings and
runtime before emitting a per-item score difference; this is no acceptance gate.

`learning_work.round_work` reports real per-attempt counters and stage durations without
importing ML packages. `latest_work` follows continuation ancestry, never sibling runs.
Prepared coverage and currently present checkpoint bytes are labeled separately from
consumed tokens and originally saved bytes. Missing history is explicit evidence, not
a reason to block recovery. Report and teacher decisions share these observations.

Continuation consumes an initial optimizer reset only after an actual completed update.
Do not propagate an already-consumed `inherit_optimizer=false` into an otherwise
compatible continuation. Diagnostics do not consume it; explicit new reset requests
and the existing recipe/trainer fingerprint checks remain authoritative.
## Bounded batch generation and work accounting

The active inference transport uses `workers/generation.py`. Keep torch/transformers
inside ML subprocess functions. `generation_batch_size` caps rows and
`generation_batch_tokens` caps `rows * (longest_prompt + max_new_tokens)`; a prompt
that cannot fit alone fails preflight. Limits constrain execution, not teaching.
`batch_groups`, when supplied by the adapter, must partition every unique prompt ID
once. The worker returns final rows in request order even when group order differs.

Each answer retains its exact unpadded IDs and serialization, generated IDs through
its first checkpoint EOS, raw/decoded text, stop reason, uncached audit, runtime and
`generation_execution`. `generation_batch` contains the shared group identity and
timer. Never sum per-answer shared seconds. Compare execution signatures in addition
to prompt/decoding/runtime before treating online pairs as matched conditions.

The existing trainer and legacy generator in `workers/model.py` remain intact to
preserve recorded optimizer compatibility. Consolidate their common inference code
only during a future deliberate trainer compatibility transition; do not silently
remove covered source from the hash. All execution source still enters the campaign
fingerprint, so inference/prompt changes use a fresh continuation.

Teacher work estimates are optional historical data and never a training admission
rule. Available/prepared/unused/repeated counts are target occurrences, not unique
knowledge. Preserve the distinction between teacher estimates, prepared execution
plans, actual metric exposure and retained optimizer lineage. Do not use update
fill or GPU activity as a learning outcome.

## Another material author

Material authors are independently registered from the primary teacher. Implement the
async MaterialAuthor protocol and explicitly register the transport factory. Compatible
HTTP APIs, including local serving endpoints, use the existing openai_chat transport with
per-author options, env-key names and resource-pool limits. No arbitrary browser-provided
command or fake provider is registered. Use typed candidate text and immutable provenance;
only the teacher selects final training rows and recipe. Extend tests for cancellation,
quota waiting, unknown charges, retry reuse and exact selection. The complete
[author contract](MATERIAL_AUTHORS.md) documents worker ownership and scope.
