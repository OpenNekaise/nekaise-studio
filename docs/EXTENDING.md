# Extending the loop

Keep an extension at the smallest boundary that owns its behavior.

## Another teacher

Implement `Teacher` in `providers/base.py`: curriculum, revision, assessment design,
grading and reflection. Contracts live in `teaching.py`. The teacher chooses task and
assessment cardinality; IDs only need to be unique and match the corresponding attempts.
It supplies exact `student_prompt` and `training_text`, inclusion decisions, sources,
historical review references, token shares, passes and durable notes/next actions.
Curriculum `train_epochs=0` explicitly skips weight updates while preserving prepared
material and assessment. Config `train_steps=0` means automatic update counting when
training is enabled; positive values override that count.

Every call needs access to the full teaching archive, corpus discovery and the local
CoAPT handbook. Do not turn pagination into a recency limit or silently replace the
teacher's choices with configured defaults. Preserve actual observations, source snapshots,
usage, recorded decisions, execution timeouts and explicit operator budgets.

Register the provider explicitly in configuration and construction. Avoid arbitrary shell
commands from browser input. An OpenCode adapter can use a documented JSON/file transport;
it is not implemented by passing an unchecked command string through the UI.

## Another student or trainer

Implement `generate(checkpoint, rows, on_answer)` and
`prepare(checkpoint, rows)` and `train(checkpoint, dataset, dataset_hash, on_metric)`.
Preparation returns exact token samples and a stream ledger; training receives that frozen
object. Teacher grading returns `needs_practice` and `priority`; reflection persists learning notes. The trainer returns a checkpoint
path and manifest containing parent, dataset identity, config, versions, and file hashes.
Each metric must describe an actual optimizer step. Never substitute generated curves or
success-shaped output for a failed process.

Keep heavy dependencies in a separate runtime. `workers/model.py` documents the current
JSON input and `LOOP` JSONL output protocol. A remote executor may implement the same
contract, but needs explicit cancellation, artifact transfer, and identity guarantees.

## Agentic trajectory SFT and OPD

These are future training stages, not aliases for the current CPT trainer. Add typed
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
where the semantics differ; do not make `train()` ambiguously mean static SFT or dynamic
rollout optimization. Extend UI inspectors for real action/observation records.

## Another stage or storage schema

Add a named stage function with explicit dependencies and serializable output, register
its order in `config.py`, and add failure/retry tests. Keep side effects behind artifacts
and provider interfaces. Record new provenance needed to explain the resulting checkpoint.

Database schema version 2 is explicit; migration from v1 preserves historical runs. Introduce a migration before changing persistent
columns; never silently repurpose old records. New UI-only presentation fields can be
derived by `service.py` without rewriting learning artifacts.

## Another recovery orchestrator

Keep this role separate from the teacher and the scheduler. Implement the strict decision
schema in `recovery.py`, owned-process cancellation, a noninteractive local repair transport,
and bounded logs. Only allow declared recipe fields, with Pydantic validation before creating
a continuation. Coding agents run under the exclusive source lock with no model worker live.
Agent output queues work after the agent exits; it cannot substitute for actual stage results.
Codex uses workspace-write and a JSON output schema; Claude Code uses local file/shell tools
with noninteractive permissions. OpenCode needs a real adapter and recovery integration tests.
