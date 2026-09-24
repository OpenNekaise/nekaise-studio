# Working on Nekaise Studio

All coding agents, including the interactive assistant and the recovery orchestrator,
must automatically commit completed code, prompt, configuration and documentation changes
after relevant validation, then push the current branch to its configured upstream.
The user authorized this workflow on 2026-09-17; do not request separate commit/push approval.
Stage only the task's intended files or hunks, preserving unrelated working-tree and staged
changes. Keep ignored workspace data, credentials, generated training material and weights
out of Git. Use a descriptive commit message with the change's purpose and validation.
The orchestrator publishes finalized repairs after reviewing host validation evidence;
include the recovery/report reference and record the commit hash, remote/branch and actual
push outcome in its report. If Git permissions, authentication, connectivity or remote
divergence blocks publication, preserve the local work, report the concrete failure and
carry an actionable publication retry into the next review. Never claim a failed push
succeeded or force-push shared history to bypass a rejection.

Read `IDEA.md`, `ARCHITECTURE.md`, and `README.md`. The user-approved behavior is an online
teacher-evaluated training loop. Independent benchmarks stay outside the loop and must not
become a startup requirement or automatic acceptance gate. The teacher is trusted for
teaching and curriculum decisions. Keep the teacher and recovery orchestrator as separate
roles; quota exhaustion is waiting, and other interruptions wake the orchestrator by default.
Studio is the default homepage; Report is the user's operational status interface. The orchestrator reads prior reports
and decides operations without a routine user approval gate. Agent wait/pause schedules
another automatic review; explicit user pause/stop still overrides execution.
The teacher has full access to all teaching history and decides sources, tasks, exact student
prompts and training text, review, token mix, assessment, notes and next teaching actions.
Do not reintroduce fixed curriculum rules, recency cutoffs or a second teaching gate.
`docs/COAPT.md` is included in teacher requests and is part of the execution fingerprint.

The orchestrator owns recovery decisions, including validation requests, further repairs,
retry, continuation, waiting and pausing. Host checks return evidence to the orchestrator;
the controller must not infer a recovery action from a check result. Preserve a report of
what the orchestrator investigated or changed, the evidence, its decision and reasons.
It also owns investigation and resolution of suspected execution or generation defects,
including empty outputs, copied directives and broken response boundaries. Pursue concrete
diagnostics, supported repairs and verification; carry unresolved operational issues into
the next actionable review rather than leaving them as unspecified future work. A wait
must identify the external blocker or bounded cooldown. The teacher still owns teaching
content and assessment; this responsibility does not add a learning-score acceptance gate.
Operator commands, process ownership, provenance and explicit execution budgets still apply.
The orchestrator also owns run/log retention: analyze usefulness and dependencies, decide
keep/archive and explicit raw-log cleanup, and retain reasons and summaries. Cleanup moves
closed logs to recoverable workspace trash. Archived runs remain in the teacher's complete
history; datasets, stage artifacts and lineage are preserved.
Do not impose age/status/score-based retention rules. The orchestrator chooses the next
review interval; the worker yields only at a completed-round boundary for routine reviews.

- Use **CoAPT Mid-training** for this project's unified CPT + SFT training. This is the
  current development scope and the default meaning of training in project discussions.
  Distinguish CPT and SFT only within recipes, material composition and provenance.
  Use **CoAPT Post-training** for the planned RL + OPD family; implementation details
  remain future work. These names do not impose a fixed mixture or training sequence.
- Work in this repository. Treat `../nekaise-corpus` and `../nekaise-studio-bak` as read-only
  sources of data and architectural context; their live operations belong to those repos.
- Keep HTTP/UI, application service, worker, stages, persistence, and providers separate.
  Never import torch/transformers in the API process.
- Source snapshots, teacher outputs, datasets and checkpoints belong in ignored workspace
  storage. Never commit corpus text, credentials, generated training data or model weights.
- Record real observations. Do not label fixtures, invented metrics, or teacher predictions
  as student training results. Fake providers belong in tests, never the live provider list.
- Preserve stage artifacts and input fingerprints. Retry from an explicit stage boundary.
  Completed rounds carry optimizer state; do not claim mid-stage resume. Python/prompt
  changes require a fresh continuation campaign when
  the existing completed-stage fingerprints no longer match.
- Avoid editing Python or prompt files during a live campaign. The executing worker and
  its recorded source fingerprint must describe the same implementation.
- Use the command queue for start/pause/resume/stop. Only the worker owns model processes.
  Cancellation targets recorded process identities, never machine-wide name matches.
- Run the relevant Python integration tests after core changes. Pure dashboard helpers
  have Node tests and syntax checks. Do not launch real training for cosmetic changes.
- `scripts/smoke_campaign.py` incurs teacher calls and GPU work. Use it for explicit live
  validation, not as an automatic unit test. Inspect and preserve failed runs.
- Keep the UI practical and consistent with its Nordic typography, surfaces and palette.
  Main surfaces are lessons, revisions, loss, online diagnostics and activity, not marketing.
- Document extension contracts and concrete limitations. CoAPT Post-training and agentic
  trajectory support remain planned until their real training and evaluation paths are
  implemented and validated.

## Independent benchmark display (authorized 2026-09-16)

The dashboard may read Bench's allowlisted aggregate projection through its dedicated read-only
endpoint. No scores enter teaching records, normal Report history, teacher tools, recovery inputs
or checkpoint decisions. Bench owns asynchronous scheduling and private evidence. This does not
claim filesystem secrecy under the shared Unix account. Preserve the training source lock during
deployment. Consult **Claude Opus 5.5** (`claude-opus-5-5`) for design questions or uncertainty
(operator preference updated 2026-09-24).

## Checkpoint storage decisions (authorized 2026-09-16)

The orchestrator owns whether checkpoint bytes remain resumable, inference-only, or records-only.
It must inspect storage inventory and preserve useful findings/reasons before explicit deletion.
Original manifests, datasets, teaching history, metrics and lineage stay immutable; weights and
optimizer states are no longer subject to a blanket keep-forever rule. Use structured
`checkpoint_retention` decisions and the guarded journaled executor, never manual unlinking.
Current resumption and active inference dependencies are protected. Low disk capacity requests
an orchestrator review before further saves. Benchmark scores do not inform these decisions.

## Aggressive checkpoint retention (authorized 2026-09-23)

The operator prefers a small working set of model checkpoints and explicitly does not
need a long history of model bytes. At retention reviews, prefer records-only history
for superseded checkpoints unless a concrete current recovery, inference or teaching
comparison need justifies their bytes. Preserve the current resumption state and active
readers; retain only a small, individually justified set of fallback/comparison models.
Historical campaign tips, old starting checkpoints and lineage references alone are not
permanent resumption dependencies. Review obsolete optimizer states as well as weights.
The orchestrator chooses the exact set after inspecting dependencies and useful findings;
this preference is not an age, status, score or fixed-count deletion rule. Use explicit
checkpoint_retention decisions and the guarded journaled executor. Keep immutable
manifests, datasets, teaching history, metrics, lineage and deletion summaries. Apply
this preference proactively at future reviews, rather than only freeing one save's worth
of space when the disk is nearly full. Independent benchmark scores remain excluded.

## Required expansion and training efficiency (authorized 2026-09-18)

Every positive-training round under default `expansion_policy=required_v1` must complete
Material Author expansion and consume teacher-selected or edited expanded targets. Pure
weight-preserving diagnostics may skip it. This supersedes older optional-delegation advice;
do not change the live campaign to `legacy_optional` without an explicit operator request.
The teacher retains content/review/candidate-rejection/recipe authority. Rejecting all candidates
requires explicitly choosing zero passes; the host must not silently change the recipe.
Provider failures follow ordinary quota waiting / orchestrator recovery. Increase useful
measured training exposure and expose trained targets / primary Teacher input+output usage in
feedback and Studio. Author usage is separate. Missing usage is not zero, repeated exposure
is not distinct coverage, and efficiency is not learning or an acceptance threshold. Keep
independent benchmark results outside all teaching feedback.

## Autonomous material recovery (authorized 2026-09-22)

The orchestrator must resolve Material Author failures and resume training without requiring
the user to click Resume. Internal author reservation exhaustion is an operational incident,
not a provider quota reset. Inspect the saved failure and repair it or provide concrete retry
diagnostics. An unchanged-stage retry may explicitly grant `material_allowance` for exactly
the unfinished batch's shortfall. Grants are journaled with the recovery and retain all prior
reservations, including unknown usage. Supplemental calls and output reservations in one round
and budget epoch are cumulatively bounded by one original round allowance; no implicit renewal,
teacher-budget reset, model substitution or expansion bypass is authorized. When this envelope
is exhausted, investigate and repair the persistent defect; do not manufacture a no-op
continuation to evade it. Real source repairs use verified continuations and compatible Adam.
Waiting must name the external change or bounded diagnostic cooldown and the concrete next
action. An unchanged internal allowance cannot be repaired by elapsed time or repeated waiting.
Explicit operator pause/stop and provider availability waits retain precedence.

## Trusted author material (authorized 2026-09-22)

The operator selected `material_review_policy=trusted_author_v1` for ongoing training:
the teacher completely trusts generated teaching content. Curriculum expansion jobs
preauthorize complete structurally valid batches at the teacher's planned mix and passes.
Do not run a separate Teacher selection call, prune content, or recreate individual
candidate review during evaluation/reflection. Assess the student's actual behavior.
Keep exact author material, provenance and full history accessible for teaching use;
record preauthorization honestly, never as individual content review. Execution checks,
budgets, required expansion and ordinary failure recovery remain. Empty required targets
are an operational failure, not permission to silently change the recipe to zero passes.
The teacher retains curriculum, source, author, task, dose and online assessment choices.
The historical `teacher_review_v1` path stays readable; recovery must preserve the selected
trust policy unless the operator explicitly changes it.

## Generated general material (authorized 2026-09-24)

Every positive-training round must include original general-purpose material from both
the primary Teacher and Material Author, including native chat instruction-response targets.
Use `general_material_policy=required_v1` for ongoing and new training. Legacy snapshots
remain readable; do not revert ongoing training to `legacy_optional` without an operator
request. Diagnostic zero-pass rounds may omit training material. This requirement supersedes
advice that every lesson or prerequisite must reconnect to the building-energy domain.

The operator wants synthetic material inspired by the coverage of Nemotron-CC v2/v2.1,
DCLM-baseline, FineWeb and Dolma 3 Mix. Do not download, stream or ingest those datasets.
Generate original prose and instruction-response examples; never claim they were sampled
from, reproduce, or are equivalent to the named datasets. Preserve real model/job provenance.

The initial recipe agreed with Claude Opus 5.5 targets 50% general and 50% domain prepared
causal targets: 38% general chat, 12% general prose, 20% fresh domain, 25% domain replay,
5% existing domain corpus; initially one pass. These are starting teaching choices, not
host ratio tolerances or permanent curriculum rules. The Teacher can adapt scope shares,
languages, replay, dose and online observations, recording reasons and retaining the required
general portion. Ordinary short requests and diverse response forms need deliberate coverage;
general examples must not all become domain exercises or use one answer scaffold.

Declare material_scope on lessons, expansion jobs and training readings. Job scope is chosen
by the Teacher and inherited by candidates; original scope survives replay, and unknown
historical scope stays unspecified. Compare planned shares with frozen target accounting
and verified completed exposure. Structural nonzero/provenance checks are not content grading
or learning gates. Trusted-author policy, execution budgets, Teacher authority and benchmark
isolation remain. Track actual prompt responsiveness as well as domain learning; successful
execution, more repeated tokens and lower loss alone do not resolve a chat regression.
