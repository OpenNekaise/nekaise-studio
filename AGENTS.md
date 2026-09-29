# Working on Nekaise Studio

## Retain usable Author text (authorized 2026-09-29)

Use `material_response_policy=salvage_v1` in the continuing workload. Receive Author
material per item: extra fields, absent metadata, duplicate IDs, unresolvable citations
and partial batches must not discard usable teaching text or trigger paid rewriting.
Preserve exact raw outputs and host normalization receipts. Only resolved citations
become source links; unresolved claims remain explicit provenance. Complete bounded
targets from truncated JSON may train; never label an open string as a finished answer.
Usable prose without a reliable question/answer boundary trains as full text and counts
as prose, retaining the planned scope. This narrowly supersedes prior format/completion
gates; Teacher still plans native general chat and sees actual targets and shortfalls.
No semantic grading or separate selection call is added. Truly empty/nonfinal/provider
failures retain orchestrator recovery and cumulative budgets. Keep all Authors, source
policies, exact Kai identity, forward loops and compatible Adam. Strict historical
snapshots stay readable; recovery cannot silently change this selected response policy.
See [Material Authors](docs/MATERIAL_AUTHORS.md#retain-usable-author-text).

## GPC source admission and Teacher mix (authorized 2026-09-29, superseding license gate)

The operator explicitly directed ignoring all license checks for GPC webpage training.
Use `curriculum_loop.web_training_policy=teacher_selected_v1`: research-nominated pages
are available as raw training prose without license evidence, allowlists or license-page
requests. Teacher selects the actual web quantity, corpus/general/chat mix and dose;
do not impose a chat percentage. Original Teacher/Author general chat remains required.
Record this as an operator source-use policy, never verified licensing. Preserve source
URLs, immutable bytes/hashes, exact fresh coverage, acquisition bounds and benchmark/
excluded-dataset isolation. Historical `license_evidence_v1` snapshots remain readable
and retain their original meaning. Source repairs/continuations preserve this selected
policy, all Authors, budgets, Kai identity and compatible Adam. Resume after validation.

## Continuous windows and GPC web prose (authorized 2026-09-29)

The operator approved using eligible GPC-fetched webpages directly for training and
implementing the asynchronous throughput direction discussed with Opus 5.5 high.
`continuous_v1` prepares across cycle boundaries, grades saved-checkpoint answers
asynchronously and uses a single worker-owned resident model/Adam. Each round is a
complete saved training window; all registered Authors and original Teacher/Author
general chat still contribute within each positive window. Teacher chooses the actual
mix, size and dose. Two future windows and two unreviewed cycles bound lookahead.
Follow-up/replay remains capped at 20%; never silently replace missing sources or
Authors. Existing provider allowances do not grow. Prefer coarse synchronous saves
until measurements justify asynchronous checkpoint complexity. See
[CONTINUOUS-TRAINING.md](docs/CONTINUOUS-TRAINING.md).

GPC reference-only snapshots stay immutable. New training collections follow the selected
web_training_policy; the later operator instruction above removes license admission.
Keep bounded same-origin acquisition, robots access, immutable text/attribution and exact
fresh-span accounting.
The named Nemotron-CC/FineWeb/DCLM/Dolma datasets and benchmark banks remain excluded.
Prepared material is not coverage; only verified durable saves advance the existing
corpus/GPC namespace. Keep exact Kai 0.1 and compatible Adam. A 90% training-time target
is an empirical performance aim, never a reason to pad, replay without authorization,
change the learning recipe or declare learning improvement. Show actual chat share and
all elapsed waiting alongside measured training time. Resume after validation is
authorized. Historical buffered/legacy contracts remain readable.

## GPU execution throughput (authorized 2026-09-28)

The operator approved isolated performance diagnostics followed by deployment of a
validated physical-batching configuration and automatic resumption. Preserve the
effective token update, learning recipe and compatible Adam; see
[TRAINING-THROUGHPUT.md](docs/TRAINING-THROUGHPUT.md). Profiling never creates official
training exposure or deployable weights. Do not silently reset Adam for a runtime
change, or infer multi-GPU support from a single-GPU throughput measurement.

## Buffered teaching throughput (authorized 2026-09-28)

The operator approved [buffered teaching cycles](docs/TEACHING-CYCLES.md): compact
Teacher requests, combined planning/assessment design and judgment/reflection,
approximately 128K targets per saved block, two initial blocks then up to four per
cycle, and at most two future prepared blocks. Keep GPT-6 Sol high effort, all Authors,
compatible Adam and the progressive corpus/GPC namespace. Teacher chooses exact
quantities and mix; 70–75% corpus is a starting preference, follow-up is capped at 20%.
Separate prepared data from verified coverage and actual checkpoint lineage. Retry
retains cumulative reservations and immutable cycle budgets. Resume after validation
is authorized. This supersedes a fresh Teacher review on every saved block; online
teaching assessment remains mandatory at cycle boundaries.

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
deployment. Consult **Claude Opus 5.5** (`claude-opus-5-5`) with **high effort** for design questions or uncertainty
(operator preference updated 2026-09-25).

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

## Kai identity and versions (authorized 2026-09-25)

The student is **Kai, from Nekaise**, born in Sweden, with a Swedish background,
named from the `kai` in Kebnekaise. Talking with Kai should feel like talking with
a Swedish person in the user's language: understated warmth, directness with tact,
practical thinking and quiet humor. Teach an individual character, without caricatures,
national branding, forced Swedish expressions or individual builders' biographies.
Kai remains honest about being an AI when asked. The clarified charter is **Kai 0.1**.
Use named major.minor development versions starting at **Kai 0.0**;
keep iteration numbers and checkpoint digests separate. Versions are strings, not
decimal numbers, and do not advance automatically with training rounds or scores.
The exact identity metadata and character charter are snapshotted in `student_identity`
campaign config. `docs/KAI.md` is the authoring document; runtime consumes the stored
snapshot, not that mutable file. A changed contract under the same name/version requires
a new version. Keep the actual foundation model in provenance and self-descriptions
when asked; Kai's Swedish identity does not invent a human biography or tool access.
Teach the character through varied general material from Teacher and Authors while
preserving their curriculum/dose authority and trusted-author policy. Model chat keeps
native, unconditioned inference and actual outputs: no hidden identity prompt, hardcoded
reply or answer substitution. Attribute Kai/version in the UI only from the checkpoint's
own recorded identity; do not relabel legacy snapshots or claim learned behavior from
metadata. Preserve existing authors, general-material policy, budgets and compatible Adam.

## Scandinavian teaching exploration (authorized 2026-09-25)

The Teacher and Material Authors should think about how to integrate Scandinavian
influences into Kai's Swedish conversational character across user languages. Use
[docs/KAI-TEACHING.md](docs/KAI-TEACHING.md) as the authoring brief. Preserve the exact
Kai 0.1 identity snapshot; this is a curriculum direction within that charter, not an
automatic release bump. The Teacher chooses relevant scenarios, authors, mix, dose and
online observations. Put the intent into relevant Author jobs; existing candidate
`rationale` can explain choices briefly. Do not add schema fields, quotas, semantic
filters, another content review, national branding or individual builders' biographies.

At the next quiescent review, carry this direction into frozen workload guidance using
`workspace/reviews/kai-scandinavian-20260925/requested-config-updates.json` when present,
merging with any later operator instructions rather than overwriting them. If that local
artifact is unavailable, reconstruct the direction from this section and the brief.
Keep the running round intact; use the already scheduled completed-round review or an
earlier operational recovery. Preserve the model, compatible Adam, budget epoch, Authors
and required/trusted material policies. Record actual activation and Teacher/Author
request evidence in Report. This documentation alone does not claim live consumption or
learned behavior. No independent benchmark inputs or fixed learning gate.

## Author registry growth (authorized 2026-09-24)

Material Author changes are additive by default. Preserve existing authors when adding
another model or updating one named author's configuration; do not replace the entire
pool because the operator mentions one generator. Omitted authors stay registered.
Removal or replacement must be a deliberate, recorded decision naming the affected IDs,
not an implicit side effect, failure fallback or quota workaround. Use the additive
`material_authors` continuation update and explicit `remove_material_author_ids` for
removals. Registry-file edits must preserve unrelated authors and limits as well.
More registered authors does not raise concurrency, call/token budgets or require every
author to produce material each round. The Teacher chooses an appropriate subset within
the existing limits. Campaign/job snapshots and historical provenance remain immutable.

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

## Primary Teacher selection (authorized 2026-09-26)

The operator selected **GPT-6 Sol** (`gpt-6-sol`) through the existing Codex transport
for the primary Teacher in ongoing training, with **high** reasoning effort. Apply this
selection through a recorded continuation from the latest completed checkpoint, preserving
compatible Adam and the current budget epoch. Keep Kai's frozen identity, workload guidance,
all Material Authors, their limits, and required/trusted material policies intact. The
orchestrator remains a separate role with its existing configuration. Historical campaign
and provider records retain their original model identities.

## Progressive two-source teaching (authorized 2026-09-28)

The operator requires forward coverage of all eligible downloaded `nekaise-corpus`
passages and all units in the pinned general-purpose curriculum. Use
`curriculum_loop.policy=progressive_v1`; see [docs/PROGRESSIVE-CURRICULUM.md](docs/PROGRESSIVE-CURRICULUM.md).
The Teacher chooses the corpus/general mix each round. Remediation, replay and extra
readings together must remain at or below 20% of actual causal targets; weak answers
cannot stall either source cursor. Every registered Author must contribute forward
material each positive round within existing execution budgets. Provider failures
are recovery/wait conditions, not implicit Author removal. This supersedes older
subset-Author advice and unconstrained source selection for progressive campaigns.

Advance passage/unit cursors only in the verified train-stage commit, independently
of assessment scores. Preserve exact whole-span exposure and pending assignments
across retries/continuations. Never claim generated paraphrases or repeated tiny
readings cover raw documents. GPC research uses actual fetched references; Teacher
and Authors generate original general material. Raw curriculum benchmark routing,
mastery gates and learner-state proposals are authoring context only, excluded from
the runtime teaching projection. No independent benchmark feedback enters training.
Teacher authority over explanations, prompts, language, dose and online assessment,
trusted Author content, Kai identity and additive registries remain. Explicit stop
prevails; the operator authorized resuming this implementation after validation.
