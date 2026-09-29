# Continuous CoAPT Mid-training windows

The September 29 operator approval allows eligible webpages found through the GPC
curriculum to train directly. Original Teacher/Author conversations and explanations
remain necessary. The named Nemotron-CC, FineWeb, DCLM and Dolma datasets are still
excluded. This direction was discussed and reviewed with Claude Opus 5.5 high effort.

## Execution and teaching authority

`teaching_cycle.policy=continuous_v1` reuses the durable progressive round contract;
each round is now a larger complete **saved training window**, rather than an assumed
128K-target block. Teacher chooses each window's source quantities, exact targets,
passes, Author jobs and assessment. Window sizing is a teaching decision informed by
measured supply and speed. A several-million-target window amortizes process/checkpoint
cost; the configured target is a suggestion, never a minimum, padding rule or mastery
gate. Source exhaustion is explicitly reported. Original general Teacher and Author
material, native chat and every registered Author remain required per positive window.
Existing Author call/output reservations and cumulative retry envelopes are unchanged.

`continuous_engine.py` runs three coordinated paths:

- CPU/network preparation creates immutable research, plans, Author jobs and tokenized
  datasets ahead of GPU consumption. At most two future windows prepare or wait.
  Preparation can cross a cycle-review boundary; at most two cycles remain unreviewed.
- One sequential consumer binds each prepared dataset to its actual saved predecessor,
  trains it, verifies the save and atomically commits the existing corpus/GPC receipt.
  No future model parent is invented while preparing. Every complete pass retains the
  existing update boundaries, causal objective and bounded remediation policy.
- At each cycle's last window, exact frozen assessment prompts are answered on that
  named saved model. Teacher grading/reflection runs asynchronously while the next
  authorized window trains. No concurrent second GPU model is needed. Historical
  comparison requests explicitly release residency and use the established sequential
  comparison path; the next train reloads compatible Adam.

Teacher feedback affects the next unfrozen plan. Frozen paid work is not edited in
place. A Teacher pause/complete takes effect at a saved-window boundary; operator pause
or stop cancels owned work immediately. This permits bounded feedback delay, not a
claim that pending assessment has succeeded. Both forward loops continue independently
of low scores. Follow-up, replay and extra readings together remain at most 20%.

Routine orchestration drains authorized preparation and saved windows before yielding
under the existing source lock. A production failure lets already prepared windows
finish and preserves the exact failed stage for the recovery orchestrator. No automatic
Author substitution, budget renewal or source reallocation occurs. Code repair still
requires quiescence and an exclusive source lock, followed by a continuation. Concurrent
live-tree repair and independent producer deployments are not implemented.

## Resident training and durability

`training_execution=resident_v1` keeps one worker-owned model and Adam in memory.
`trainer_sessions` records PID/start identity, session status and campaign. Startup
reaps an interrupted owned session using that exact identity. A parent-death signal,
request deadlines and bounded cancellation prevent a forgotten GPU holder. HTTP never
imports Torch/Transformers. Tokenizer preparation remains a separate CPU subprocess.

The resident worker calls the same microbatch collation, token-weighted backward pass,
clipping and Adam update functions. Each complete window still flushes its final short
update. Model training mode and cache settings are restored after online generation.
The exact previous batched runtime and numerical-function source fingerprint, unchanged
legacy serialization/recipe fingerprint, parameter layout and library versions govern
the execution-only migration. Changed or unreviewed state is rejected, never silently
reset. Named Kai identity and global optimizer/token counters carry forward.

Initial implementation uses **synchronous coarse saves**. A save still atomically
publishes weights, tokenizer, Adam, counters, dataset binding and file hashes. Only a
verified save followed by the receipt transaction advances coverage. Crash or cancellation
mid-window loses unsaved updates and reuses the frozen dataset from a stage boundary;
there is no mid-window resume. No next window starts before the preceding save commits.
Asynchronous CPU snapshots are deferred until measured save cost justifies the added
memory and consistency complexity. Resident state alone is never durable exposure.

Large preparation results travel through hashed local result files instead of giant
stdout lines. Full content-addressed datasets remain in ignored workspace storage;
small derived preparation artifacts serve UI/accounting requests without repeatedly
parsing token arrays. No training sources or weights enter Git.

Checkpoint retention remains the orchestrator's inspected, explicit, journaled decision.
This mode introduces no automatic count/age/score deletion rule. Current resumption and
active readers remain protected. Large windows reduce save frequency; low headroom still
requests orchestration. Independent benchmark scores remain outside every decision.

## GPC web prose

`curriculum_loop.web_training=true` enables a separate explicit source-use contract.
A research source may carry `training` permission: an accepted license identifier,
license URL, exact evidence quote, scope rationale, and optionally a same-origin path
prefix with a bounded page count. Public accessibility alone is not admission. The host
retrieves and snapshots the licensing evidence and checks that the supplied quote is
present; Teacher is responsible for accurately interpreting its scope and third-party
exceptions. No legal determination is inferred from a domain's historical reputation.

Collection retrieval honors snapshotted robots instructions and per-host pacing, public
HTTP(S)/DNS restrictions, redirect/content/time/byte bounds, the licensed path boundary
and excluded dataset locations. The initial path supports HTML/text pages and bounded
ordinary page-link traversal (up to 256 visited pages per nominated collection, with a
64M-character collection bound). It is not a PDF, JavaScript-rendering, sitemap or bulk
export crawler. A complete collection means this bounded acquisition finished, not that
every page on a site was downloaded. Failed/excluded pages remain recorded. Cached
source work survives retries; sources are not refetched on every plan.

Readable-body extraction removes navigation/footers and retains main/article text and
preformatted indentation. It is deliberately simple and does not promise semantic
cleaning or near-duplicate removal. Exact extracted-text hashes deduplicate pages. Full
text, extractor version, URL, source/content hashes, license evidence and robots snapshots
are retained separately from the short Author reference excerpts. Existing
`research_reference_only` artifacts are never relabeled or retroactively trained.

Teacher chooses `curriculum.web_target_tokens` independently of
`curriculum.raw_target_tokens`. Preparation includes a contiguous prefix of whole fresh
web spans; no cropping, artificial repetition or corpus backfill is used to meet the
estimate. Web rows have `learning_track=gpc`, `material_scope=general_prose` and a raw
source stream, so they cannot impersonate original Teacher/Author targets or chat.
Exact source offsets are stored in a content-addressed coverage map referenced by the
existing namespace. Only verified training updates that map. Later units reusing the
same exact text receive only its unread suffix. Exhaustion is a supply shortfall, not
permission to pretend replay is fresh coverage. Ordinary GPC unit completion still
means exposure, not mastery or complete coverage of every possible related website.

## Observability and practical limits

Studio shows GPC web source characters, per-window raw/web prepared targets, prepared
supply, saved general-chat share and measured training time divided by **all elapsed
wall time since the continuous lineage began**. Startup, production, saves, recovery
and operator holds remain in the denominator. Saved target throughput is separate.
Uncommitted update-loop time is displayed as observed work, never counted as retained
training. The metric describes the measured optimizer loop, not instantaneous CUDA
utilization, GPU allocation or learning quality.

90% is a measurement target. It requires sustained source and Author supply as well as
an efficient consumer. A finite queue does not prove sustainable utilization. Website
supply is bounded, licenses vary, and generated chat can remain the limiting input.
Teacher sees actual chat proportions and student responsiveness and can choose smaller
windows or another teaching mix. No automatic quality-score gate is added. Long-term
claims require an observed continuous interval that includes starvation and recovery;
a short isolated diagnostic demonstrates execution only.

Validation evidence and concrete rollout observations are recorded in
[CONTINUOUS-TRAINING-VALIDATION.md](CONTINUOUS-TRAINING-VALIDATION.md).

In continuous mode, Model chat follows the latest verified completed **train stage**
without waiting for asynchronous Teacher grading. Its reply still identifies the
exact immutable checkpoint and runs on CPU. Historical modes retain their previous
completed-round selection rule. A pending review never controls checkpoint eligibility.

The GPU window has a separate `train_timeout_seconds` execution budget. Increasing it
does not extend Teacher/Author deadlines or call/token allowances. Teacher call reservations
are atomic across planning and grading. Source admission failures are recorded before
planning so the Teacher can select a valid recipe explicitly; no failure is disguised as
successful ingestion. Publisher evidence must identify the claimed license, and NC/ND
prefix matches or unrelated generic license deeds do not grant training permission.
