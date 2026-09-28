# Forward corpus and general curriculum

The optional [buffered cycle path](TEACHING-CYCLES.md) preserves these cursors but uses
an explicit Teacher raw-target allocation and several saved blocks per assessment.
Future windows remain speculative until verified training commits; they never count as
coverage during preparation. The legacy ratio-based sizing below remains readable.

The September 28 operator direction adds two independent exposure cursors to CoAPT
Mid-training. Assessment still informs teaching, but poor answers cannot hold the
model on one topic. The Teacher chooses the forward corpus/general mix each round;
adaptive remediation, replay and extra readings together occupy **at most 20%** of
prepared causal targets. Complete passes preserve that bound in completed exposure.

`curricula/general_purpose_curriculum.json` is the exact operator-supplied authoring
file, originally `/home/zengp/Code/GPQA/LLM_CURRICULUM.json`. Its SHA-256 is
`56e047acde55c0e8bf4eac5df9207deba770fa5c554e2d3a398262f74a109c5d`.
It describes 20 domains and 265 units, with concepts, objectives, suggested searches,
practice and teaching strategies. It is a curriculum map, not a ready-made training
set. Its proposed mastery gates, benchmark profiles, baseline requirements, learner
state and benchmark feedback routing are **not** the Studio runtime contract.

`general_curriculum.project_curriculum` uses a field allowlist to extract teaching
units in the complete `learning_order`. Import snapshots this projection into ignored,
content-addressed storage. Campaign config pins that artifact. Runtime never rereads
the mutable source JSON. No benchmark items, answers, results or acceptance gates are
imported into training. Tool/agent subjects produce text lessons only; this does not
implement agentic execution or CoAPT Post-training.

## Corpus exposure

A pass inventories every eligible, downloaded document in the corpus's published
training view, directly from authoritative manifests. Collection-only restrictions
retain their existing meaning. Ineligible and non-successful entries are counted as
excluded, never counted as trained. Stable hashed ID ordering spreads publishers and
topics through the pass. Corpus is read-only; its independent downloader is unchanged.

The ignored SQLite inventory is content addressed, records each manifest's hash and
stamp, and supports indexed passage discovery. An unchanged shard uses its inventoried
metadata; a changed shard is checked through the normal authoritative reader. Document
bytes must match the corpus hash before being snapshotted. Once a document is opened,
its verified text is pinned until its last span. New IDs and later document revisions
enter a subsequent pass. Missing, empty, restricted or mismatched data causes a visible
operational failure, not a silent skip or false completion.

The worker offers a bounded passage window (initially 262,144 characters, spans of
2,400 characters). Freezing includes a **contiguous prefix of whole spans** sized to
the Teacher's `forward_corpus_share`. It does not crop passages to make a ratio or
repeat a small passage to fill a stream. Every source character in that prefix is
represented by the serialized full-text targets; generated explanations do not count
as raw corpus coverage. Whitespace and short document tails are retained. Token
chunking overlaps one conditioning token and preserves all causal targets, including
EOS. Historical sampled/repeated readings are not retroactively credited as coverage.

The span budget bounds preparation, not the lifetime corpus frontier. Unconsumed spans
remain at the cursor. A very large requested corpus share can exhaust the offered
window; the frozen receipt reports that shortfall and the actual share. Whole-span
rounding also changes the actual share. These are recorded composition differences,
not learning gates. The Teacher can adapt its next package from those observations.

## GPC research and every Author

Each positive round teaches the next unit. After the last unit, another exposure pass
starts. Unit completion means delivered training exposure, **not mastery**. A unit may
cover more material in a later pass without blocking other units now.

Before planning, the Teacher searches the live web for that unit and selects up to
four public primary HTML/text sources. Codex research uses `web_search="live"`; Claude
research enables WebSearch/WebFetch. The Codex option follows the
[official configuration reference](https://developers.openai.com/codex/config-reference/). The host fetches pages, bounds time/bytes and
redirects, rejects non-public destinations, and records URL, time, response-content
hash, extracted-text hash and exact bounded excerpts. Unsupported pages or failed
requests are recorded. At least one real fetched source is required. DNS validation
is an application-level check, not a separate network sandbox. Retrieval is reference
research, not permission to bulk ingest named general datasets or benchmark banks.

The saved research plan survives retries. Successful research is reused for the same
pending assignment, including continuations. An all-failed fetch exposes its artifact
and saved plan for orchestrator diagnosis/retry; it never invents successful research.
Authors receive the actual unit and source excerpts in their job snapshot. They write
original examples, explanations and native chat responses, not copied exercises.

Every registered Author must have a forward corpus/GPC job and nonzero prepared
forward targets every positive round. Current Authors remain GPT-6 Luna, DeepSeek
Flash and Kimi K3. Teacher and Authors both contribute GPC material. No model is
silently dropped or substituted after a failure; provider quota waits and normal
orchestrator recovery remain. Growing the registry preserves other Authors. If the
pool exceeds existing call/output capacity, an explicit budget/pool decision is
needed; registry growth does not silently increase concurrency or allowance.

`learning_track` distinguishes `corpus`, `gpc`, and `remediation`; `material_scope`
continues to describe domain/general prose/general chat. GPC rows also identify the
assigned `curriculum_unit_id`. Author metadata is inherited from the job. Candidates
do not add schema keys. All trusted batches are preserved. Remediation above the cap
is an execution error requiring a revised package, not permission to prune material.
All replay and extra readings count toward this cap regardless of original scope.

## Saves, recovery and inspection

`curriculum_progress`, `curriculum_assignments` and `curriculum_receipts` keep durable
state by namespace. An assignment is frozen before Teacher calls. A continuation uses
the same pending window and projection, or the next committed cursor after a save.
Only the transaction committing a verified train stage can advance progress. Its
receipt checks complete row targets, all Authors, the assigned GPC unit, raw-span
identity and completed trainer token/update counters. Checkpoint files are verified
before that transaction. Failed preparation/training and zero-pass diagnostics give
no credit; a later assessment failure does not undo completed exposure. Repetition
changes exposure, not unique source coverage. Step-limited partial passes are disabled
for this policy (`train_steps=0`); Teacher-chosen complete passes remain available.

The new preparation worker uses the existing serialization and trainer. It changes
composition without changing loss masking, optimization or Adam compatibility. API
processes still import no Torch/Transformers. Python/prompt changes use a fresh
continuation. Namespace/parent progress rejects stale double commits; intentionally
branching from old weights needs a new explicit namespace, not inherited future credit.

Studio displays verified corpus characters/document completions, GPC unit/pass,
latest actual shares and participating Authors. `/api/campaigns/{id}` exposes the
same `curriculum_progress` aggregate. It is separate from independent benchmark cards.
Historical campaigns without this policy keep their original readable artifacts.

New general-material campaigns created through Service default to a pinned progressive
curriculum. Historical config snapshots retain `curriculum_loop=null`. To prepare an
explicit continuation update:

```sh
.venv/bin/nekaise-loop import-curriculum curricula/general_purpose_curriculum.json
```

The command snapshots the teaching projection and prints `curriculum_loop` and
`train_steps=0` updates. Apply through the ordinary Service continuation operation
while quiescent (`nekaise-loop continue CAMPAIGN --updates updates.json` starts it),
preserving the source lock, identity, Author pool and compatible Adam.
Operator stop/pause always overrides automatic execution. Independent eval remains
asynchronous and never feeds these cursors, recipes or recovery decisions.
