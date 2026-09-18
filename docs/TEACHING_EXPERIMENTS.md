# Teaching experiments and strategy versions

Studio's **Experiments** section is a read-only register across all runs and
continuations. It connects a Teacher's pre-training hypothesis to actual work,
online assessment and the Teacher's later interpretation. Archived, interrupted
and unsuccessful investigations remain accessible. Historical rounds without an
explicit plan are not backfilled with inferred intent.

This is the first implementation inspired by [Dream-RSI](https://arxiv.org/abs/2609.14858).
It does not implement matched-start A/B execution, alternative checkpoint selection,
historical policy simulation or an automatic teaching-score objective.

## Teacher contract

`Curriculum.experiment` is optional. A non-null plan supplies:

- `title`, `hypothesis`, `intervention`: what is being investigated and changed.
- `budget_basis`: the intended comparison axis, or why budgets are unmatched.
- `observation_plan`, `reconsider_if`: evidence to inspect and what would change
  the Teacher's judgment. These describe intent, not host-enforced thresholds.
- `related_round_ids`: any earlier recorded rounds, including other campaigns.
- Either `strategy_version` (a full existing hash) with `strategy=null`, or a
  `strategy` definition with `strategy_version=""`.

A strategy definition contains `name`, `approach` and `parent_version`. The parent
is a previously committed strategy hash, or empty for an original definition.
Its version is the content hash of the canonical artifact
`{kind:"teaching_strategy", schema_version:1, definition:{...}}`. It contains no
round IDs, timestamps or experiment-specific results. Identical definitions reuse
their version. Any definition change, including parent provenance, changes it.

The plan is saved by `select`, before this round's student attempts or training.
`Reflection.experiment_review` separately records `status` (`ongoing`, `concluded`
or `abandoned`), `findings`, `limitations`, `next_action` and optional
`evaluation_ids` from this round's completed grading. Null means no interpretation
was recorded; it is not inferred to mean ongoing, successful or failed. Negative
and null findings remain Teacher judgments with evidence attached.

Reference validation checks existence, type and chronology. It never evaluates a
hypothesis, prescribes the curriculum, requires a fixed test panel, or selects a
checkpoint by score. Content strategy changes need no Python patch. Changes to
Python, fixed prompts or the handbook still require the source lock and a fresh
continuation when fingerprints differ.

## Persistence and provenance

The immutable `select` artifact owns the resolved experiment card. It records the
strategy version, input checkpoint path, available parent training artifact,
campaign config hash and select input fingerprint. The path is not presented as
an independently measured checksum of a new external Base input. Existing worker
checkpoint validation and subsequent training/assessment provenance still apply.

`freeze`, `train` and `adapt` link the completed select artifact. Training checks
that the frozen link matches. Final material selection and preparation remain
separate artifacts: a plan describes intent, while actual recipe and consumption
are measured later. A card does not imply that every intended intervention was
executed exactly as planned.

The `records` table's `kind=experiment` projection is indexed and committed in the
same SQLite transaction as select completion, after artifact persistence. It
contains browse metadata and the select stage/artifact identity. Retrying later
stages cannot replace it. An unsuccessful select attempt may leave an unreferenced
artifact, but never publishes an experiment. Conclusions live only in completed
adapt artifacts, preserving the original hypothesis.

Missing index entries can be rebuilt from completed select artifacts:

```bash
.venv/bin/nekaise-loop reindex-experiments
```

Rebuild is idempotent; a conflicting existing entry raises an integrity error
instead of rewriting committed intent. This is an explicit maintenance operation,
not a Teacher tool or a dashboard mutation.

## Read interfaces

- `GET /api/experiments`: `limit` (1–100), `before` cursor, optional exact
  `campaign_id` or `strategy_version`. New inserts do not shift older cursor pages.
- `GET /api/experiments/{round_id}`: card, execution state, separate review,
  immutable stage references, actual work and per-question paired evidence.
  Existing rounds without a plan return null; unknown rounds return 404.
- Round detail includes `experiment` alongside the existing `learning_work`.
- The read-only Teacher archive exposes `experiments` (`limit/before`),
  `experiment` (`round_id`) and `strategy` (`version`). All history remains
  accessible. Curriculum receives one latest ancestor-context card as a shortcut;
  ancestor reads stop at the child's creation boundary and exclude sibling runs.

The dashboard supports older pages, strategy filtering, references back to the
owning iteration and links between parent strategy versions. No model is called
by these readers. Experiment readers never access the Bench projection or private
evidence. Independent benchmark results stay outside Teacher, Report, recovery
and checkpoint decisions.

## Interpretation and limitations

- The observation plan is saved before training. Actual online questions are
  still authored after training and frozen before answers. This is **not** a
  pre-training frozen test or a matched-start A/B trial.
- Per-item deltas come only from completed immutable grading with changed weights
  and matching recorded generation conditions. Missing grading, absent comparison,
  unchanged weights and incompatible conditions are distinguished. Mutable record
  projections cannot manufacture a completed comparison.
- Different rounds may use different questions. No cross-round mean gain or
  strategy learning leaderboard is calculated.
- Trained targets include passes and retry attempts. Teacher usage may be missing
  or partial; Material Author usage is shown separately. Efficiency remains a
  throughput diagnostic, not evidence of learning or a decision threshold.
- Sequential rounds carry prior weights and optimizer state. Strategy labels and
  chronology improve traceability, but alone do not identify causal effects.
- A hypothesis can be revised in a future plan. Earlier cards, null findings and
  abandoned investigations remain unchanged and readable.

## Validation (2026-09-18)

The integrated change passed 315 Python tests, 60 Node tests and syntax checks for
all 11 dashboard modules. A real headless browser exercised experiment pagination,
strategy filtering, iteration links and 1440px/390px layouts against explicitly
labelled fixtures, with no uncaught JavaScript errors or mobile horizontal overflow.
These checks do not claim student learning results.
