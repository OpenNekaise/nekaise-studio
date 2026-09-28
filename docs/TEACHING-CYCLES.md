# Buffered teaching cycles

The September 28 throughput change separates a Teacher review cycle from a saved
training block. `teaching_cycle.policy=buffered_v1` uses progressive corpus/GPC coverage
and required, trusted Author expansion. Historical campaigns without this setting
retain their original stages and artifacts.

Initial settings suggest 131,072 causal targets per block, at most two blocks in the
first cycle and four thereafter (configurable up to eight). The Teacher may choose an
ordered prefix for earlier feedback. Approximately 70–75% forward corpus is an initial
preference, not an enforced ratio. The Teacher chooses `curriculum.raw_target_tokens`;
a small generated batch no longer shrinks raw passage allocation automatically.
Whole passage prefixes and all valid trusted Author rows are retained. Actual targets
and window exhaustion are recorded, without padding or metric-driven repetition.
Follow-up remains capped at 20%.

## Execution

1. `cycle_research` batches web research plans for upcoming GPC units. The host fetches
   actual references, saving plans and successful units for retry. Previously researched
   pending assignments remain reusable.
2. `cycle_plan` combines curriculum, exact Teacher targets and frozen online assessment
   prompts/rubrics. Each positive block includes every registered Author and general
   Teacher/Author material. Seeds have `student_observation=not_requested`; no current
   student attempt or errors are invented.
3. One producer executes Author, authorization and CPU tokenization stages. At most two
   unbound blocks may be preparing/prepared ahead of the GPU. Existing global/provider
   Author concurrency limits apply. Prefetch never crosses a review boundary.
4. A sequential consumer binds each dataset to its actual predecessor checkpoint,
   trains, verifies the save and atomically commits passage/unit exposure. Every block
   carries optimizer state. A future preparation failure allows the active block to
   finish, then sends the exact failed stage to the separate recovery orchestrator.
5. After the final block, the student answers the frozen assessment. One `cycle_review`
   Teacher call grades actual answers and records reflection/next instructions.
   Intermediate blocks have null scores; `latest_strategy` skips deferred reflections.

This normally uses three Teacher calls per cycle, instead of six per block. GPT-6 Sol
remains high effort. Compact requests contain current observations, production summaries,
source previews and exact retrieval references. The complete archive remains accessible;
this is not a recency cutoff or a content audit of trusted Authors.

Online assessments remain Teacher-owned. Independent benchmarks are excluded. Optional
`compare_before` compares the final block's input/output, **not the entire cycle**.
Legacy per-round experiments and historical GPU-scoring requests cannot execute inside
speculative preparation; their historical engine and archive remain readable.

## Durability and budgets

`teaching_cycles` stores config/source/tokenizer bindings, anchor, budget epoch and
research/plan/review artifacts. `teaching_blocks` links round identities, predecessors,
speculative assignments and a separate write-once parent binding. Unbound future rounds
have empty `model_before`; the tokenizer anchor is preparation provenance only.

Future windows follow the previous frozen coverage prefix. They do not claim the global
`(namespace, sequence)` until verified training commits. Before training, the consumer
compares the coverage frontier with committed progress, verifies the predecessor save
and compares actual tokenizer files with the pinned fingerprint. Invalidation never
advances coverage or deletes paid work. Retry reuses completed stages and Author jobs.
Code changes require a fresh continuation; there is no mid-stage optimizer resume.

Per-block Author limits remain unchanged. Cumulative cycle calls and reserved output
are capped at the lesser of the configured envelope and actual planned block count times
per-block limits. Unknown calls and reported output overruns count. The frozen epoch
survives Resume even if the primary Teacher epoch changes. Explicit recovery grants are
bounded at both block and cycle scope. No-op recovery continuations cannot renew budgets.

Studio distinguishes prepared targets, completed blocks and pending review. Aggregate
lineage efficiency retains actual trained targets and primary Teacher usage; Author
usage is separate. Individual block ratios are not directly comparable when shared
planning is charged to block one and review to the last. Missing usage is not zero;
more exposure is not demonstrated learning.

The trainer recipe, existing BF16 autocast and compatible Adam stay unchanged. CPU
preparation overlaps GPU training, but synchronous cycle planning/review and Author
generation still limit utilization. Larger blocks delay feedback and can allow more
drift before review, which is why rollout starts with two blocks.

`tests/test_teaching_cycles.py` uses explicitly fake providers to verify overlap, bounded
preparation, exact failure routing, stop/resume, no double coverage, tokenizer mismatch,
all Authors and nonrenewing retry budgets. Live evidence belongs in ignored
`workspace/reviews/teaching-throughput-20260928`.
