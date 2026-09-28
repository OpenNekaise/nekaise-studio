# Progressive curriculum activation — 2026-09-28

Implementation `99aa194` was committed and pushed to `origin/main`. The operator chose
Teacher-controlled forward corpus/general shares, a 20% maximum remediation share,
and resumption after validation. Claude Opus 5.5 at high effort supplied the design
consultation. An optional second code-review call timed out after 600 seconds without
a verdict; it is not counted as a passed review.

Offline and host checks:

- Full Python suite: 678 passed before the final boundary regression additions.
- Final progression suite: 17 passed, covering full source partitioning, pass refresh,
  retries, continuations, failed assessment after a successful save, stale branches,
  zero-pass diagnostics, missing Authors, mismatched counters, the remediation cap,
  reference boundaries, inventory integrity and recovery-policy bypass attempts.
- Final core boundary run: 85 passed across progression, recovery, restoration and
  Teacher context. These checks overlap with the suites above; counts are not additive.
- Dashboard: 75 tests passed, with Node syntax checks and asset stamping.
- Actual corpus inventory: 1,617,782 eligible document IDs from 116 authoritative
  manifest shards; 14,771 entries excluded by the published training view/status.
  Metadata inventory and the first verified passage window took 26.9 seconds. This
  inventories the whole pass; document byte verification occurs when each is reached.
- Real HTTP retrieval from Python's official JSON documentation retained URL,
  timestamp, content/text hashes and the exact bounded excerpt.
- CPU preparation with Kai's actual checkpoint tokenizer produced 1,079 targets
  from explicitly labelled validation fixtures plus a real source passage. It loaded
  no model weights and made no claim of student training or actual Author generation.
- Training/serialization code hash and optimizer recipe matched iteration 61's
  saved checkpoint. API processes continue to import no ML runtime.

The authorized continuation is `campaign_26e0e616f972`, from
`campaign_9194ab90d4ba` and `round_43f2a9c58996/train-1/checkpoint`. Kai 0.1 identity,
Teacher `gpt-6-sol`, the existing Author registry and all per-round Author execution
limits were preserved. The ordinary explicit operator continuation renewed the
Teacher call epoch; its call allowance remains unlimited. The parent stop record
remains historical; the new continuation has no operator hold.

The first live round, `round_217bc3f65b10`, completed these real operations:

- Teacher web-search calls and four successful reference fetches: NASA requirements
  guidance, GOV.UK user stories, JSON Schema objects, and NIST SI units.
- Ten primary Teacher lessons and eight completed Author jobs: six GPT-6 Luna,
  one DeepSeek Flash and one Kimi K3. They returned 117 accepted Author examples;
  the plan had estimated 116. Complete trusted batches were retained.
- Teacher chose a 0.47 corpus share of forward targets and one training pass.
- The verified checkpoint records **16,381 targets and 8 optimizer updates**, with
  `optimizer_origin=inherited`. Full trainer counters matched the frozen traversal.
- Forward corpus material: 7,701 targets, including 7,288 raw-passage targets and
  413 primary Teacher domain targets. GPC: 8,589 targets. Remediation: 91 targets,
  **0.556%** of total exposure. Actual forward corpus share: **47.274%**.
- Forward Author targets: GPT-6 Luna 6,670; DeepSeek Flash 822; Kimi K3 439.
- Thirteen whole raw spans covered **31,200 new source characters**. The first large
  document is still in progress; no complete document was falsely credited.
- The atomic training commit advanced GPC from `foundations.task_contracts` to
  `foundations.evidence_and_search`, before online assessment had completed.

The new checkpoint is `workspace/runs/round_217bc3f65b10/train-1/checkpoint`.
Its completed training artifact is
`548d3950eb78a0a9a3e4f7e884f0d4a21b97bf119e9b20daf77e8b136a9e8105`.
The live dashboard endpoint returned HTTP 200 and exposed curriculum progression.
Training continued into ordinary online assessment. These observations verify
execution and exposure, not improved ability or an independent benchmark result.

Detailed operational evidence remains in ignored storage at
`workspace/reviews/progressive-curriculum-20260928/`, including import, consultation,
host checks, publication/activation, live plan, web references and training receipts.
The pre-existing uncommitted `docs/VALIDATION.md` was left untouched and unstaged.
