# Independent progress display

The user authorized coordinated development in Studio and `../nekaise-bench` on 2026-09-17.
The purpose is a comparable long-term observation history under fixed questions and evaluation
conditions. Bench owns scheduling, scoring, paired comparisons and private evidence. Studio
displays only a typed aggregate allowlist on its dedicated benchmark surface.

The Studio card opens a history panel with full pagination, a score-independent overview,
metric selection and per-observation details. The panel pins an immutable history snapshot
while live polling continues; a visible refresh action adopts new observations. Its paired
comparisons explicitly describe the latest observation in that snapshot against the root
baseline and an earlier declared milestone. Ordinary samples never substitute for milestones.

`GET /api/campaigns/{id}/benchmark` supports projection versions 1 and 2. Version 2 adds a
sealed history reference and descriptive comparisons. The separate read-only history endpoint,
`GET /api/campaigns/{id}/benchmark/history?history={sha256}&page={index}`, verifies the content
hash, campaign, series identity, order and typed fields of each aggregate page. Neither route
reads raw questions/responses or writes training events. They are excluded from ordinary Report,
teacher tools, recovery inputs, checkpoint decisions and automatic training acceptance.

Retained-weight training tokens are the horizontal coordinate, not campaign-local iterations
or total expenditure. Only comparable measured observations are shown; missing evaluations are
not zeros. Pair intervals and gained/lost counts describe these tasks, not proof of a particular
modification's effect. A causal claim still needs a common starting point, matched training
budgets, predeclared comparisons and independent confirmation in Bench. No new confirmation bank
or experiment launcher is implemented by this display. See Bench's `docs/LONGITUDINAL_PROGRESS.md`.

Deploy Studio Python changes under the exclusive training source lock after a command-queue
maintenance boundary. Resume through the orchestrator's continuation decision. The display does
not change the trainer implementation or recipe. Preserve Bench's actual evaluation source and
runtime identity separately when upgrading its observer; never rewrite old protocol identities.

Validation on 2026-09-17: 282 Python tests and 47 dashboard Node tests passed. Headless Chromium
exercised the complete app with fixture-only data: historical pages, metric selection, observation
details, overview and desktop/mobile layouts. Deployed read-only endpoints passed aggregate
schema, immutable-history, cache and Report-separation checks. The live browser also passed history
opening, metric changes, observation selection during loading, closing and Report navigation.
No benchmark values are recorded
here, and fixtures are not student learning results.
