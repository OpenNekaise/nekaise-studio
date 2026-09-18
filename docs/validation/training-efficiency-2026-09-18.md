# Required expansion and training efficiency — 2026-09-18

The user requires Material Author expansion on every actual training round, larger useful
measured training exposure, and a trained-targets / primary-Teacher-tokens metric available
to the teacher and plotted in Studio. Claude Fable 5.1 reviewed the design and code.

The default required_v1 policy rejects missing/invalid expansion plans before student work.
Freeze and train require a round-bound receipt for completed expansion and selected expanded
prepared targets. The teacher may reject everything and explicitly choose zero passes.
Provider failures keep the normal waiting/recovery path. Historical artifacts are unchanged;
test fixtures explicitly select legacy_optional where expansion is outside their scope.

The efficiency counter reuses existing normalized provider evidence, counts cache/reasoning
once, includes retries, and keeps author usage separate. Missing/pending usage is explicitly
partial. Reflection gets provisional round usage; the next selection gets completed usage.
The plot shows completed-round observations and a descriptive fit; the headline is the ratio
of lineage sums. All historical totals retain their meaning.

Validation before deployment:

- Full Python suite: 292 passed; then added ancestor-boundary coverage and reran the 16
  efficiency/telemetry/learning-work tests after review corrections, all passed.
- Dashboard: 53 Node tests passed; all modules passed syntax checks. Browser fixtures
  verified five cards, three descriptive curves and no mobile horizontal overflow.
- Read-only comparison against the live database confirmed the previous Teacher tokens and
  Tokens trained counters were identical before and after the change (270,475,202 and 695,339
  at the sampled observation). This is accounting evidence, not a learning result.
- The three optimizer-compatibility source files remain byte-identical, so this workflow/UI
  change does not itself require an Adam reset.

The deployment uses the command queue to pause at a stage boundary, takes the exclusive
source lock, and creates a fresh continuation from the latest verified training checkpoint.
The new workload guidance supersedes historical optional-delegation advice. No benchmark
questions/results enter the new teacher feedback or operational decision.
