# Continuous training validation — 2026-09-29

Implementation and review evidence is retained in ignored
`workspace/reviews/continuous-training-20260929/`. Claude Opus 5.5 high effort
provided an architecture consultation and a read-only implementation review.
No benchmark scores or private benchmark questions informed these changes.

## Automated checks

- Application suite: **757 passed, 1 ML-module skip**, 81.7 seconds. The ML module
  runs separately in the actual PyTorch environment.
- Real ML numerical tests: **9 passed**, including two saved windows with online
  generation between them, activation checkpointing both on and off, and bitwise
  equality of parameters/Adam against save/reload. Tokenizer/model/generation
  configuration identities and generated token IDs also agree.
- Dashboard: **76 Node tests passed**, plus syntax checks.
- Integration covers cross-cycle preparation, asynchronous review, multiple blocks,
  exact predecessor binding, producer failure draining, pause/retry without double
  credit, diagnostic windows, consuming an explicit reset after a verified save,
  the routine-review drain, atomic Teacher call reservation, and request-tagged
  resident messages acknowledged after all metrics.
- Web tests cover license evidence, NC-prefix rejection, unrelated deed rejection,
  same-origin/path boundaries, cross-site redirects, explicit reference-only admission
  failures, cached exact-text deduplication, fresh suffix coverage and readable extraction.

## Actual GPU diagnostic

Parent: the verified `round_5789117e25bf` save, global target count **12,440,765**.
Two isolated windows each used the real frozen **126,809-target** dataset, with four
native online prompts between them (production batch size 4, maximum 192 new tokens).
These are diagnostic updates only. Their checkpoints stay outside official run lineage,
are not deployed and do not enter teaching exposure or efficiency totals.

| Observation | Window 1 | Window 2 |
|---|---:|---:|
| Load/check inherited state | 12.594 s | 0.00042 s |
| Update loop | 24.616 s | 24.731 s |
| Throughput | 5,152 targets/s | 5,128 targets/s |
| Save | 20.951 s | 20.684 s |
| Peak allocated GPU memory | 27.795 GB | 27.794 GB |
| Adam | verified inherited | same resident tensors |

The whole two-window diagnostic took **124.1 s**, including generation, saving,
verification and process lifetime. These intentionally small windows do **not** establish
90% wall-time duty. They show that residency removes repeated load cost while save/verify
cost remains material, supporting larger teaching windows and synchronous coarse saves.

The first diagnostic attempt correctly refused migration before updates: Python 3.13
and 3.12 produce different default AST dumps for the same source. The fingerprint was
changed to exact function source bytes, and validated in both environments. No Adam reset
or failed-run erasure was used to bypass that guard. The final bridge additionally pins
all reviewed trainer/load/generation implementation files, not only update arithmetic.

## Actual source diagnostic

A bounded, isolated collection retrieved three Python tutorial pages under the official
publisher's PSF licensing evidence: **62,300 extracted characters**, 27 whole spans,
zero retrieval failures. After a simulated source-offset receipt, the same text exposed
**zero fresh rows**. This was a retrieval/coverage diagnostic, not official training.
Full source, license and robots evidence remain in the ignored diagnostic directory.

The implementation review's claimed tokenizer-retention gap was checked against the
actual executor: records-only retirement deletes weights and Adam, while tokenizer,
model/generation configuration and manifests remain. No unnecessary permanent weight
retention rule was added. Existing resumption/reader protections remain in force.

## Limits and follow-up evidence

The sustained 24–72-hour training duty and supply rate are not established by these
checks. Studio reports the real elapsed denominator, retained tokens and actual chat
share after deployment. Queue starvation, admission failures and review/recovery time
remain visible. Coarse saves are synchronous; source preparation is bounded and may
still constrain throughput. A missing or restricted webpage remains reference-only
with a visible reason before the Teacher chooses a recipe.

A crash after checkpoint rename but before stage commit can leave uncommitted bytes in
an interrupted attempt directory, as in the earlier trainer. They remain evidence and
must not be credited or silently deleted; the next operational storage investigation
must identify these outputs explicitly. Diagnostic checkpoint bytes and their manifests
are retained separately for review. No automatic retention rule or unjournaled deletion
was introduced.

## Live rollout

Implementation commit `704da1c140916b09dd865c7cd83f06475f840d35` was pushed to
`origin/main`. The dashboard and supervisor were restarted while the source lock was
held. After release, command-queue action **808** started continuation
`campaign_3c9a381dd023` at **2026-09-29 09:20 UTC**. It inherits the verified parent
`round_5789117e25bf`, the exact Kai 0.1 identity, all three Authors and the original
Teacher budget epoch `2026-09-28T10:54:26.290+00:00`.

Deployment uses two windows per cycle, two-window look-ahead, a suggested 4,194,304
targets per window, a 24,000,000-character fresh corpus assignment and a separate
3,600-second GPU request limit. Teacher/Author timeouts and call/output reservations
are unchanged. Teacher may choose smaller rollout windows and remains responsible
for actual source mix and dose. Microbatch 4, 2,048 effective targets per update,
512-token context, activation checkpointing off and learning rate 2e-5 are unchanged.

The live API health check passed, and the dashboard returned the new measured
continuous-training projection. The first source research stage entered execution.
This startup observation alone is not evidence of a completed training window or
sustained 90% training duty. The ignored deployment receipt records source/runtime
fingerprints, the compatible optimizer transition and the actual command action.

## Operator-selected GPC admission — later September 29

The operator subsequently removed all GPC license checks and explicitly left source/
general/chat proportions to the Teacher. `teacher_selected_v1` makes research-nominated
pages available for direct prose training, without fetching or checking license evidence.
Legacy licensed-policy artifacts and checks retain their meaning. Acquisition bounds,
source hashes, exact fresh coverage, excluded datasets and independent benchmark isolation
remain; the policy does not claim a verified license.

Before editing, the fifth window's verified save, `round_80e9ffc8fe68`, was allowed to
finish at **27,624,423** global targets. Command-queue pause **814** and the exclusive
source lock followed that save.
The ML runtime and recipe hashes remain unchanged, and continuation preflight confirms
`identical_runtime` Adam inheritance. No unsaved GPU work was cancelled for this change.

- Full application suite: **762 passed, 1 ML-module skip**. No ML arithmetic changed.
- Web-policy regression suite: **14 passed**, covering absent metadata, restrictive or
  unreachable license metadata being ignored, source/cache provenance, whole fresh-span
  preparation and preservation of the selected policy during automatic recovery.
- Actual source diagnostic retried three previously rejected page nominations. Two
  LibreTexts pages supplied **12,112 characters** across six spans, with no license
  fetches. The third page could not be acquired because robots retrieval returned HTTP
  403; that remains a retrieval failure, not a license admission rejection.
- Kai's actual tokenizer prepared **2,664 web targets** from those pages. After simulated
  source-offset coverage, the same collection exposed zero fresh rows. These are isolated
  retrieval/tokenization checks, not official student updates or claimed training exposure.

Evidence is in ignored `workspace/reviews/web-source-policy-20260929/`. Current source
admission and Teacher instructions are described in `docs/CONTINUOUS-TRAINING.md`.
