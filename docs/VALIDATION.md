# Validation record

Validated locally on 2026-09-14. This is an execution and observability check, not a claim
that the resulting student is useful or that its general capabilities improved.

## Repository promotion and directory rename

The active checkout is now `~/Code/nekaise-studio`, continuing the existing
`OpenNekaise/nekaise-studio` Git history from `6d5aa9f`. The old checkout remains at
`~/Code/nekaise-studio-bak`. A `nekaise-studio-loop` symlink preserves absolute historical
paths without rewriting immutable artifacts, database records or checkpoints.

The control-plane virtual environment was rebuilt at the canonical path. Dashboard and
supervisor units use that path and are active; the legacy campaign unit points to the
backup and remains disabled/inactive. All seven completed training artifacts passed
checkpoint manifest/file verification after the move. All seven campaigns retained their
prior lifecycle states. No new training or teacher calls were launched for the rename.

47 Python tests and seven Node tests passed from the new directory, along with JavaScript
syntax, systemd unit and lightweight API-import checks. Health, system, static assets and
preserved campaign APIs returned HTTP 200 on the existing Tailscale address. GitHub CI now
runs the current control-plane and dashboard tests; it does not launch teacher/GPU work.

## Teacher authority update

47 Python tests and 7 Node tests passed after the teacher-authority changes. Integration
coverage includes access to the earliest of 140 historical rounds from another campaign,
read-only archive queries, replay across campaigns, source selection outside the configured
prefix, exact source spans, teacher-defined prompts/training text, arbitrary task and
assessment counts, token shares/passes, omitted material, deferred assessment, durable
notes and teacher pause/completion. The teacher can also prepare nonempty datasets with
`train_epochs=0`, assess the student, and preserve weights without training metrics.

The Codex and Claude adapters are tested with mocked CLI envelopes for the local handbook,
archive command and strict output schema. The real CPU model probe still passes optimizer
continuity and token-weighted-loss checks, and records observed inference prompt truncation.
API imports do not load torch or transformers. JavaScript syntax checks passed; all four
view renderers accepted preserved real training records and the new planning records.

Real Codex `gpt-5.6-terra` curriculum checks use the actual archive and eligible corpus.
`campaign_18e2cc92e5ae` was stopped through the command queue after exposing a relative
corpus-path bug in continuations. Its failed attempt and logs remain preserved. Paths are
now normalized for continuations and teacher working directories, with regression tests.

`campaign_da706dd07cd3` completed curriculum selection and paused before student inference
or training. Its nine archive/source commands completed successfully. The teacher chose
three QA lessons against a suggestion of two, planned four assessment questions, and set
teacher/corpus/replay shares to 75/25/0. Its notes also exposed an ambiguity about
`train_steps=0`; the handbook now explains automatic update counting, and the explicit
teacher decision `train_epochs=0` disables weight updates. The original output remains
intact; changed instructions require a fresh campaign.

Final contract validation, `campaign_2c36ce067f20` / `round_015a07c3bb90`, completed a real
curriculum call with all 29 archive/corpus commands successful. The teacher selected three
QA lessons, three reading spans and two historical replay lessons, set token shares to
70/15/15 and one dataset pass, and explicitly described training as enabled. It planned
a four-question assessment; those questions have not yet been generated or administered.
The select artifact is complete and its campaign source fingerprint matches the deployed
implementation. The queued pause was consumed at the next stage boundary. There are no
student inference stages, training stages or optimizer metrics in this campaign.

The dashboard and supervisor user services are active; no learning worker remains running.
Health, system, assets and campaign APIs returned HTTP 200 at `100.123.76.107:8766`. All
four dashboard renderers accepted the final paused curriculum and preserved training
snapshot. Browser interaction and screenshots were not exercised in this update.

The maintained [CoAPT handbook](COAPT.md) is included in every teacher request and in source
fingerprints. Its [upstream snapshot](reference/COAPT-upstream-2026-09-14.md) matches the
read-only studio source byte for byte. No new student weight training was run for these
checks. The historical GPU results below predate this teaching contract.

Remote access follow-up: the enabled user service listens on `100.123.76.107:8766`.
Dashboard, JavaScript, styles, font, health, and campaign APIs returned HTTP 200 through
both that IP and `afk.tail5ec85b.ts.net:8766` from this server. Same-origin requests reached
validation and foreign-origin writes were rejected. All 17 Python tests passed after the
hostname/bind changes. Access from a separate tailnet device has not been tested here.

## Continuous-loop update

The continuous loop, trusted-teacher path, token mixture, recovery scheduler and separate
orchestrator passed 37 Python tests. Tests cover quota wait/backoff, stage reuse, manual
pause/stop, bounded repair attempts, exclusive source ownership, atomic continuation/start,
concurrent schema migration, source-patch rollback and fixture-based orchestrator decisions.
No real coding-agent repair session or unbounded production campaign was run for this update.

The actual ML worker was exercised with a tiny, local CPU-only GPT-2 fixture: two 2-update
rounds and one 4-update run matched parameters within 1e-7; global counters and Adam state
survived, weights stayed FP32, recipe changes were rejected until explicitly reset, and
unequal-length examples produced token-weighted loss. This validates implementation behavior,
not MiniCPM quality or long-term continual-training stability. It uses no teacher calls/GPU.

Five Node tests cover helpers, escaped waiting messages and the prepared/consumed token panel.
JavaScript syntax checks passed. The older real GPU runs below remain historical records;
they used the previous reset-each-round optimizer and evidence-gate behavior.

The updated dashboard and supervisor user services are active; the supervisor is enabled
for restart/reboot recovery. HTTP health, assets, system and schema checks passed on
100.123.76.107:8766, and all four view renderers accepted the preserved real campaign
snapshot. SQLite migrated to v2. Existing campaigns retain their previous complete/failed
states. The supervisor is idle; this update did not start a new training campaign.

## Original automated checks

- 16 Python tests passed, covering two-round execution, checkpoint lineage/integrity,
  evidence rejection, interrupted-stage retry, command conflicts, stale pause/stop
  consumption, owned-process cancellation, corpus eligibility, and local HTTP boundaries.
- Dashboard helper tests passed for text escaping, revision differences, and loss charts.
- Dashboard JavaScript syntax checks passed. All four view renderers accepted the actual
  completed campaign snapshot without missing-value or non-finite-number output.
- The local dashboard returned HTTP 200. Browser interaction, screenshots, and the optional
  browser-agent interface were not tested.

## Real student and teacher run

Campaign `campaign_8462c46ff65e` completed at 11:17:15 UTC with:

- Cached `openbmb/MiniCPM5-1B-Base`, PyTorch 2.10.0, Transformers 5.5.0, and an RTX 6000 Ada.
- Codex CLI teacher `gpt-5.6-terra`, using the existing authenticated CLI.
- Two rounds, each with two corpus lessons, two fresh online questions, and 12 optimizer
  steps. All 22 stages completed; all four teacher corrections passed the evidence gate.
- Ten teacher calls, 24 recorded optimizer metrics, and two verified checkpoint manifests
  and file hashes. Codex did not report a dollar cost; the dashboard does not invent one.
- Both prose CPT and QA-as-text examples. QA uses the current full-token causal-LM loss,
  not assistant-masked trajectory SFT.

The first evaluation identified coupled heat/moisture transfer and Kiva surface mapping
gaps. Round two selected a new ground-transfer passage using those terms and replayed one
earlier passage. Its `model_before` and checkpoint manifest point to round one's output.
Online references stayed outside student inference prompts. No fixed benchmark was used.

Round-one loss changed from 2.951 to 0.919; round-two loss from 6.733 to 1.736. These are
individual training batches, not held-out loss estimates. The two online scores were
0.125 and 0.25 on different questions and must not be treated as a benchmark improvement.
Peak reported GPU memory was about 22 GB. Restarting the web server during the run did
not interrupt the independent loop worker.

## Preserved failures

`campaign_a0e07fc99542` trained both rounds but stopped when Claude's account exhausted
its usage credits before the second evaluation. `campaign_e01c9cb24c0e` correctly refused
to train round two when evidence checks rejected both revised examples. Inspection found
extra quotation wrappers around exact evidence spans; the parser now removes one matching
wrapper only when the enclosed text is an exact source substring, with regression coverage.
Both failures remain in the workspace and dashboard. Teacher changes require an explicit
new campaign; there is no automatic provider fallback.

## Remaining scope

Indefinite production runs and real autonomous code repairs are not yet live-validated.
Cross-round optimizer continuity is implemented and CPU-tested; mid-stage resume is not.
Remote GPU execution, OpenCode, agentic trajectory SFT and probability-based on-policy
distillation remain unimplemented.
Their extension boundaries are described in [EXTENDING.md](EXTENDING.md). Benchmark
evaluation remains an independent future process with no feedback into this loop.
