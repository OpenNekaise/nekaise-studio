# Buffered cycle rollout — 2026-09-28

Implementation `7338b6195a4bb8b00afe585cf691c7bf4c5153e6` was pushed to
`origin/main`. Campaign `campaign_681063fe7c4c` continues from the verified
`round_217bc3f65b10` checkpoint. GPT-6 Sol remains high effort; GPT-6 Luna,
DeepSeek Flash and Kimi K3 remain registered and participate in each positive block.
Kai 0.1, the Teacher budget epoch and compatible Adam were preserved.

Validation: the full Python suite passed with 693 tests; the final ten cycle tests
also passed, including two subsequently added cases (695 tests collected overall).
All 75 dashboard tests passed, followed by the affected five-test subset and syntax
checks after the final phase-label change. Compile checks and `git diff --check`
passed. These use fake providers and are not learning results. Two design consultations
with Claude Opus 5.5 high completed; an additional code-review call timed out without
a completed response and is not counted as a successful review.

## Actual first saved block

`round_7d0ce8b81d55` completed 44 updates and saved **89,525 causal targets**, compared
with 16,381 in the previous small round: 5.47 times more targets per save. This is not
a claim of 5.47 times better learning or end-to-end throughput.

| Observation | Recorded value |
| --- | ---: |
| Original corpus targets | 65,896 |
| Generated targets | 23,629 |
| GPT-6 Luna targets | 14,139 |
| DeepSeek Flash targets | 5,427 |
| Kimi K3 targets | 3,548 |
| Primary Teacher targets | 515 |
| Inner optimizer elapsed | 37.969 seconds |
| Complete train stage | 74.986 seconds |
| Overlap with block two's Author stage | 74.986 seconds |
| Reported GPU allocation peak | 21.985 GB |

The saved manifest records `optimizer_origin=inherited`. Its payload hashes were
verified after saving. Trainer fingerprint remains
`bec6927258b084448ca51167aaab6b7bd9fb61858e16810243576ba5acdf9cb2`.
Verified corpus coverage advanced from 31,200 to 294,898 source characters, with eight
document completions and two cumulative GPC unit exposures. Prepared future data did
not advance coverage. These are exposure counts, not mastery.

The first block reused its pre-existing, smaller pending corpus window. It exhausted
that window at 65,896 raw targets against the Teacher's requested 98,304. The new
1,048,576-character window applies to later assignments. The approximate 131,072-target
block size is a planning objective, not fabricated completion accounting.

## Recovery and current limits

Kimi call 3701 reached its 7,000-output-token limit and returned an incomplete response.
Report 515 investigated the terminal envelope, preserved 115 candidates from eleven
completed jobs and selected an unchanged-stage retry with saved completion feedback.
The retry completed in 42.690 seconds using 1,940 output tokens. The remaining Kimi
prose job also completed. No source repair, Author substitution, supplemental grant,
budget renewal or new Teacher plan was used. The first verified training save followed.

Research plus the two-block plan used 281,973 reported primary Teacher input/output
tokens. At this observation, block two was preparing and the mandatory cycle assessment
had not run. This is **not a final cycle efficiency ratio**. Author generation latency,
provider retries and synchronous Teacher boundaries remain real throughput limits.
The cycle review and subsequent orchestrator review must inspect final usage, actual
saved exposure and student behavior before drawing further conclusions.

Full artifacts and deployment evidence remain in ignored
`workspace/reviews/teaching-throughput-20260928/`, including `host-validation.json`,
`deployment.json`, `live-validation.json` and the timestamp-derived `live-overlap.json`.
Unrelated operator changes in `docs/VALIDATION.md` were preserved and not committed.
