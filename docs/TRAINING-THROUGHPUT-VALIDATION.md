# GPU batching validation — 2026-09-28

Implementation `f54985666dc028c77b3f3c09d45420d42ed40428` was pushed to `origin/main`.
The selected configuration is `training_execution=batched_v1`, physical batch size 4,
and activation checkpointing disabled. Learning rate, 2,048 target tokens per optimizer
update, corpus/GPC progression, Authors, Kai identity and compatible Adam are preserved.

Continuation `campaign_09a29b1ad72c` was created from the verified checkpoint and
automatically started through command 725. The worker entered `cycle_research` with
no recovery incident; dashboard and supervisor were active. The existing Teacher budget
epoch (`2026-09-28T10:54:26.290+00:00`) was retained. This verifies activation and resumption,
not completion of the first new optimized teaching block.

## Measured throughput

One RTX 6000 Ada 48 GB, FP32 weights/Adam, BF16 autocast, maximum sequence length 512.
Every case reloaded the same verified `round_7d0ce8b81d55` checkpoint and optimizer.
The baseline reimplements the preserved serial loop, including its CUDA Adam step
placement and per-row synchronization. The new cases use the production batching code.

| Frozen data | Serial targets/s | Batch 4, checkpointing off | Ratio | Selected peak allocation |
| --- | ---: | ---: | ---: | ---: |
| Raw corpus rows | 2,898.8 | 5,108.7 | 1.76× | 27.770 GB |
| Actual mixed teaching block | 2,590.3 | 4,794.7 | 1.85× | 27.769 GB |

Raw timing used eight warmup updates followed by 64 measured updates and 127,696
actual target tokens. Mixed timing used eight warmup updates followed by 32 measured
updates and 65,536 targets. Some raw samples repeat during timing; these diagnostics
receive zero official exposure or coverage. Their updated weights are never saved.
The starting checkpoint's original payload hashes verified again after each profile.

Raw-profile whole-process time, including loading and warmup but excluding a checkpoint
save, fell from 58.439 to 41.883 seconds. These are short diagnostic samples, not a
measurement of sustained whole-corpus or complete Teacher/Author-cycle throughput.
The slower teaching and material-generation stages remain separate bottlenecks.

At 5,108.7 targets/s, 19 billion raw tokens imply approximately 43 days of GPU-loop
computation. This extrapolation excludes loading, saves, interruptions and Teacher or
Author work; the corpus token count itself is an estimate from characters, not a full
tokenizer census. No multi-GPU result is claimed: only one GPU is installed.

## Correctness and compatibility

The application suite passed **699 tests**, with the ML module skipped in the API
environment. **Five ML tests** passed separately, including uneven padded causal-loss
and gradient comparisons, EOS-as-padding, rejection before a nonfinite optimizer update,
checkpoint/Adam round-trip and rejection of a changed parameter-name layout.

Actual-model BF16 checks compare uneven real sequences and a complete 2,048-target update.
Batch 4 passed the preset absolute-loss difference limit of 0.01 and relative gradient
L2 limit of 3%: loss differences were approximately 0.00119 and 0.000728, and gradient
differences approximately 2.98% and 2.87%. Repeated serial controls showed approximately
0.34–0.53% gradient variability. This is numerical compatibility, not bitwise identity
or a claim that learning trajectories remain identical.

Batch 2 exceeded the preset gradient limit on the complete update (approximately 3.16%).
That failed candidate and the failed aggregate diagnostic remain saved; batch 2 was not
selected for deployment. No tolerance was relaxed to accept it.

The exact legacy trainer fingerprint remains
`bec6927258b084448ca51167aaab6b7bd9fb61858e16810243576ba5acdf9cb2`.
The new runtime fingerprint is
`0bea6517cd6252d6d5b9eb42f74117dba35bb480053f4c690a58f654ffb1d868`.
All loaded moments and step values compared equal to the source state. New manifests
also record parameter layout, runtime/library identities and the explicit transition.
Opus 5.5 high reviewed the design and implementation; its supported findings were
addressed, including reverse-transition reset protection, library/layout checks and
CPU placement of non-capturable Adam step scalars.

Ignored evidence lives in `workspace/reviews/vanilla-throughput-20260928/` and
`workspace/profiles/`. Final raw and mixed profiles are `training_dddda8a94fa8` and
`training_ce585c24f964`; the extended numerical profile is `training_ee1eaa834c02`.
`deployment.json` records the actual continuation, queued start, inherited budget epoch
and source/optimizer bindings. Diagnostic observations are not student learning results.
