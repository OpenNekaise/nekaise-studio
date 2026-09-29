# Independent MMLU-Pro display

The Studio card alongside GPQA consumes only Bench's allowlisted aggregate projection:
`../nekaise-bench/workspace/mmlu-pro/projection/mmlu-pro.json`, served read-only at
`GET /api/benchmarks/mmlu-pro`. Override the directory with
`NEKAISE_BENCH_MMLU_PRO_PROJECTION_DIR` for deployments. The file is capped at 256 KiB,
40 attempts, a pinned dataset and protocol, and exactly 14 validated subject counts.
Extra fields are discarded at both aggregate and subject level. Missing, stale or invalid
results cannot appear as zero scores. A full score requires all 12,032 verified responses.

Bench owns execution, evidence, the original MiniCPM5-1B-SFT reference, and the weekly
Sunday 09:00 Europe/Stockholm timer. This is zero-shot answer-choice likelihood, distinct
from MMLU-Pro's published five-shot reasoning evaluation. See
[Bench's protocol and CLI](../../nekaise-bench/docs/MMLU-PRO.md).
The card identifies that protocol, shows an exact-compatible starting-point horizontal
line, historical completed scores and subject breakdowns. No independent result enters
normal snapshots, teaching, Report, recovery or checkpoint decisions. GPU training and
all teacher/author policies are unaffected by benchmark scheduling.

Validation covers aggregate coherence, privacy allowlisting, no training-store writes,
real zero versus missing scores, stale/error progress, escaping, and protocol-matched
baseline comparison. Static assets are released together with the standard asset stamp.
