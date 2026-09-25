# Independent GPQA Diamond display

Studio's overview includes GPQA Diamond alongside the building-energy benchmark. Bench
owns data, inference, grading and private evidence. Run from the sibling Bench repository:

```bash
python -m nekaise_bench gpqa --model latest --studio ../nekaise-studio
# Or: --model /absolute/path/to/a/local/HF/checkpoint
```

Use the ML Python environment; `--limit 2` is a smoke check and never publishes a full
198-question score. See [Bench's protocol](../../nekaise-bench/docs/GPQA.md) for complete
options, dataset attribution and limitations. Runs are command-driven, not automatically
scheduled after training. Score changes do not control checkpoint or teaching decisions.

`GET /api/benchmarks/gpqa-diamond` reads only a bounded, allowlisted JSON catalogue from
`../nekaise-bench/workspace/gpqa/projection/gpqa-diamond.json`. Override the directory with
`NEKAISE_BENCH_GPQA_PROJECTION_DIR`. No Torch imports, subprocesses or training database
writes occur in this endpoint. Unsupported protocols, inconsistent counts and missing seals
reject the affected entries while preserving valid results; rejected-entry counts remain
explicit. Oversize projections and escaping symlinks fail closed to an unavailable display.

The card spans explicit model evaluations, independently of the selected campaign. It
shows the actual evaluated identity and date, correct/198, a descriptive interval,
invalid/budget failure counts and protocol details. It keeps the last completed result
visible during a new attempt; progress has no provisional accuracy. Lack of progress for
ten minutes is labelled explicitly, without pretending to establish process death.
The chart uses the original `openbmb/MiniCPM5-1B-SFT` starting model as its reference.
Its inference-file identity is pinned as
`bf0f16bf094d8942f90a53ecefb4749ec52dd60406988438bceb896dab89aec8`, verified
from Studio lineage and local snapshot revision `a60b37f1fc409c54e1e337b0723aaac6f92dfec0`.
The identity is configuration, never a hardcoded score: the horizontal dashed line
requires a completed aggregate for those actual bytes. Model labels cannot designate
a baseline. The baseline does not replace the latest non-baseline model's headline.

Only matching protocol IDs, dataset and token budgets share a chart or score difference.
Incompatible results remain in the table with an explicit comparison-unavailable note.
The chart has a fixed 0–100% scale, a separate 25% random-choice reference, model points,
and the measured baseline's interval. Pending/failed evaluations never draw a baseline.
Format and budget failures stay visible; this strict generative score combines science
performance with format compliance, and does not establish general-chat improvement.
Bench retains completed baseline references within its bounded aggregate catalogue.
This rendering uses the existing aggregate endpoint and changes no training Python or prompts.

The only code consumer of `benchmark_catalog` is the dedicated API route. Results never
join normal snapshots, teacher history, operational Report, recovery inputs or training
metrics. Read/display failures never gate training. Shared-user storage is not enforced
filesystem secrecy. Changes to this Python reader require Studio's ordinary source lock
and continuation boundary; the external evaluation itself does not pause training.
