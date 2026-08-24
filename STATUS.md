# STATUS — current phase and targets

> Durable procedure lives in `SPEC.md`, `BOUNDARY.md`, `skills/`, and `AGENTS.md`.
> This file contains only the active state. Last update: 2026-08-24.

## Phase: LINKEDIN EQUAL-SOURCE CoAPT-CPT A/B — COMPLETE

Human directive 2026-08-21 supersedes the unstarted two-round MVP recipe. The active
campaign is a compact public showcase comparing raw CPT with Codex-authored CoAPT-CPT on
**`openbmb/MiniCPM5-1B-Base`**. It is one R0 dataset pair and three paired training seeds,
not a scaling study and not a claim about larger models.

## Frozen design

Both arms start from the same base model, use the same frozen student-selected frontier
source-span pool, the same 20% anchor stream, the same trainer, and a **1.4M**
content-token cap:

- **Raw CPT control:** source chunks 80% + anchor 20%.
- **CoAPT-CPT:** source-grounded Codex teacher text 80% + anchor 20%; no QA text.
  The validated v4 teacher stream prepends the exact source sentence containing the
  first source numeric token only when the authored teacher's first numeric token
  differs.

Each unique source span is used at most once. The campaign uses 650 pool documents, a
500-document frontier, and up to 12 position-stratified source chunks per document. Only
the continuation draft type is authored, so the same source span is not counted twice as
continuation plus summary. The raw control reads the exact `source_chunk` records used by
the teacher arm. The successful campaign recipe files are frozen at:

- `workspace/configs/linkedin-ab-v4-coapt.yaml`
- `workspace/configs/linkedin-ab-v4-raw-control.yaml`

The corpus manifest snapshot is `workspace/corpus-snapshots/linkedin-ab-v1/`; every
selected cleaned file is verified against its manifest `corpus_sha256` before use.

## Measurement and decision

- Primary: `coapt_pool_absorption_dev`.
- No-regression guardrail: `corpus_transfer_macro_dev`.
- Paired seeds: **3407, 3408, 3409** on one immutable dataset per arm.
- The three-seed raw control establishes this campaign's noise band
  `max(std, half_range)`. Historical CPT noise bands are not reused.
- NLL, teacher gate yield, and training loss are diagnostics only; perplexity is never a
  success metric.

The public claim is limited to the measured 1B, equal-source, equal-token pilot. CoAPT is
called better only if its absorption gain exceeds the new noise band and transfer does
not breach the guardrail.

## Current state

- Training/eval doctor passes in the pinned Conda environments; RTX 6000 Ada is visible.
- Codex CLI 0.149.0 is installed and authenticated through ChatGPT, without an API key.
- `gpt-5.6-luna`, `gpt-5.6-terra`, and `gpt-5.6-sol` all passed real availability calls.
- The Codex author → separate Codex source-grounding gate completed an end-to-end smoke.
- The v1 4M recipe stopped at preflight: 5,385 unique source spans contain 2,935,792
  tokens, below its 3.2M domain-token requirement. No teacher corpus or training run was
  produced. The v2 1.4M cap was frozen from the bake-off's passed-token projection with
  margin; it requires 1.12M unique gate-passed teacher tokens.
- The 50-span bake-off selected **`gpt-5.6-terra` low** as author: 43/50 passed the common
  Sol-low gate versus Luna 29/50 and Sol 31/50. Terra was also fastest. Full authorship
  uses the measured batch size 8; larger batches compressed the output materially.
- Full CPU guardrail suite passes.
- S0 reference is recorded (`coapt_pool_absorption_dev=0.1278`,
  `corpus_transfer_macro_dev=0.1189`). The Sol-low gate passed 4,369/5,385 authored rows.
- The v2 full-teacher campaign improved pool absorption (0.14323 vs raw 0.13537) but
  failed transfer (0.11670 vs raw 0.12033). The v3 full-source fallback repaired
  transfer on its first seed but erased the pool gain, so the remaining v3 arms were
  stopped as a failed ablation.
- The single v4 change preserves the teacher text and prepends only the exact source
  sentence containing the first numeric token when teacher/source first-number order
  differs. Its immutable datasets are `7932c970db30b577` (CoAPT, 1,399,965 tokens) and
  `180dbe13e572c884` (matched raw control, 1,399,936 tokens).
- All three paired seeds completed. Mean `coapt_pool_absorption_dev` is **0.13990** for
  CoAPT versus **0.13443** for raw CPT: **+0.00547**, or 5.11x the newly measured raw
  noise band of 0.001069. Every paired pool delta is positive and above the band.
- Mean `corpus_transfer_macro_dev` is **0.11990** for CoAPT versus **0.11940** for raw
  CPT, so the no-regression guardrail passes. The public claim remains limited to this
  1B, equal-source, equal-token pilot.
- The full report, run IDs, checkpoint digests, and validation evidence are in
  `workspace/coapt/linkedin-ab-v4/r0/round_report.md`.

## Next actions

1. Treat v4 as the retained 1B pilot result; do not tune further against this dev split.
2. Before a broader effectiveness claim, repeat the frozen rule on a new corpus or model
   as a fresh campaign and consult milestone referees only at the prescribed phase gate.
