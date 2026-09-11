# STATUS — current phase and targets

> Durable procedure lives in `SPEC.md`, `BOUNDARY.md`, `skills/`, and `AGENTS.md`.
> This file contains only the active state. Last update: 2026-09-11.

## Phase: LINKEDIN EQUAL-SPAN DETAIL-RESIDUAL A/B — COMPLETE

Human directive 2026-08-21 supersedes the unstarted two-round MVP recipe. The retained
follow-up campaign is a compact public showcase comparing raw CPT with Codex-authored
CoAPT-CPT on **`openbmb/MiniCPM5-1B-Base`**. It is one R0 dataset pair and three paired
training seeds, not a scaling study and not a claim about larger models.

## Frozen design

Both arms start from the same base model, use the same frozen student-selected frontier
source-span pool, the same 20% anchor stream, the same trainer, and a **1.4M**
content-token cap:

- **Raw CPT control:** source chunks 80% + anchor 20%.
- **CoAPT-CPT:** length-matched source-grounded teacher composition 80% + anchor 20%;
  no QA text. Each row retains the validated v4 teacher prose, then adds verbatim source
  detail prioritized by missing protected literals until it exactly matches its paired
  source span's token length.

Each unique source span is used at most once. The campaign uses 650 pool documents, a
500-document frontier, and up to 12 position-stratified source chunks per document. Only
the continuation draft type is authored, so the same source span is not counted twice as
continuation plus summary. The raw control reads the exact `source_chunk` records used by
the teacher arm. The retained campaign recipe files are frozen at:

- `workspace/configs/linkedin-ab-v5-coapt.yaml`
- `workspace/configs/linkedin-ab-v5-raw-control.yaml`

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
- The v5 equal-span follow-up diagnoses the remaining v4 confound: on matched
  Simulation spans, teacher text retained only 56.5% of source tokens and about 47.8%
  of number/acronym occurrences. Its topic-blind residual composition raises
  protected-literal macro recall to 98.33% while matching every selected source key and
  per-row token count across arms.
- The immutable v5 datasets are `26085e28149b52ff` (CoAPT) and `9daa5084d03e063f`
  (raw). Each has 2,073 paired domain rows, 227 domain documents, 1,119,990 domain
  tokens, the same 279,967 anchor tokens, and 1,399,957 total tokens.
- Across seeds 3407/3408/3409, mean pool absorption is **0.13610 CoAPT versus 0.13130
  raw** (+0.00480; every paired delta positive). Mean transfer macro is **0.12123 CoAPT
  versus 0.12103 raw**, so aggregate no-regression passes.
- On the exploratory 96-document Simulation union, CoAPT remains lower: **0.08333
  versus 0.08681 raw**, exactly one fewer correct response per seed. The gap is about
  half of v4's but was not reversed. This does not support a Simulation-specific CoAPT
  superiority claim and must not trigger further tuning on the same dev slice.
- The full v5 report and run identities are in
  `workspace/coapt/linkedin-ab-v5/r0/round_report.md`.

## Next actions

1. Treat v5 as the retained equal-span 1B result for overall CoAPT effectiveness; do
   not tune further against this dev split.
2. For the corpus release, use base versus standard CPT as the primary corpus claim.
   Report CoAPT only on the overall pool, not as Simulation-specific superiority.
3. A Simulation-specific CoAPT claim requires a newly frozen unseen evaluation, ideally
   on a new corpus or model, before consulting milestone referees at the phase gate.

## Active campaign: CPT scale ladder v2 (2026-08-25 → )

Human directive 2026-08-25: keep the focus on continued pretraining but scale the raw
corpus stream (1.4M → 14M → 140M → 1.4B → 5B one-epoch tokens on
`openbmb/MiniCPM5-1B-Base`), track held-out next-token loss on a new frozen split, and
compare CoAPT with raw CPT at each scale. Teacher tokens cannot scale with the stream
(Codex ≈ 37k gated tokens/hour), so CoAPT is tested as the frozen v5 pair annealed from
each ladder checkpoint, plus one re-diagnosed block at the largest scale. Full design,
cost table (~52 GPU-hours through 1.4B, +7 days for 5B), decision rules, and the five
open decisions: `workspace/campaigns/cpt-scale-ladder.md`.

State 2026-08-28 (full log: `workspace/campaigns/cpt-scale-ladder.md` §10):

- L0–L3 raw ladder done (3 seeds ≤140M, 1 seed at 1B). `corpus_heldout_nll_dev` falls
  monotonically 2.1736 → 2.1501 → 2.1228 → 2.0794 → 2.0423 (≈ −0.038 nats per decade,
  all 11 topics); `generic_heldout_nll` rises 2.6110 → 2.7170, accelerating; probes flat.
- Tier-1 anneal: Δpool(CoAPT − raw) positive in 15/15 paired seeds at N = 0…1B; transfer
  guardrail fails at L0/L1, passes at L2/L3. CoAPT is called better at L2 and L3 only.
- §7 L4 gate failed on (b); human directive 2026-08-28 overrode the resource gate.
  L4 (5B, 1 seed) completed 2026-09-05: NLL dev 2.0331 (−0.013/decade vs the
  −0.038/decade L0–L3 trend — the curve bends); generic 2.7951 (forgetting still
  accelerating); transfer 0.1244 (ladder-best, above base); four topics rise 1B→5B.
  L4 anneal: Δpool +0.0033 (band 0.0018; 18/18 pairs positive overall, but one L4 pair
  inside the band) — the CoAPT edge persists yet roughly halves at 5B.
- 2026-09-11: nekaise-bench v6 **dev** sweep on 24 checkpoints (human request; test
  split untouched, all metrics diagnostic): base 0.0593 (14/236); raw 5B 0.0763 with
  churn (loses 6 base-correct Qs, gains 10); v5 CoAPT at N=0 1–2 Qs below raw in all
  seeds; L4-anneal CoAPT 0.0791 vs raw 0.0636 (+0.0155, 3/3 seeds), largely by
  restoring base-correct answers the raw 5B path forgot. A 1B closed-book model is
  near floor here. Log: `workspace/campaigns/cpt-scale-ladder.md` §10.
- 2026-09-11: **extension L5 (10B) + L6 (15B) approved by the owner (option C).**
  Composite plan v3 sealed (L4's exact bytes as anchor, v2's remaining spans, then
  new documents; L5 ⊂ L6 byte prefixes). Training path hardened after three Codex
  reviews (strict resume, structural checkpoint validation, monotonic budget with
  forced save on timeout, recoverable timeboxed runs, byte-authenticated audits,
  anchor re-derivation audit, GPU lock); final-code GPU smokes passed with resumed
  training inside the four-run nondeterminism envelope. L5 dataset build + anchor
  audit running detached; launch via `workspace/coapt/scale-v2/master_l5l6.sh` once
  the manifest shows prefix_verified + anchor_audit_bound and the leakage audit
  passes. Design and decision record: campaign note §11; log entries of 2026-09-11.
- Next: tier-2 (re-diagnose the 5B/1B student, re-select frontier, Codex teacher
  block, four arms × 3 seeds), then the once-only milestones (frozen NLL +
  nekaise_bench on base + largest raw + largest CoAPT checkpoints).
