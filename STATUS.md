# STATUS — current phase and targets

> Durable procedure lives in `skills/` and `AGENTS.md`; this page is the part that CHANGES.
> Update it when the phase, metric, or priorities shift. Last update: 2026-07-07.

## Phase: CEILING

Bake general building-energy knowledge into sub-4B weights: **CPT over
`nekaise_data/hvac_corpus/` → corpus-QA distill SFT**, measured on **nekaise-bench**
(`packs/bench`, dev split = loop metric, test split = FROZEN milestones). Building-specific
gap work (`granite-4.1-3b-building`, GRPO anchor-recall) is deferred; its holdout guard
stays armed.

- Active experiment: `experiments/ceiling-sub4b/`
- Loop metric: `python tools/eval_bench.py --checkpoint <stage> --split dev` (batched, ~40 s)
- Campaigns: `python tools/campaign.py <spec.json>` (idempotent resume)

## Best known (dev split, checkpoint mode, 2026-07-07)

granite-4.1-3b **base** is still the strongest sub-4B (0.1606) and immune to all treatments
so far. The CPT→distill recipe is validated on Qwen3.5-0.8B only (0.1224 → 0.1453, McNemar
p≈0.017, direction confirmed on the frozen test). Qwen3.5-2B's dev gain REVERSED on the
frozen test — not shipped. Full evidence: [`docs/RESULTS.md`](docs/RESULTS.md).

## Next levers (one change per run)

1. **Scale distill data** 309 → ~1500 pairs (more chunks/doc, second seed pass); re-test
   0.8B and 2B. Cheapest promising lever: 309 pairs / 40 steps already moved 0.8B.
2. **GRPO on the bench pack** — `packs/bench.reward()` is a verifiable reward now; apply the
   validated anchor-recall recipe shape to the ceiling phase (init from distill checkpoints).
3. **No-leak ablation** — exclude bench-cited source docs from CPT+distill to split
   doc-specific recall from transfer.
4. **MCQ-style distill pairs** — the mcq track moves most under treatment.
5. **Corpus hygiene** — prune the known off-topic DOE docs (in nekaise-corpus) before the
   next CPT round.

## Recently retired

- Local skill `verify-holdout-before-judge` — superseded by the in-code guard
  (`prepare.require_holdout_matches_exam`, commit f0cbab6) + doctor check.
