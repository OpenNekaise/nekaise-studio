# Task pack: gsm8k

**Bootstrap pack.** A public, automatic, fine-tuning-sensitive task used to prove the
autoresearch loop works end-to-end before swapping in building-ontology data.

| | |
|---|---|
| **Dataset** | `openai/gsm8k` (config `main`) — grade-school math word problems |
| **Train / test** | 7.5k / 1.3k (loop scores the first `EVAL_N`, default 200) |
| **Metric** | `gsm8k_acc` — exact match on the final numeric answer (higher is better) |
| **Scorer** | [`scorer.py`](scorer.py) — **fixed; the agent must not edit it** |

## Why GSM8K for the bootstrap

- Fully **automatic & deterministic** scoring (no human, no LLM judge) → cheap to run hundreds
  of times overnight.
- **Sensitive to fine-tuning** → the loop sees a metric it can climb.
- Single-GPU friendly with Unsloth.

## The swap to building ontology

A future `packs/building-ontology/` will expose the same contract — a `scorer.py` with
`is_correct()` + `load_test()` and a `prepare.py` — measuring ontology lookups, point-name
normalization, and query construction. Nothing in the loop, the skill, or `train.py`'s
structure changes; only `BASE_MODEL` stays and the pack import swaps.
