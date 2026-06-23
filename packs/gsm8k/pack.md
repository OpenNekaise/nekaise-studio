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
`load_split()`, `is_correct()`, `reward()`, `extract_answer()` and a `prepare.py` —
measuring ontology lookups, point-name normalization, and query construction. Nothing in the
loop, the skill, or the recipe files' structure changes; only the pack import swaps.

## Scorer contract (the fixed referee, reused by every method)

| Function | Used by |
|---|---|
| `load_split(split, n)` | data building (`train`) and evaluation (`test`) |
| `is_correct(pred, gold) -> bool` | the eval metric **and** the rejection-sampling filter |
| `reward(pred, gold) -> float` | graded signal for GRPO/RL (1.0 correct; 0.1 well-formed) |
| `extract_answer(text)` | shared parsing helper |
