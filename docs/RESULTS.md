# Results — ceiling phase, sub-4B campaign (2026-07-07)

Everything here uses public assets only (public base models, the public
[nekaise-bench](https://github.com/OpenNekaise/nekaise-bench) benchmark, the public
[nekaise-corpus](https://github.com/OpenNekaise/nekaise-corpus) recipe) and is reproducible
from a clone of this repo plus those two.

## Question

For sub-4B models, does **CPT on the corpus → corpus-QA distillation SFT** put measurable
building-energy knowledge into the weights, judged by an independently-authored benchmark
the training code cannot game?

## Method

- **Metric:** nekaise-bench, checkpoint mode (`tools/eval_bench.py`), greedy decoding,
  128-token budget. Deterministic id-hash split: `dev` 523 questions for all decisions,
  `test` 154 questions FROZEN, used once at the end.
- **CPT:** next-token training on the cleaned corpus (~3.5M tokens; full-text chunked or
  full-parameter depending on model — see per-checkpoint `meta.json` provenance going forward).
- **Distill:** 309 short factual QA pairs authored from corpus chunks by a local
  qwen3.6:27b teacher (`think=false`), grounding-gated (answer must appear in the source
  chunk), LoRA SFT r16, 2 epochs (~40 steps).
- **Controls:** distill-without-CPT for every model; paired McNemar on per-question flips.

## Results (dev split, 523 questions)

| model | base | CPT only | distill only | CPT+distill | paired base→CPT+distill |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.1224 | 0.1358 | 0.1415 | **0.1453** | +17/−5, p≈0.017 |
| Qwen3.5-2B | 0.1377 | 0.1491 | 0.1472 | **0.1549** | +24/−15, p≈0.20 |
| granite-4.1-3b | 0.1606 | 0.1606 | 0.1625 | 0.1606 | +16/−16, p=1.0 |

## Milestone: frozen test split (154 questions, run once)

| model | base | CPT+distill | paired |
|---|---|---|---|
| Qwen3.5-0.8B | 0.0909 | 0.1039 | +4/−2 (direction holds) |
| Qwen3.5-2B | 0.1558 | 0.1169 | **+1/−7 — dev gain did NOT generalize** |

## Conclusions

1. **The recipe is real but small, and validated only on the 0.8B** (p≈0.017 on dev,
   direction confirmed on the frozen test). CPT and distill stack.
2. **The 2B's dev gain was split-fit, not knowledge** — the frozen test caught it. This is
   the dev/test protocol paying for itself on its first outing; report paired flips, not
   just aggregates.
3. **granite-4.1-3b is immune to every treatment** (flips balance exactly) and remains the
   strongest sub-4B base. If you deploy today, deploy granite base.
4. Corpus CPT did **not** degrade bench scores for any model here (checkpoint-mode eval) —
   contrary to an earlier observation on different eval sets; keep watching this.

## Caveats

- The test split is a weak instrument (154 questions, 22 mcq).
- Checkpoint-mode numbers are not comparable to Ollama/GGUF-mode numbers (different
  runtime, quantization, token budget). Compare like with like.
- The distill set was tiny (309 pairs); scale is the obvious next lever (see STATUS.md).

## Reproduce

```bash
git clone https://github.com/OpenNekaise/nekaise-studio && cd nekaise-studio
git clone https://github.com/OpenNekaise/nekaise-bench ../nekaise-bench
pip install -r requirements.txt && python tools/doctor.py
python tools/eval_bench.py --checkpoint unsloth/granite-4.1-3b --split dev   # ~40 s on a GPU
# then: experiments/ceiling-sub4b/build_data.py + train.py, or drive it with tools/campaign.py
```
