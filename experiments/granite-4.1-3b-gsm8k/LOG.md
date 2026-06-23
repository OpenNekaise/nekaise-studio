# Experiment log — granite-4.1-3b-gsm8k

Base model: `unsloth/granite-4.1-3b` · Pack: `gsm8k` · Metric: `gsm8k_acc` (higher is better)

The autoresearch loop appends one entry per run. Newest at the bottom. Each run changes one
thing in a recipe file — `train.py` (method/hyperparameters) or `build_data.py` (data).
The first run makes **no change** — it establishes the baseline (`train.py` with
`DATASET="render"`, i.e. SFT on the pack's gold answers).

| # | hypothesis | recipe change | method | gsm8k_acc | train_min | result |
|---|-----------|---------------|--------|-----------|-----------|--------|
| 0 | baseline  | none (DATASET=render) | sft | _pending_ | _pending_ | _run me_ |

## Notes

- Best so far: _none yet_.
- Dead ends (don't retry): _none yet_.
