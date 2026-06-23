# Experiments

One folder per (base model × task pack). Each holds the two **editable recipe files** the
autoresearch agent mutates — `train.py` (HOW to train: method, hyperparameters) and
`build_data.py` (WHAT to train on: distilled / rejection-sampled data) — plus `LOG.md` (its
journal). Runtime-only, git-ignored: `data/<id>/` (cached dataset artifacts + provenance)
and `outputs/<stage>/` (checkpoints; `best.json` tracks the winner for `serve/`). See
[`../skills/run-experiment.md`](../skills/run-experiment.md).

| Experiment | Base model | Status |
|------------|-----------|--------|
| `granite-4.1-3b-gsm8k` | `unsloth/granite-4.1-3b` | active (bootstrap) |
| _granite-4.1-8b-gsm8k_ | `unsloth/granite-4.1-8b` | planned |
| _gemma-4-*-gsm8k_ | _placeholder_ | planned |
| _qwen-3.5-*-gsm8k_ | _placeholder_ | planned |
| _granite-4.0-1b-gsm8k_ | `unsloth/granite-4.0-1b` | planned (sub-1B target) |

To add a model: copy an existing folder, change `BASE_MODEL` in `train.py`, reset `LOG.md`.
The task pack, the loop, and the recipe-file structure stay identical.
