# Experiments

One folder per (base model × task pack). Each holds `train.py` (the editable fine-tune
script the autoresearch agent mutates) and `LOG.md` (its journal). See
[`../skills/run-experiment.md`](../skills/run-experiment.md).

| Experiment | Base model | Status |
|------------|-----------|--------|
| `granite-4.1-3b-gsm8k` | `unsloth/granite-4.1-3b` | active (bootstrap) |
| _granite-4.1-8b-gsm8k_ | `unsloth/granite-4.1-8b` | planned |
| _gemma-4-*-gsm8k_ | _placeholder_ | planned |
| _qwen-3.5-*-gsm8k_ | _placeholder_ | planned |
| _granite-4.0-1b-gsm8k_ | `unsloth/granite-4.0-1b` | planned (sub-1B target) |

To add a model: copy an existing folder, change `BASE_MODEL` in `train.py`, reset `LOG.md`.
The task pack and the loop stay identical.
