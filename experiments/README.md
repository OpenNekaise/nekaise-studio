# Experiments

One folder per (base model × task pack). Each holds the two **editable recipe files** the
autoresearch agent mutates — `train.py` (HOW to train: method, hyperparameters) and
`build_data.py` (WHAT to train on: distilled / rejection-sampled data) — plus `LOG.md` (its
journal). Runtime-only, git-ignored: `data/<id>/` (cached dataset artifacts + provenance),
`outputs/<stage>/` (checkpoints; `best.json` tracks the winner for `serve/`), and `runs/`
(dashboard telemetry). See [`../skills/run-experiment.md`](../skills/run-experiment.md).

| Experiment | Base model | Pack | Status |
|------------|-----------|------|--------|
| `granite-4.1-3b-building` | `unsloth/granite-4.1-3b` | `building` | **flagship** — needs `nekaise_data/` + `NEKAISE_HOLDOUT` |
| `granite-4.1-3b-gsm8k` | `unsloth/granite-4.1-3b` | `gsm8k` | bootstrap — works on a bare clone |
| _granite-4.1-8b-*_ / _gemma-4-*_ / _qwen-3.5-*_ / _sub-1B granite_ | — | — | planned |

The building experiment carries extra (still recipe-level, editable) files:

- `build_cpt_data.py` / `augment_corpus.py` — continued-pretraining corpus prep + augmentation.
- `eval_judge.py` — grades the frozen realistic exam by anchors → `building_judge` (the gap).
- `eval_domain.py` + `domain_quiz*.jsonl` — closed-book domain quiz → `domain_quiz` (the ceiling).
- `gen_corpus_quiz.py` — generates corpus-knowledge probes.

To add a model: copy an existing folder, change `BASE_MODEL` in `train.py` (or set
`NEKAISE_BASE_MODEL`), reset `LOG.md`. The task pack, the loop, and the recipe-file
structure stay identical.
