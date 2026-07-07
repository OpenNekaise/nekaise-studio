# Experiments

One folder per (base model × task pack). Each holds the two **editable recipe files** the
autoresearch agent mutates — `train.py` (HOW to train: method, hyperparameters) and
`build_data.py` (WHAT to train on: distilled / rejection-sampled data) — plus `LOG.md` (its
journal). Runtime-only, git-ignored: `data/<id>/` (cached dataset artifacts + provenance),
`outputs/<stage>/` (checkpoints, each with a provenance `meta.json`; `best.json` tracks the
winner for `serve/`), `runs/` (dashboard telemetry), and `results.jsonl` (the measurement
ledger every eval/train appends to). See
[`../skills/run-experiment.md`](../skills/run-experiment.md) and the repo-root `STATUS.md`.

| Experiment | Base model | Metric | Status |
|------------|-----------|------|--------|
| `ceiling-sub4b` | sub-4B bases (granite-4.1-3b, Qwen3.5-0.8B/2B via `NEKAISE_BASE_MODEL`) | `nekaise_bench` dev (independent) | **current phase** — CPT + corpus-QA distill |
| `granite-4.1-3b-building` | `unsloth/granite-4.1-3b` | `building` pack + `building_judge` | deferred — needs `nekaise_data/` + `NEKAISE_HOLDOUT` |
| `granite-4.1-3b-gsm8k` | `unsloth/granite-4.1-3b` | `gsm8k` pack | bootstrap — works on a bare clone |
| _granite-4.1-8b-*_ / _gemma-4-*_ / _sub-1B granite_ | — | — | planned |

The building experiment carries extra (still recipe-level, editable) files:

- `build_cpt_data.py` / `augment_corpus.py` — continued-pretraining corpus prep + augmentation.
- `eval_judge.py` — grades the frozen realistic exam by anchors → `building_judge` (the gap).
- `eval_domain.py` + `domain_quiz*.jsonl` — closed-book domain quiz → `domain_quiz` (the ceiling).
- `gen_corpus_quiz.py` — generates corpus-knowledge probes.

To add a model: copy an existing folder, change `BASE_MODEL` in `train.py` (or set
`NEKAISE_BASE_MODEL`), reset `LOG.md`. The task pack, the loop, and the recipe-file
structure stay identical.
