# AGENTS.md — Nekaise Studio

Nekaise Studio is an **agentic fine-tuning platform**: you (Claude Code or Codex) improve
small language models (<8B) by running an autoresearch loop, driven entirely through skills
— no human runs Python directly.

## How to work in this repo

The canonical instructions are in **[`skills/run-experiment.md`](skills/run-experiment.md)**.
Read it first. The loop:

1. Pick an experiment in `experiments/<name>/`.
2. Read its `LOG.md` and the two editable recipe files: `train.py` (HOW) + `build_data.py` (WHAT).
3. Propose **one** change, edit one recipe file.
4. Run it (`build_data.py` to (re)build a dataset, then `train.py`; time-boxed) and read the `METRIC` line.
5. Keep the change if it beats the best in `LOG.md`; otherwise revert. Log either way.

## Hard rules

- Edit **only** the experiment's recipe files: `train.py` and/or `build_data.py`.
- **Never** edit `packs/*/{scorer,prepare}.py` or `lib/*` — that's the fixed referee/plumbing.
  Using the scorer to *filter training data* is allowed (eval still runs on the held-out
  test split with the same scorer); never change how the metric is computed.
- One change per run. Honor the training time box. Log every run, including failures.

## Map

- `skills/` — the source-of-truth skill (Claude Code mirrors it under `.claude/skills/`).
- `experiments/` — two editable recipe files (`train.py` = method/hyperparameters,
  `build_data.py` = distilled/rejection-sampled data) + `LOG.md`. Runtime, git-ignored:
  `data/<id>/` (cached datasets + provenance), `outputs/<stage>/` (checkpoints + `best.json`).
- `packs/` — fixed task packs: dataset + scorer (`load_split` / `is_correct` / `reward`).
- `lib/` — fixed plumbing: `pack.py` (load a pack), `datakit.py` (dataset cache+provenance),
  `llm.py` (teacher/sampler backends: ollama/anthropic/openai).
- `serve/` — export the winning checkpoint (`outputs/best`) to GGUF + Ollama (handoff to nekaise-edge).
