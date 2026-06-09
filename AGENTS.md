# AGENTS.md — Nekaise Studio

Nekaise Studio is an **agentic fine-tuning platform**: you (Claude Code or Codex) improve
small language models (<8B) by running an autoresearch loop, driven entirely through skills
— no human runs Python directly.

## How to work in this repo

The canonical instructions are in **[`skills/run-experiment.md`](skills/run-experiment.md)**.
Read it first. The loop:

1. Pick an experiment in `experiments/<name>/`.
2. Read its `LOG.md` and `train.py`.
3. Propose **one** change, edit only `train.py`.
4. Run `python experiments/<name>/train.py` (time-boxed) and read the `METRIC` line.
5. Keep the change if it beats the best in `LOG.md`; otherwise revert. Log either way.

## Hard rules

- Edit **only** `experiments/<name>/train.py`.
- **Never** edit `packs/*/scorer.py` or `packs/*/prepare.py` — that's the referee.
- One change per run. Honor the training time box. Log every run, including failures.

## Map

- `skills/` — the source-of-truth skills (Claude Code mirrors these under `.claude/skills/`).
- `experiments/` — editable `train.py` + `LOG.md`, one folder per base model × pack.
- `packs/` — fixed task packs (dataset + scorer = fitness function).
- `serve/` — export a winning model to GGUF and load it into Ollama (the handoff to nekaise-edge).
