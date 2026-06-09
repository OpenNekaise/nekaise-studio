---
name: run-experiment
description: Run an autoresearch fine-tuning loop — propose a change to an experiment's train.py, fine-tune a small model with Unsloth, score it on the task pack, keep or revert. Use when improving a model in experiments/, kicking off an experiment, or driving the auto-research loop.
---

# run-experiment

This is the Claude Code adapter for Nekaise Studio's autoresearch loop. The canonical,
driver-agnostic instructions live in **[`skills/run-experiment.md`](../../../skills/run-experiment.md)** —
read and follow that file. It is the single source of truth (Codex reads the same file via
`AGENTS.md`).

In short: pick an experiment under `experiments/`, read its `LOG.md` and `train.py`, propose
**one** change, run `python experiments/<name>/train.py`, read the printed `METRIC` line, and
keep or revert based on whether it beat the best in `LOG.md`. Never edit `packs/*/scorer.py`.
