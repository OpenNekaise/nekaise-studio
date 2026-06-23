---
name: run-experiment
description: Run an autoresearch fine-tuning loop — propose a change to an experiment's train.py, fine-tune a small model with Unsloth, score it on the task pack, keep or revert. Use when improving a model in experiments/, kicking off an experiment, or driving the auto-research loop.
---

# run-experiment

This is the Claude Code adapter for Nekaise Studio's autoresearch loop. The canonical,
driver-agnostic instructions live in **[`skills/run-experiment.md`](../../../skills/run-experiment.md)** —
read and follow that file. It is the single source of truth (Codex reads the same file via
`AGENTS.md`).

In short: pick an experiment under `experiments/`, read its `LOG.md` and its two editable
recipe files — `train.py` (HOW: method sft/dpo/grpo + hyperparameters) and `build_data.py`
(WHAT: distilled / rejection-sampled data). Propose **one** change in one file, run it
(`build_data.py` to (re)build a dataset, then `train.py`), read the printed `METRIC` line,
and keep or revert vs the best in `LOG.md`. Never edit `packs/*/{scorer,prepare}.py` or
`lib/*` (the fixed referee/plumbing).
