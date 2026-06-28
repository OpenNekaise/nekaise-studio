---
name: crystallize-skill
description: After an experiment, turn a VALIDATED, reusable finding into a new local skill (skills/local/). The "mutation" half of the studio's self-improvement. Use at the end of a run-experiment cycle when a result is confirmed by the eval harness and worth reusing — not for hunches, building-specific facts, or one-off results.
---

# crystallize-skill

Claude Code adapter for Nekaise Studio. The canonical, driver-agnostic instructions live in
**[`skills/crystallize-skill.md`](../../../skills/crystallize-skill.md)** — read and follow that
file. It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short: if an experiment produced a **validated, reusable** lesson not already captured, write it
as `skills/local/<name>.md` (git-ignored, this machine's) in the `# Skill:` format, citing the
evidence that validated it. Do NOT crystallize hunches (→ `LOG.md`), building-specific facts
(→ `nekaise_data/`), or one-off results. Promote a local skill into the core `skills/` kernel only
by an eval-gated, human-reviewed PR.
