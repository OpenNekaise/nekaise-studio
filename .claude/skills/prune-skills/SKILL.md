---
name: prune-skills
description: Review and consolidate the local skill library (skills/local/) — merge duplicates, drop stale / contradicted / overfit advice, cut bloat. The "selection" half of the studio's self-improvement. Use periodically or whenever skills/local/ has grown.
---

# prune-skills

Claude Code adapter for Nekaise Studio. The canonical, driver-agnostic instructions live in
**[`skills/prune-skills.md`](../../../skills/prune-skills.md)** — read and follow that file. It is
the single source of truth (Codex reads the same file via `AGENTS.md`).

In short: read every skill in `skills/local/`, merge duplicates, delete stale / disproven / overfit
ones, and cut bloat to the actionable lever. The core `skills/*.md`, `packs/*`, and `lib/*` are
maintainer-owned — propose changes as PRs, never edit in place. A skill survives only if its
validating evidence still holds against the current eval harness. Leave the library leaner and
non-contradictory; note what you changed in `LOG.md`.
