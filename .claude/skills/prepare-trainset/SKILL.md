---
name: prepare-trainset
description: Act as a senior building engineer and prepare grounded SFT training data for the small "junior" model, by reading a building's full corpus (ontology TTL + control-card PDFs/images + guides + trends + alarms) under nekaise_data/ and writing provenance-tracked data.jsonl via lib/datakit. Use when building or refreshing the training set for a building experiment, or authoring the frozen open-ended exam. Excludes the holdout building. Keeps proprietary names out of tracked files.
---

# prepare-trainset

This is the Claude Code adapter for Nekaise Studio's **agentic data-prep** skill — Claude Code,
as a senior building engineer, distilling a junior building engineer's training data from a
building's real documentation. The canonical, driver-agnostic instructions live in
**[`skills/prepare-trainset.md`](../../../skills/prepare-trainset.md)** — read and follow that
file. It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short: enumerate buildings under `nekaise_data/` (skip the holdout), read each training
building's **whole** folder (TTL + control cards + guides + trends + alarms), author grounded
Q&A across classification / topology / sensors / control sequences / setpoints / alarms /
operational reasoning, gate every pair with the `judge` skill, and write the dataset through
`lib/datakit` so `train.py` (`DATASET="auto"`) picks it up. Once per pack, author the frozen
open-ended exam `packs/building/eval_open.jsonl` for the holdout.

**Hard rules:** never generate from the holdout; never edit `packs/*/scorer.py` or `prepare.py`;
ground everything (no invented entities/values); and **no real building/partner names in any
tracked file** — the raw data, datasets, and exam are git-ignored; keep it that way.
