---
name: prepare-trainset
description: Act as a senior building engineer and prepare training data for the small "junior" model by authoring the REALISTIC questions a building engineer or operator would actually ask (topology / control / factual / timeseries, in their natural phrasing) with grounded correct answers, read from a building's full corpus (ontology TTL + control-card PDFs/images + guides + trends + alarms) under nekaise_data/. Each answer nails the verifiable anchors (values, vendor tags, file paths, component names, time windows) + source. Writes provenance-tracked data.jsonl via lib/datakit and the frozen realistic exam. Excludes the holdout. Keeps proprietary names out of tracked files.
---

# prepare-trainset

This is the Claude Code adapter for Nekaise Studio's **agentic data-prep** skill — Claude Code,
as a senior building engineer, distilling a junior building engineer's training data from a
building's real documentation. The canonical, driver-agnostic instructions live in
**[`skills/prepare-trainset.md`](../../../skills/prepare-trainset.md)** — read and follow that
file. It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short: enumerate buildings under `nekaise_data/` (skip the holdout), read each training
building's **whole** folder (TTL + control cards + guides + trends + alarms), and author the
**real questions an engineer or operator would ask** — categories `topology` / `control` /
`factual` / `timeseries`, in their natural voice (engineer-precise and operator-casual) — each
with a grounded answer that nails the verifiable anchors (values, tags, file paths, component
names, windows) + `source`. Gate every pair with the `judge` skill (keep only fully-correct),
write the dataset through `lib/datakit` so `train.py` (`DATASET="auto"`) picks it up, and author
the frozen realistic exam `packs/building/eval_open.jsonl` for the holdout once.

**Hard rules:** never generate from the holdout; never edit `packs/*/scorer.py` or `prepare.py`;
ground everything (no invented entities/values); and **no real building/partner names in any
tracked file** — the raw data, datasets, and exam are git-ignored; keep it that way.
