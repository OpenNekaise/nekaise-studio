---
name: judge
description: Act as an impartial senior-building-engineer examiner that grades against a FIXED reference (never free-judges). Two modes — gate mode filters hallucinated/ungrounded training pairs before they enter the dataset (called by prepare-trainset); eval mode scores the student model on the frozen open-ended exam (packs/building/eval_open.jsonl) as an advisory second metric alongside building_acc. Use when filtering generated training data or running open-ended evaluation. Never replaces the deterministic scorer or the GRPO reward.
---

# judge

This is the Claude Code adapter for Nekaise Studio's **LLM-as-judge** skill — Claude Code acting
as a building-engineer examiner, out of the fine-tuning hot loop. The canonical, driver-agnostic
instructions live in **[`skills/judge.md`](../../../skills/judge.md)** — read and follow that
file. It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short, two modes, both grading against a **fixed reference**:
- **Gate mode** — for each candidate training pair, verify it is grounded and correct against the
  building's corpus/ontology; drop hallucinations. Fail closed when unsure.
- **Eval mode** — grade the student's answers on the **frozen** `packs/building/eval_open.jsonl`
  (reference + rubric), aggregate to `building_judge`, and report it **alongside** `building_acc`.

**Hard rules:** grade only against the corpus or the frozen exam — never free-judge; never edit
`packs/*/scorer.py`, `prepare.py`, or `eval_open.jsonl`; never feed your output to GRPO or use it
for keep/revert (deterministic `building_acc` is the boss); temperature 0; no real names in
tracked files.
