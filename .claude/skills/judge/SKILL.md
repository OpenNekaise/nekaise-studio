---
name: judge
description: Act as an impartial expert grader for realistic building QA, scoring a candidate answer against a FIXED grounded reference on a 3-point rubric (1.0/0.5/0.0 — strict on values/tags/file-paths/component-names/time-windows, lenient on phrasing), blind to which model produced it. Two modes — gate mode keeps only fully-correct (1.0) teacher demos before they enter the training set; eval mode scores the student on the frozen realistic exam (packs/building/eval_open.jsonl) to produce building_judge, the primary building-quality signal. Use when filtering generated training data or running realistic-question evaluation.
---

# judge

This is the Claude Code adapter for Nekaise Studio's **LLM-as-judge** skill — Claude Code acting
as a building-engineer examiner, out of the fine-tuning hot loop. The canonical, driver-agnostic
instructions live in **[`skills/judge.md`](../../../skills/judge.md)** — read and follow that
file. It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short, two modes, both grading **blind** against a **fixed reference** on the 3-point rubric
(strict on numeric values / vendor tags / file paths / component names / time windows; lenient on
phrasing and ordering):
- **Gate mode** — for each candidate teacher demo, score it against its source/corpus and **keep
  only 1.0** (fully correct, fully grounded); drop the rest. Fail closed when unsure.
- **Eval mode** — grade the student's answers on the **frozen** realistic exam
  `packs/building/eval_open.jsonl` (`ground_truth` + `source`), aggregate to `building_judge`
  (plus per-category means). This is the **primary** building-quality signal; the deterministic
  scorer is a cheap sanity check.

**Hard rules:** grade only against the corpus or the frozen exam — never free-judge; judge blind;
never edit `packs/*/scorer.py`, `prepare.py`, or `eval_open.jsonl`; temperature 0; no real names
in tracked files.
