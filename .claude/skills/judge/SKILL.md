---
name: judge
description: Act as an impartial expert grader for realistic building QA, scoring a candidate answer against each question's required ANCHORS (fraction matched, in [0,1]) — strict on the fact itself (values/tags/file-paths/component-names/time-windows), lenient on phrasing — blind to which model produced it. Two modes — gate mode keeps only fully-correct (score==1.0) teacher demos before they enter the training set; eval mode scores the student on the frozen realistic exam (packs/building/eval_open.jsonl) to produce building_judge, the primary building-quality signal. Use when filtering generated training data or running realistic-question evaluation.
---

# judge

This is the Claude Code adapter for Nekaise Studio's **LLM-as-judge** skill — Claude Code acting
as a building-engineer examiner, out of the fine-tuning hot loop. The canonical, driver-agnostic
instructions live in **[`skills/judge.md`](../../../skills/judge.md)** — read and follow that
file. It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short, two modes, both grading **blind** by each question's **`anchors`** — the score is the
**fraction of required facts present** (strict on the fact itself: values / vendor tags / file
paths / component names / time windows; lenient on phrasing and order):
- **Gate mode** — for each candidate teacher demo, score it against its anchors/corpus and **keep
  only `score==1.0`** (every anchor matched, nothing contradicted); drop the rest. Fail closed.
- **Eval mode** — grade the student's answers on the **frozen** realistic exam
  `packs/building/eval_open.jsonl` (`anchors` + `ground_truth`), aggregate to `building_judge`
  (plus per-`intent` means). This is the **primary** building-quality signal; the deterministic
  scorer is a cheap sanity check.

**Hard rules:** grade only against the corpus or the frozen exam — never free-judge; judge blind;
never edit `packs/*/scorer.py`, `prepare.py`, or `eval_open.jsonl`; temperature 0; no real names
in tracked files.
