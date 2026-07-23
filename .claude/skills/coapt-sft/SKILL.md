---
name: coapt-sft
description: Act as the examiner-then-tutor for nekaise-coapt's Personalized SFT branch (d → q_T → a_S → T(d, q_T, a_S) → a*) - author closed-book questions with verbatim evidence spans from each frontier document, collect the student's closed-book answers via tools/student.py answer, record a student_verdict per question, correct each answer into a concise a* that keeps the student's correct words and traces every claim to the evidence, and gate rows in a second pass. The training pair is (question, a*), serialized in the same Question:/Answer: format the diagnosis uses. Use during a CoAPT round after the frontier exists.
---

# coapt-sft

This is the Claude Code adapter for the Personalized SFT branch. The canonical,
driver-agnostic instructions live in
**[`skills/coapt-sft.md`](../../../skills/coapt-sft.md)** — read and follow that file.
It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short: author 2 typed questions per frontier doc (own words, verbatim `evidence`
substring, short `reference_answer`), run `tools/student.py answer` closed-book, then
correct each student answer into `a*` grounded only in the document, recording
`student_verdict` — that rate is the baseline next round's loop-closure check must beat.
Gate evidence and claims in a second pass; failed rows stay as evidence. Never open the
probe referee's files; these questions are training data, never an exam.
