---
name: coapt-cpt
description: Act as the teacher-editor for nekaise-coapt's Adaptive CPT branch (d → r_S → T(d, r_S) → d̃) - generate greedy student drafts over frontier corpus chunks with tools/student.py, then correct each draft into textbook prose grounded STRICTLY in the source chunk (numbers/terms verbatim from the source, keep the student's correct sentences, no outside knowledge even when true), gate every row claim-by-claim against the source in a second pass, and write provenance-complete cpt_teacher.jsonl plus the pre-training NLL baseline for the next round's loop-closure check. Use during a CoAPT round after the frontier and draft prompts exist.
---

# coapt-cpt

This is the Claude Code adapter for the Adaptive CPT branch. The canonical,
driver-agnostic instructions live in
**[`skills/coapt-cpt.md`](../../../skills/coapt-cpt.md)** — read and follow that file.
It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short: run `tools/student.py draft` on the frozen draft prompts, then for each row
diagnose the student's errors and write `teacher_text` supported entirely by
`source_chunk` — the draft decides what needs saying, the source decides what is true.
Gate in a separate pass (one unsupported claim fails the row; failed rows stay as
evidence), then score the teacher texts under the pre-training student for next round's
closure check. Never open the probe referee's files.
