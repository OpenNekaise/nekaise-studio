---
name: coapt-round
description: Run one round of nekaise-coapt (Co-Adaptive Pretraining and Tuning), the closed loop where the current 1B student's measured state selects the data, the agent teaches against the real corpus through the coapt-cpt and coapt-sft branch skills, and the token-ledgered mix (raw + adaptive CPT + personalized QA + anchor) trains the next student through the frozen CPT stage. Covers diagnosis (student NLL + pool probes), frontier selection, build/train/eval/decide with the noise band, the R1+ loop-closure checks, and the round report. Use when running or resuming a CoAPT campaign round.
---

# coapt-round

This is the Claude Code adapter for the CoAPT round conductor. The canonical,
driver-agnostic instructions live in
**[`skills/coapt-round.md`](../../../skills/coapt-round.md)** — read and follow that
file. It is the single source of truth (Codex reads the same file via `AGENTS.md`).

In short: read `STATUS.md` and `SPEC.md` §1, diagnose the current student with
`tools/student.py score` and the pool view of `tools/eval_probes.py`, run the R1+
loop-closure checks, select the frontier, execute the `coapt-cpt` and `coapt-sft`
branch skills as the teacher, then `studio.cli build coapt` → `studio.cli train cpt`
(chained checkpoints) → pool + transfer eval → `studio.cli decide`. The recipe is frozen
across rounds; only round number, round dir, `init_from`, and seed change.
