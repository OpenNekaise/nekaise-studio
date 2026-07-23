# Nekaise Studio — Claude Code entrypoint

@AGENTS.md

The constitution (algorithm card, ban list, null-hypothesis rule, environment lock):
**SPEC.md** — binding on every experiment decision. Current phase, targets, and next
levers: read **STATUS.md** first — it is the part that
changes; skills and AGENTS.md are the parts that don't.

First run on a fresh clone: `python tools/doctor.py` (preflight: GPU, deps, `.env`, data,
holdout). Then read `skills/coapt-round.md` and drive the loop.
