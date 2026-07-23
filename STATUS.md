# STATUS — current phase and targets

> Durable procedure lives in `SPEC.md`, `BOUNDARY.md`, `skills/`, and `AGENTS.md`.
> This file contains only the active state. Last update: 2026-07-23.

## Phase: NEKAISE-COAPT MVP (two rounds)

Active model: **`openbmb/MiniCPM5-1B-Base`**. The campaign is the first full
**nekaise-coapt** loop (SPEC.md §1): two chained rounds, R0 and R1, driven by the
`coapt-round` skill. The goal is NOT a score target — it is to demonstrate that the
closed loop is real: training changes the student, the changed student changes the
diagnosis, and the changed diagnosis changes the data.

The CPT data-scaling ladder (R0/4M/40M/400M/1B/2B, stream plan `nekaise-1b-cpt-v1`) is
**closed by human directive 2026-07-23**, superseding the 2026-07-22 finish-the-ladder
directive. Its runs, datasets, and the 40M noise band (0.003143 on
`corpus_transfer_macro_dev`, seeds 0.1221/0.1181/0.1159) remain in the RunStore as
evidence; nothing from it chains into this campaign.

## Success criteria (fixed before R0)

1. **Pipeline** — both rounds run end to end through the skills; every round artifact
   (pool, frontier, drafts, teacher rows, gate verdicts, token ledger, run ids, evals)
   exists with provenance.
2. **Loop closure** (the MVP's core evidence, checked at R1 entry):
   R0 teacher-text NLL falls under S1; R0 questions answer better under S1 (vs the
   recorded `student_verdict` rate); the frontier turns over. The first two are hard
   requirements — if either fails, stop and report.
3. **Learning** — `coapt_pool_absorption_dev`: S1 > S0 and S2 > S1 beyond the R0
   three-seed noise band, with `corpus_transfer_macro_dev` never falling more than the
   band below the previous student (guardrail; a breach reverts the round).

## Agent control path

1. `python tools/doctor.py`
2. Follow `skills/coapt-round.md`. R0 sequence in brief:
   `build_data.py init-pool` → baseline reference (full dev + pool view) →
   `emit-docs` → `student.py score` → `select-frontier` → `make-drafts` →
   branch skills (`coapt-cpt`, `coapt-sft`) →
   `python -m studio.cli build coapt --config configs/coapt.yaml` →
   `python -m studio.cli train cpt --config configs/coapt.yaml --seed {3407,3408,3409}` →
   pool + transfer eval per seed → noise band `max(std, half_range)` →
   `python -m studio.cli decide <run_id> --metric coapt_pool_absorption_dev ...` →
   round report.
3. R1 chains from **seed 3407's** R0 checkpoint (fixed in advance): set
   `run.init_from: run:<r0 seed-3407 run_id>`, `data.round: 1`,
   `data.round_dir: coapt/rounds/r1`; run the loop-closure checks before generating.

All commands emit bounded JSON or stable `RUN_ID`/`EVAL_RESULT`/`METRIC` records. Query
runs with `studio.cli list/show/resolve`; use run ids as evidence.

## Current next actions

- R0 has not started. First moves: `python tools/doctor.py`, then freeze the pool
  (`python experiments/coapt/build_data.py init-pool`) and record the S0 baseline
  (full dev `--record-reference coapt`, then the pool view on that reference run).
- Round budget: `target_content_tokens: 4M` cap; teacher volume drives the actual total
  (ratio-locked mix 40/25/15/20). Expect ~1–2k teacher corrections per round; the
  teacher work in the branch skills is the wall-clock bottleneck, not the GPU.
- No noise band exists yet for `coapt_pool_absorption_dev` — R0's three seeds create it.
  Until then every decide is `pending`/`baseline`.

## Deferred

Chat-format SFT consolidation (`studio/stages/sft.py`), the CoAPT-vs-raw-CPT
null-hypothesis A/B (mandatory before any effectiveness claim, SPEC.md §3), building-pack
work (`prepare-trainset`/`judge` remain dormant assets), `nekaise_bench` milestones,
export, and multi-machine execution.
