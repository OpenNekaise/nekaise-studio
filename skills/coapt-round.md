# Skill: coapt-round

Run one round of **nekaise-coapt** (CoAPT = Co-Adaptive Pretraining and Tuning) under the
constraints in `SPEC.md` §1. CoAPT is a closed loop: the current student's measured state
selects the data, you teach against the real corpus, the mix trains the next student, and
the next student is re-measured. Read `STATUS.md` first; it names the active round,
student checkpoint, and noise band. `<round_dir>` below is `workspace/coapt/rounds/r<N>`.

## Separation of responsibilities

- **You** are the orchestrator and the teacher. Teaching (correcting drafts, authoring
  questions, correcting answers, gating) happens in the two branch skills —
  `skills/coapt-cpt.md` and `skills/coapt-sft.md` — executed by you, grounded in corpus
  text only.
- **Scripts** own the GPU and determinism: `tools/student.py` (student inference, eval
  env), `experiments/coapt/build_data.py` (pool/frontier/drafts/mix),
  `studio.cli train` (the frozen CPT stage), `tools/eval_probes.py` (the referee).
- **The recipe is frozen across rounds.** Between rounds of one campaign, only
  `data.round`, `data.round_dir`, `run.init_from`, and `run.seed` change in
  `configs/coapt.yaml`. Changing anything else (mix shares, templates, pool, frontier
  rule, gate rules, prompts) starts a NEW campaign — the loop's only variable is the data
  content the closed loop itself regenerates.
- Checkpoints chain: R0 trains from `base_model`, R(N) trains from
  `run:<R(N-1) run_id>`. Never restart from base mid-campaign.

## One round

1. `python tools/doctor.py`; stop on failures.
2. **R0 only — freeze the referee side once:**
   `python experiments/coapt/build_data.py init-pool` (refuses if the pool exists), then
   register the untrained baseline and its pool view:
   `$NEKAISE_EVAL_PYTHON tools/eval_probes.py --checkpoint <base_model> --split dev
   --record-reference coapt`, then re-run with
   `--run-id <reference_run> --doc-ids workspace/coapt/pool.jsonl`.
3. **Diagnose** the current student S (base at R0, else the previous round's run).
   Start a persistent server ONCE per checkpoint so every call skips the engine load:
   `$NEKAISE_EVAL_PYTHON tools/student.py serve --run-id <S>` (prints the
   `server:student@...` target; `serve-stop` when done with this checkpoint). Then
   `python experiments/coapt/build_data.py emit-docs`, then
   `$NEKAISE_EVAL_PYTHON tools/student.py score --target <server target>
   --in <round_dir>/docs.jsonl --out <round_dir>/nll.jsonl`, and the pool closed-book
   state: `$NEKAISE_EVAL_PYTHON tools/eval_probes.py --run-id <S> --target
   <server target> --doc-ids workspace/coapt/pool.jsonl --split dev` (the `EVAL_RESULT`
   line carries `records_file` — that path feeds `select-frontier`).
4. **R1+ only — loop-closure checks** (see below). If the loop is not closing, STOP and
   report; do not generate new data on a dead signal.
5. **Frontier:**
   `python experiments/coapt/build_data.py select-frontier --nll <round_dir>/nll.jsonl
   --probe-records workspace/probe_eval/<target>.dev.pool.jsonl
   [--previous <previous round_dir>/frontier.jsonl]`, then
   `python experiments/coapt/build_data.py make-drafts`.
6. **Branches:** execute `skills/coapt-cpt.md`, then `skills/coapt-sft.md`. They leave
   gated `cpt_teacher.jsonl` and `sft_final.jsonl` in `<round_dir>`.
7. **Build:** `python -m studio.cli build coapt --config configs/coapt.yaml`. Capture
   the `DATASET_ID` line from the build output and read
   `<round_dir>/token_ledger_<dataset_id>.json`; achieved shares must sit on `data.mix`
   (the builder refuses gross drift, you sanity-check the rest).
8. **Train:** `python -m studio.cli train cpt --config configs/coapt.yaml
   --dataset-id <dataset_id> [--seed N]` — always bind the dataset explicitly; the
   mutable LATEST pointer is a fallback, not the loop's contract.
   R0 runs THREE seeds (3407, 3408, 3409) on the same dataset to establish the pool
   noise band `max(std, half_range)`; the campaign chains from seed 3407's checkpoint —
   fixed in advance, never the best seed. R1+ runs one seed. Preserve every `run_id`.
9. **Evaluate** each trained run — the loop metric and the guardrail:
   `$NEKAISE_EVAL_PYTHON tools/eval_probes.py --run-id <run_id> --doc-ids
   workspace/coapt/pool.jsonl --split dev` and
   `$NEKAISE_EVAL_PYTHON tools/eval_probes.py --run-id <run_id> --split dev`.
10. **Decide:** `python -m studio.cli decide <run_id> --metric coapt_pool_absorption_dev
    --split dev --noise-band <band> [--baseline-run-id <previous>]`. Guardrail:
    `corpus_transfer_macro_dev` may not fall more than the band below the previous
    student's value — a guardrail breach is a revert regardless of pool gains.
11. **Round report:** write `<round_dir>/round_report.md` — run ids, pool/transfer
    metrics and deltas, closure signals, ledger summary, verdict, next action — and
    update the state sections of `STATUS.md`.

## Loop-closure checks (R1+ step 4 — the MVP's core evidence)

The previous round produced supervision; the new student must measurably differ on it.
All three run BEFORE generating this round's data, against the previous `<round_dir>`:

1. **Teacher-text NLL fell.** The branch skill stored
   `teacher_nll_pre.jsonl` (teacher texts scored under the pre-training student).
   Re-score the same file under the new student and compare mean NLL. It must be lower.
2. **Old questions answer better.** `$NEKAISE_EVAL_PYTHON tools/student.py answer
   --run-id <S> --in <previous>/sft_questions.jsonl --out <round_dir>/recheck_answers.jsonl`,
   grade against each `reference_answer` (you are the examiner; diagnostic only), and
   compare with the previous round's recorded `student_verdict` rate. It must be higher.
3. **The frontier turned over.** `select-frontier --previous` reports turnover; treated
   docs the student absorbed should exit. Report the fraction.

Signals 1 and 2 are hard: if either fails, the loop is not closing — stop, report, and
wait for human review. Signal 3 is reported, not thresholded.

## Hard rules

- One round = one loop iteration. No recipe edits, no hyperparameter opinions, no
  mid-round re-freezing. A failed round is recorded (`studio.cli decide` / `invalidate`),
  never silently rerun.
- Never regenerate the pool, the probes, or any frozen artifact. The referee is
  `corpus-probes-v2` through `tools/eval_probes.py` only.
- Never open `gym/tasks/corpus_probes/probes.jsonl` — not while teaching, not while
  selecting. The pool file and eval records are your only view of the referee.
- Transfer/frozen documents never enter any stream; the builder enforces it, you never
  work around it.
- Evidence is a `run_id` (or `run:`/`latest:`/`best:` pointer), never a pathname.
  Training loss and NLL are diagnostics; `coapt_pool_absorption_dev` decides, with
  `corpus_transfer_macro_dev` as guardrail.
- Perplexity is never a success metric.
