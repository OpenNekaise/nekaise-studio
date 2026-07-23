# SPEC.md — the constitution

> **Immutable bootloader layer.** This file defines what the agent may and may not do when
> training models in this repo. The `coapt-round` skill's adjudication logic references
> this file explicitly; every keep/revert decision is bound by it. Changing ANYTHING here is
> a human-reviewed spec change, never a loop move. Last human revision: 2026-07-23
> (CoAPT rewrite — supersedes the five-stage card and the CPT scale-ladder campaign).

Development and experiments are executed by an agent; an agent's behavior is determined by
readable files. Consensus that is not written down will be re-invented or violated by the
next agent instance. Hence this constitution. Its sibling **BOUNDARY.md** defines what
gym and studio each are and fixes the containment direction (Article 0: studio contains
gym, never the reverse) — equally binding.

## 1. Algorithm card — nekaise-coapt

**CoAPT (Co-Adaptive Pretraining and Tuning)** is the studio's algorithm: a closed loop
in which training data is conditioned on the current student's measured state, the
teacher is the coding agent grounded strictly in real corpus text, and the trained
student re-selects the next round's data.

    real corpus
        │
        ├── Adaptive CPT branch (CoAPT-CPT):   d → r_S → T(d, r_S) → d̃
        │     student drafts over a corpus chunk; the teacher corrects the draft
        │     USING ONLY THAT CHUNK into textbook text
        │
        └── Personalized SFT branch (CoAPT-SFT): d → q_T → a_S → T(d, q_T, a_S) → a*
              teacher authors questions from the document; student answers closed-book;
              teacher corrects the answer USING ONLY THAT DOCUMENT
                      │
                      ▼
        raw corpus + teacher CPT + QA-as-text + anchor   (token-ledgered mix)
                      ▼
        one frozen CPT training run  →  new student  →  re-diagnose  →  next round

**Principles (binding):**

1. **Corpus is the only source of truth.** Student drafts exist to expose gaps; the
   teacher corrects, rewrites, and questions but may not introduce facts absent from the
   source, even when true. Every training row traces to a corpus document.
2. **Student-conditioned supervision.** Round data is a function of the current student:
   `D_r = f(S_r, corpus, T)`. Diagnosis signals (per-doc NLL, closed-book pool probes,
   prior-round question accuracy) are re-measured every round; evaluation artifacts are
   never regenerated.
3. **Joint acquisition and utilization.** CPT text and QA pairs train together in one
   mixed stream (Instruction-Pre-Training style: QA serialized as plain
   `Question:/Answer:` text, full-token loss — the MVP deliberately has no loss masking;
   chat-format SFT consolidation via `studio/stages/sft.py` is a later, separate stage).

**Frozen round protocol:** checkpoints chain (`R0` from base, `R(N)` from `R(N-1)`); the
recipe — mix shares, templates, prompts, gate rules, pool, frontier rule, all `frozen:`
config values — is frozen across rounds of a campaign; between rounds only
`data.round`, `data.round_dir`, `run.init_from`, and `run.seed` change. The trainer is
the CPT card row (Unsloth BF16 full-parameter causal-LM + WSD; `frozen:` asserted in
code). The loop metric is `coapt_pool_absorption_dev` over a pool frozen at campaign
start; `corpus_transfer_macro_dev` is the no-regression guardrail; milestone (frozen)
splits are consulted only at phase gates.

Stage entry points that remain: `cpt` (the CoAPT trainer) and `sft` (chat-format
consolidation, deferred). RLVR, OPD, and the agentic stage were retired with the
2026-07-23 rewrite; resurrecting any of them is a spec change.

## 2. Ban list (as binding as the card)

- **Never regenerate evaluation artifacts.** The probe referee, the pool, and any frozen
  split are minted once; a "refreshed" exam is a different campaign, not a loop move.
- **The teacher never grades its own student's exam.** Referee metrics come from
  deterministic verifiers only; agent judgment gates training data and produces
  advisory diagnostics, never the keep/revert number.
- **No teacher free-knowledge.** Supervision unsupported by the source document is
  banned from the training set regardless of correctness.
- **No mid-campaign recipe drift** — no loss masking, template edits, mix retuning, or
  gate loosening between rounds. Any of these starts a new campaign.
- **No dynamic sampling inside a training run.** Between-round frontier reselection IS
  the algorithm; within-run difficulty filtering or example reweighting is not.
- Legacy bans that remain: PPO + critic, process reward models, MCTS / search
  augmentation, DPO stacking, model merging. (The former blanket ban on curriculum
  learning is lifted as of 2026-07-23: CoAPT's measured, between-round curriculum is the
  hypothesis under test. Its in-run form stays banned per the dynamic-sampling clause.)

## 3. Null-hypothesis meta-rule

> **Before any algorithm change is declared effective, it must beat the control: the same
> plain baseline trained for the same additional compute.** If both rise equally, the
> change is judged ineffective and reverted.

For CoAPT itself this binds the campaign conclusion: the loop is only declared effective
against the null of **equal-token raw-corpus CPT** (same total content tokens, same
trainer). The two-round MVP establishes that the loop closes (diagnosis responds to
training); the null-hypothesis A/B is the mandatory next campaign before any stronger
claim. Data-content changes produced by the loop itself compete at equal wall-clock as
usual — no control needed per round. The keep/revert additionally requires the effect
size to exceed the measured noise band (three-seed `max(std, half_range)` at R0).

## 4. Environment lock

Training and independent evaluation use two isolated, pinned environments because vLLM's
torch/transformers requirements differ from the validated Unsloth stack:

    training (`requirements.txt`):
      unsloth==2026.6.9   trl==0.24.0   transformers==5.5.0

    gym eval (`requirements-eval.txt`):
      vllm==0.25.1   torch==2.11.0   transformers==5.6.0   ninja==1.13.0

The training process never imports gym or vLLM. It exits after saving a checkpoint; a
fresh eval process then loads that checkpoint through the gym runner (student inference
for diagnosis uses the same eval environment via `tools/student.py`). **The agent has no
authority to change dependency versions.** Installing, bumping, or removing a dependency
is a human-approved operation. Revalidate the CoAPT smoke path end to end after any
approved change.

## 5. Explicitly not doing

- **No RL stages in this phase.** RLVR/OPD/agentic return, if ever, by spec change with
  their own cards — not by the loop.
- **No banned technique** without run-record evidence of the pathology + human approval.
- **No hyperparameter search.** The trainer's `frozen:` values are the card; the loop's
  entire search surface is the data the closed loop generates.
- **No multi-machine execution, no export work** until the MVP verdict is recorded.

## 6. Where complexity IS allowed (asymmetry principle)

Complexity saved on algorithms transfers to three places where it compounds:

1. **Verifiers and gates** (`gym/verifiers/`, branch-skill gate rules) — the referee
   stays deterministic; the gates stay strict and evidence-bound.
2. **Data recipes** — the CoAPT branch quality (draft probes, correction discipline,
   question design) is the one dimension this philosophy encourages iterating — between
   campaigns, with the recipe re-frozen each time.
3. **Measurement** — the token ledger, per-doc NLL trajectories, per-doc probe accuracy,
   frontier turnover, and gate pass rates ride along with every round. Keep/revert reads
   mechanism metrics, not just the score. Defense against gaming is measurement, never
   algorithm patches.

## 7. Terminology

- `gym/runner/` is the **runner** (batch generation + evaluation orchestration).
- The word **harness** is reserved for the agent runtime (opennekaise / nekaise-edge).
  Do not use it for anything inside `gym/` or `studio/`.
- Container identities and the gym/studio boundary live in **BOUNDARY.md** (Article 0:
  containment direction; the examination-hall / workshop definitions).

## 8. Run identity and artifact invariants

- A `run_id` is the permanent identity of one attempt. Conclusions, metrics, decisions,
  retries, and exported models reference it directly.
- Checkpoints and datasets are immutable content-addressed artifacts. Saving a later run
  never mutates an earlier artifact. `latest` and `best:<metric>` are pointer records only.
- Round artifacts (`frontier`, drafts, teacher rows, gate verdicts, token ledger) are
  retained under the round directory with content hashes recorded in the dataset spec —
  a round's dataset is reproducible from its inputs or it does not train.
- Training reaches `trained`; independent evaluation records the split hash and metric;
  explicit adjudication reaches `succeeded`. Failures/timeboxes are terminal records too.
- `experiments/.studio/runs.sqlite3` is a rebuildable query index. Per-run state, metrics,
  decisions, and artifact manifests remain the recovery source.
- Agent queries are bounded and JSON-first. No loop step scans all historical run folders.
