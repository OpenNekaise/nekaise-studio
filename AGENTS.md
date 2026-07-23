# AGENTS.md — Nekaise Studio

Nekaise Studio is an **agentic fine-tuning platform**: you (Claude Code or Codex) improve
a small language model (<8B) by running **nekaise-coapt** — Co-Adaptive Pretraining and
Tuning — a closed loop driven entirely through skills, no human runs Python directly. The
current student's measured state selects the data, YOU are the teacher (grounded strictly
in real corpus text), and the trained student re-selects the next round's data.

## How to work in this repo

**First run on a fresh clone:** `python tools/doctor.py` — a zero-dep preflight that checks
GPU, packages, `.env`, data, and the holdout. Fix what it flags before training.
Then read **`STATUS.md`** for the current campaign, round, and next actions — this
file and the skills describe the durable procedure; STATUS.md is the part that changes.

The skills are the program you execute — and, over time, extend. Each is canonical under
`skills/`, mirrored to `.claude/skills/`. The CoAPT loop is three skills:

- **[`skills/coapt-round.md`](skills/coapt-round.md)** — the *conductor*; read it first
  when training. One round: diagnose the student (`tools/student.py` NLL + pool probes) →
  loop-closure checks (R1+) → select the frontier → run both branches → build the
  token-ledgered mix → train through the frozen CPT stage (chained checkpoints) →
  evaluate → decide with the noise band → round report.
- **[`skills/coapt-cpt.md`](skills/coapt-cpt.md)** — the *Adaptive CPT branch*
  (`d → r_S → T(d, r_S) → d̃`): the student drafts over frontier corpus chunks to expose
  its gaps; you correct each draft into textbook prose supported ONLY by the source
  chunk, then gate claim-by-claim.
- **[`skills/coapt-sft.md`](skills/coapt-sft.md)** — the *Personalized SFT branch*
  (`d → q_T → a_S → T(d, q_T, a_S) → a*`): you author evidence-backed questions from the
  document, the student answers closed-book, you correct into `a*` and gate. The training
  pair `(q_T, a*)` is serialized as plain `Question:/Answer:` text into the same mix.

The studio also **improves itself** through two meta-skills — this is the recursive part,
and the reason we are a bootloader rather than a fixed pipeline:

- **[`skills/crystallize-skill.md`](skills/crystallize-skill.md)** — turn a *validated,
  reusable* finding from a round into a new local skill in the active workspace's
  `skills/local/`. The **mutation**. Do it at the end of a cycle, not for hunches.
- **[`skills/prune-skills.md`](skills/prune-skills.md)** — review and consolidate the
  workspace's `skills/local/`: merge duplicates, drop stale / contradicted / overfit
  advice. The **selection**.

Dormant building-phase skills (kept, not part of the CoAPT loop):
[`skills/prepare-trainset.md`](skills/prepare-trainset.md) and
[`skills/judge.md`](skills/judge.md) return when the proprietary-building phase resumes.

## Hard rules

- **BOUNDARY.md Article 0:** studio contains gym, never the reverse — "extract the gym"
  means carving a component out of studio, not renaming studio into gym. A repo's name
  must equal its contents; gym's contents never include training code. When writing
  instructions about container relationships, state both directions explicitly.
- **SPEC.md is the constitution** — the CoAPT algorithm card, ban list, null-hypothesis
  meta-rule, environment lock. Every keep/revert is bound by it; changing SPEC (or any
  `frozen:` config section) is human review, never a loop move.
- **The recipe is frozen across rounds.** Between rounds only `data.round`,
  `data.round_dir`, `run.init_from`, and `run.seed` change in `configs/coapt.yaml`.
  Recipe changes (mix, templates, gates, pool, frontier rule) happen between CAMPAIGNS,
  through the normal one-change discipline. Stage implementations and training
  hyperparameters are fixed.
- **Corpus is the only source of truth.** Teacher output unsupported by the source
  document never enters the training set, even when correct. Every training row traces
  to a corpus document; transfer/frozen probe documents are never visible to any stream.
- **Never** edit `gym/` (tasks + verifiers + runner — the single definition of correct,
  one-way dependency studio → gym), `packs/*/scorer.py` (thin shims over gym),
  `packs/building/eval_open.jsonl`, or `lib/*` — that's the fixed referee/plumbing.
  Never regenerate the pool, the probes, or any frozen split; never open
  `gym/tasks/corpus_probes/probes.jsonl` while teaching. The referee is reached only
  through `tools/eval_probes.py`.
- **Privacy (proprietary partner data):** never write a real building/partner/address
  name into any *tracked* file (skills, prompts, code, commit messages). Refer to
  buildings generically. `nekaise_data/`, `experiments/**/data/`, and
  `packs/**/eval_open.jsonl` are git-ignored — keep them so; never `git add -f` them.
- One loop change per round. Honor the time box. Log every run, including failures.
- **Always set `NEKAISE_HOLDOUT` explicitly** (in `.env`, see `.env.example`) before any
  building-pack work. If unset, the scorer silently falls back to the first folder by
  name — which can grade against the wrong building and leak training data into the exam.

## Bootloader vs workspace

Like an OS, the repo is an immutable **bootloader** we maintain + a mutable **workspace**
that stays on your machine. Prefer an external overlay initialized with
`python -m studio.cli workspace init <path>` and selected by the global
`--workspace <path>` option or `NEKAISE_WORKSPACE`. With neither set, the legacy ignored
directories in this checkout remain active:

- **Bootloader (pushed):** core `skills/*.md`, the eval referee (`gym/`, `packs/*`,
  `tools/eval_probes.py`), `lib/*`, the round toolkit and config templates, guardrails.
  The most load-bearing piece is the **referee** — it is the fitness function the
  self-improving loop trusts; getting it right matters more than any single training
  trick. Metrics are split by ROLE: the **loop metric** (`coapt_pool_absorption_dev`,
  dense, phase-picked in `STATUS.md`) drives keep/revert with
  `corpus_transfer_macro_dev` as the no-regression guardrail; **milestone referees**
  (frozen probe split; `nekaise_bench` via `tools/eval_bench.py`) are consulted only at
  phase gates and never optimized against. Diagnosis signals (NLL, draft quality,
  question accuracy) stay advisory. Never one number to game; perplexity is never a
  success metric (loss curves are diagnostics — probe accuracy is what counts as
  knowledge).
- **Workspace (yours):** an external overlay owns config overrides, round directories
  (`coapt/rounds/r<N>` with drafts, teacher rows, gate verdicts, token ledger),
  immutable `experiments/**/{data,runs}`, `.studio/` SQLite + artifact CAS, proprietary
  `nekaise_data/`, `scratch/`, `extensions/`, and `skills/local/`. The legacy equivalents
  in this checkout remain supported. See `docs/WORKSPACE.md`.
- **Promotion:** a local skill enters the shared kernel only by surviving the eval
  referee and a human-reviewed PR. Local skills are the mutation pool; promotion is the
  selection that feeds the commons. Don't encode a fresh finding by editing core
  `skills/` / `packs/` / `lib/` in place — crystallize it locally, then propose it.

## Map

- `SPEC.md` — the constitution (bootloader): CoAPT algorithm card, ban list,
  null-hypothesis rule, environment lock. `BOUNDARY.md` — its sibling: gym/studio
  definitions + containment direction (Article 0).
- `gym/` — tasks (`gym/tasks/`, triple-checked intake: prompt + reference solution +
  verifier, gold must pass), verifiers (`gym/verifiers/`, pure `verify(prompt, response,
  meta) -> float`), runner (`gym/runner/`, vLLM offline + OpenAI-compatible server — the
  word "harness" is reserved for the agent runtime). `gym/tools/`: calibrate, monitors.
- `studio/` — agent JSON CLI (`python -m studio.cli`) + stages `cpt` (the CoAPT trainer)
  and `sft` (chat consolidation, deferred; frozen sections asserted) + tools
  (variance_check, typed decisions, crystallize_gate ≥2-experiment admission).
- `configs/` — one frozen config per stage plus `coapt.yaml` (the round recipe; the
  agent's movable surface is the data the loop generates, not the recipe).
- `skills/` — core source-of-truth skills (mirrored under `.claude/skills/`):
  `coapt-round` (the loop), `coapt-cpt` + `coapt-sft` (the branches),
  `crystallize-skill` + `prune-skills` (the self-improvement meta-loop),
  `prepare-trainset` + `judge` (dormant building phase). The active workspace's
  `skills/local/` holds emergent skills.
- `experiments/` — `coapt/build_data.py` (the round toolkit: init-pool, emit-docs,
  select-frontier, make-drafts, build) and `cpt/build_data.py` (the raw-corpus stream
  builder, kept for the null-hypothesis control). `data/objects/<digest>` stores
  immutable datasets; `runs/<run_id>` stores manifest/state/events/process output;
  `.studio/runs.sqlite3` indexes everything and `.studio/artifacts/checkpoints/<digest>`
  stores immutable weights. `latest`/`best` are database pointers, never directories.
- `packs/` — fixed task packs: dataset + scorer (`load_split` / `is_correct` / `reward`).
- `lib/` — fixed plumbing: `runstore.py` (SQLite index + immutable artifacts),
  `datakit.py` (dataset CAS), `runlog.py` (trainer adapter), `corpusprep.py` (shared
  corpus cleaning), `pack.py`, and `llm.py` (scripted-teacher fallback).
- `tools/` — `doctor.py` (preflight), `student.py` (batch student inference: score /
  draft / answer, eval env), `eval_probes.py` (the referee + `--doc-ids` pool view),
  `privacy_check.py` (leak guard; pre-commit hook via `git config core.hooksPath
  tools/hooks`), `eval_bench.py` (independent milestone metric, advisory),
  `build_probes.py` (maintainer-only referee minting).
- `serve/` — resolve a run/best pointer, then export to GGUF + Ollama.
- `tests/` — CPU-only guardrails for the fixed parts (referee contract, plumbing,
  holdout guard, skills mirror, CoAPT mix ledger). Keep green (`pytest`); CI runs them
  plus the privacy scan.
- `examples/example-building/` — fully synthetic building; `cp -r` it into
  `nekaise_data/` to exercise the building pipeline without proprietary data. Never mix
  into real training runs.

## Agent interface

Do not scan run directories or parse Markdown. Use the bounded JSON interface:

    python -m studio.cli [--workspace <path>] list --experiment coapt --limit 20
    python -m studio.cli [--workspace <path>] show <run_id> --events-tail 20
    python -m studio.cli [--workspace <path>] resolve --run-id <run_id>
    python -m studio.cli [--workspace <path>] integrity

One run ID names one immutable training attempt. A run conclusion must use that ID and its
checkpoint digest; mutable `latest:<experiment>` aliases are only pipeline inputs. Direct
stage entry points remain implementation/debug interfaces, not the agent workflow.
