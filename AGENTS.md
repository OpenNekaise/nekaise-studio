# AGENTS.md — Nekaise Studio

Nekaise Studio is an **agentic fine-tuning platform**: you (Claude Code or Codex) improve
small language models (<8B) by running an autoresearch loop, driven entirely through skills
— no human runs Python directly.

## How to work in this repo

**First run on a fresh clone:** `python tools/doctor.py` — a zero-dep preflight that checks
GPU, packages, `.env`, building data, and the holdout. Fix what it flags before training.
Then read **`STATUS.md`** for the current phase, active experiment, and next levers — this
file and the skills describe the durable procedure; STATUS.md is the part that changes.

The skills are the program you execute — and, over time, extend. Each is canonical under
`skills/`, mirrored to `.claude/skills/`. Three core jobs:

- **[`skills/prepare-trainset.md`](skills/prepare-trainset.md)** — be the *senior engineer*:
  read a building's **full** corpus under `nekaise_data/` (ontology + control cards + guides +
  trends + alarms) and write grounded training data for the small "junior" model via
  `lib/datakit`. Excludes the holdout building. Also authors, once, the frozen open-ended exam.
- **[`skills/judge.md`](skills/judge.md)** — be the *examiner*: **gate** hallucinated training
  pairs at build time, and **eval** the student on the frozen open-ended exam (advisory second
  metric). Always grades against a fixed reference; never replaces the deterministic scorer.
- **[`skills/run-experiment.md`](skills/run-experiment.md)** — be the *loop*; read it first when
  training:
  1. Pick an experiment in `experiments/<name>/`.
  2. Read its `LOG.md` and the editable recipe files: `train.py` (HOW) + `build_data.py` (WHAT —
     the teacher authors realistic engineer/operator Q&A + anchors; `eval_judge.py` grades by anchors).
  3. Propose **one** change, edit one recipe file.
  4. Run it (rebuild the dataset, then `train.py`; time-boxed) and read the `METRIC` line.
  5. Keep the change if it beats the best in `LOG.md`; otherwise revert. Log either way.

The studio also **improves itself** through two meta-skills — this is the recursive part, and the
reason we are a bootloader rather than a fixed pipeline:

- **[`skills/crystallize-skill.md`](skills/crystallize-skill.md)** — turn a *validated, reusable*
  finding from a run into a new local skill in `skills/local/`. The **mutation**. Do it at the end
  of a cycle, not for hunches or building-specific facts.
- **[`skills/prune-skills.md`](skills/prune-skills.md)** — review and consolidate `skills/local/`:
  merge duplicates, drop stale / contradicted / overfit advice. The **selection**.

## Hard rules

- **SPEC.md is the constitution** — algorithm card, ban list, null-hypothesis meta-rule,
  environment lock. Every keep/revert is bound by it; changing SPEC (or any `frozen:`
  config section) is human review, never a loop move.
- Edit **only** the experiment's recipe files (`train.py` / `build_data.py`) or a stage
  config's **`data:` section** (`configs/<stage>.yaml`). Data is also produced by the
  `prepare-trainset` skill (which writes `data/` via `datakit`).
- **Never** edit `gym/` (tasks + verifiers + runner — the single definition of correct,
  one-way dependency studio → gym), `packs/*/scorer.py` (thin shims over gym),
  `packs/building/eval_open.jsonl`, or `lib/*` — that's the fixed referee/plumbing. Using the scorer to *filter training data* is allowed (eval
  still runs on the held-out test split with the same scorer); never change how the metric is
  computed. The judge's `building_judge` is advisory — never the keep/revert decision.
- **Privacy (proprietary partner data):** never write a real building/partner/address name into
  any *tracked* file (skills, prompts, code, `LOG.md`, commit messages, dashboard). Refer to
  buildings generically. `nekaise_data/`, `experiments/**/data/`, and `packs/**/eval_open.jsonl`
  are git-ignored — keep them so; never `git add -f` them.
- Never generate training data from the holdout building. One change per run. Honor the time
  box. Log every run, including failures.
- **Always set `NEKAISE_HOLDOUT` explicitly** (in `.env`, see `.env.example`) before any
  building-pack work. If unset, the scorer silently falls back to the first folder by name —
  which can grade against the wrong building and leak training data into the exam.

## Bootloader vs workspace

Like an OS, the repo is an immutable **bootloader** we maintain + a mutable **workspace** that
stays on your machine:

- **Bootloader (pushed):** core `skills/*.md`, the eval harness (`packs/*`, `eval_judge.py`,
  `eval_domain.py`), `lib/*`, recipe templates, guardrails. The most load-bearing piece is the
  **eval harness** — it is the fitness function the self-improving loop trusts; getting it right
  matters more than any single training trick. The harness is plural on purpose, and split by
  ROLE: **loop metrics** (dense, studio-owned, phase-picked in `STATUS.md` — e.g.
  `corpus_probes` for pure-CPT phases, `building_acc` for the building pack) drive keep/revert;
  **milestone referees** (`nekaise_bench` via `tools/eval_bench.py` — independently authored,
  hardened, version-recorded; frozen splits everywhere) are consulted only at phase gates and
  never optimized against. `building_judge` and `domain_quiz` stay advisory diagnostics. Never
  one number to game; perplexity is never a success metric (loss curves are diagnostics —
  probe/exam accuracy is what counts as knowledge).
- **Workspace (git-ignored, yours):** `nekaise_data/`, `experiments/**/{data,outputs,runs,LOG.md}`,
  `workspace/` (throwaway scratch), and `skills/local/` (emergent skills you write).
- **Promotion:** a local skill enters the shared kernel only by surviving the eval harness and a
  human-reviewed PR. Local skills are the mutation pool; promotion is the selection that feeds the
  commons. Don't encode a fresh finding by editing core `skills/` / `packs/` / `lib/` in place —
  crystallize it locally, then propose it.

## Map

- `SPEC.md` — the constitution (bootloader): algorithm card, ban list, null-hypothesis
  rule, environment lock, JustRL→TRL mapping. `REFACTOR.md` is the executed charter;
  deviations in `docs/REFACTOR-NOTES.md`.
- `gym/` — tasks (`gym/tasks/`, triple-checked intake: prompt + reference solution +
  verifier, gold must pass), verifiers (`gym/verifiers/`, pure `verify(prompt, response,
  meta) -> float`), runner (`gym/runner/`, vLLM offline + OpenAI-compatible server — the
  word "harness" is reserved for the agent runtime). `gym/tools/`: calibrate (difficulty
  bands), monitors (anchor-density hacking watch).
- `studio/` — stages (`python -m studio.stages.<stage> --config configs/<stage>.yaml`;
  frozen sections asserted) + tools (explog schema'd LOG, variance_check noise band,
  crystallize_gate ≥2-experiment admission).
- `configs/` — one frozen config per stage; the agent edits `data:` sections only.
- `attic/` — pre-refactor code kept unreferenced (old HF-generate eval paths).

- `skills/` — core source-of-truth skills (mirrored under `.claude/skills/`): `prepare-trainset`
  (data prep), `judge` (gate + eval), `run-experiment` (loop), `crystallize-skill` + `prune-skills`
  (the self-improvement meta-loop). `skills/local/` holds your emergent skills (git-ignored).
- `experiments/` — two editable recipe files (`train.py` = method/hyperparameters,
  `build_data.py` = distilled/rejection-sampled data) + `LOG.md`. Runtime, git-ignored:
  `data/<id>/` (cached datasets + provenance), `outputs/<stage>/` (checkpoints + `best.json`).
- `packs/` — fixed task packs: dataset + scorer (`load_split` / `is_correct` / `reward`).
- `lib/` — fixed plumbing: `pack.py` (load a pack), `datakit.py` (dataset cache+provenance),
  `llm.py` (teacher/sampler backends: ollama/anthropic/openai), `runlog.py` (run telemetry
  the dashboard reads).
- `serve/` — export the winning checkpoint (`outputs/best`) to GGUF + Ollama (handoff to nekaise-edge).
- `tools/` — maintainer utilities, not part of the loop: `doctor.py` (preflight),
  `privacy_check.py` (leak guard; pre-commit hook via `git config core.hooksPath tools/hooks`),
  `eval_bench.py` (independent third metric, advisory).
- `tests/` — CPU-only guardrails for the fixed parts (referee contract, plumbing, holdout
  guard, skills mirror). Keep green (`pytest`); CI runs them plus the privacy scan.
- `examples/example-building/` — fully synthetic building; `cp -r` it into `nekaise_data/` to
  exercise the building pipeline without proprietary data. Never mix into real training runs.
- `dashboard-ui/` — optional live dashboard: `cd dashboard-ui && npm install && npm run dev`.
