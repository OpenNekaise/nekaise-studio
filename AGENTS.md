# AGENTS.md — Nekaise Studio

Nekaise Studio is an **agentic fine-tuning platform**: you (Claude Code or Codex) improve
small language models (<8B) by running an autoresearch loop, driven entirely through skills
— no human runs Python directly.

## How to work in this repo

Three skills, three jobs (each canonical under `skills/`, mirrored to `.claude/skills/`):

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
  2. Read its `LOG.md` and the editable recipe files: `train.py` (HOW) + `build_data.py` (WHAT,
     the cheap TTL-only fallback; the richer data path is the `prepare-trainset` skill).
  3. Propose **one** change, edit one recipe file.
  4. Run it (rebuild the dataset, then `train.py`; time-boxed) and read the `METRIC` line.
  5. Keep the change if it beats the best in `LOG.md`; otherwise revert. Log either way.

## Hard rules

- Edit **only** the experiment's recipe files: `train.py` and/or `build_data.py`. Data is also
  produced by the `prepare-trainset` skill (which writes `data/` via `datakit`).
- **Never** edit `packs/*/{scorer,prepare}.py`, `packs/building/eval_open.jsonl`, or `lib/*` —
  that's the fixed referee/plumbing. Using the scorer to *filter training data* is allowed (eval
  still runs on the held-out test split with the same scorer); never change how the metric is
  computed. The judge's `building_judge` is advisory — never the keep/revert decision.
- **Privacy (proprietary partner data):** never write a real building/partner/address name into
  any *tracked* file (skills, prompts, code, `LOG.md`, commit messages, dashboard). Refer to
  buildings generically. `nekaise_data/`, `experiments/**/data/`, and `packs/**/eval_open.jsonl`
  are git-ignored — keep them so; never `git add -f` them.
- Never generate training data from the holdout building. One change per run. Honor the time
  box. Log every run, including failures.

## Map

- `skills/` — the source-of-truth skills (Claude Code mirrors them under `.claude/skills/`):
  `prepare-trainset` (agentic data prep), `judge` (gate + open-ended eval), `run-experiment` (loop).
- `experiments/` — two editable recipe files (`train.py` = method/hyperparameters,
  `build_data.py` = distilled/rejection-sampled data) + `LOG.md`. Runtime, git-ignored:
  `data/<id>/` (cached datasets + provenance), `outputs/<stage>/` (checkpoints + `best.json`).
- `packs/` — fixed task packs: dataset + scorer (`load_split` / `is_correct` / `reward`).
- `lib/` — fixed plumbing: `pack.py` (load a pack), `datakit.py` (dataset cache+provenance),
  `llm.py` (teacher/sampler backends: ollama/anthropic/openai).
- `serve/` — export the winning checkpoint (`outputs/best`) to GGUF + Ollama (handoff to nekaise-edge).
