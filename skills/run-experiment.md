# Skill: run-experiment

You are running an **autoresearch loop** (after Karpathy's `autoresearch`) to improve a
small language model's score on a **task pack**. You fine-tune with Unsloth and edit the
experiment's two recipe files; the human only edits *this skill*.

> **Related skills.** For the `building` pack, training data is authored by the
> **`prepare-trainset`** skill (a senior engineer reading the real corpus) and filtered/graded
> by the **`judge`** skill. This loop consumes their output (`data/LATEST`) and decides
> keep/revert on the **deterministic** `METRIC` only — `building_judge` is advisory.

## The setup

An **experiment** lives in `experiments/<name>/` and has **two editable recipe files**:

- `train.py` — **HOW** to train. Picks a `METHOD` (sft / dpo / grpo), a dataset, and an
  init checkpoint; trains, evaluates on the pack, saves to `outputs/<stage>/`, prints a
  `METRIC` line. Tune hyperparameters, LoRA, optimizer, prompt/format, method.
- `build_data.py` — **WHAT** to train on. The teacher authors the realistic questions an engineer
  or operator asks (with grounded answers + `anchors`), built into open-book SFT demos + the frozen
  exam (`eval_judge.py` grades by anchors). Change the questions-per-building, prompts, teacher.
- `LOG.md` — the running journal of every run: hypothesis, change, metric, kept/reverted.

A **task pack** lives in `packs/<pack>/` and is the **fixed referee** — `scorer.py` and
`prepare.py`. **Never edit these.** `scorer.py` exposes the contract every method shares:
`load_split(split, n)`, `is_correct(pred, gold)`, `reward(pred, gold)`, `extract_answer`.

Shared plumbing in `lib/` (`pack.py`, `datakit.py`, `llm.py`) is also fixed — you call it,
you don't edit it.

## The loop

Repeat, one change at a time:

1. **Read** `experiments/<name>/LOG.md` (full history) and the current `train.py` /
   `build_data.py`.
2. **Hypothesize** one concrete, minimal change likely to raise the metric. One sentence.
3. **Edit** the smallest thing in **one** recipe file (`train.py` *or* `build_data.py`).
4. **Run** it (time-boxed — don't remove the limit):
   - changed `build_data.py`? → `python experiments/<name>/build_data.py` (builds + caches a
     dataset artifact), then run `train.py` with `DATASET="auto"`.
   - changed only `train.py`? → `python experiments/<name>/train.py`.
5. **Read the metric** from the printed `METRIC ...` line.
6. **Decide by the constitution (SPEC.md §3 + R5)** — not by eyeball:
   `studio.tools.explog.decide(value, best, noise_band)` is the rule: **keep ⇔ effect
   size > the measured noise band** (`experiments/<exp>/noise_band.json`, from
   `python -m studio.tools.variance_check`). No band measured → the verdict is `pending`,
   not keep: run variance_check first. An *algorithm* change additionally needs the
   **null-hypothesis control** (SPEC §3): same baseline, equal extra compute, as its own
   logged run — both rising equally = change ineffective, revert.
7. **Log both formats** (R8): `studio.tools.explog.append(...)` writes the schema'd
   `log.jsonl` row (hypothesis / variable / expectation / result / noise_band / verdict /
   confidence) AND the human-readable `LOG.md` line. Tag records that support a reusable
   finding with `finding: <slug>` — the crystallize gate counts those.
8. **Repeat.**

## Methods you can reach (all use the same fixed referee)

- **SFT** — `METHOD="sft"`. Train on an SFT dataset. `DATASET="render"` = SFT on the pack's
  gold (the baseline); `DATASET="auto"` = SFT on whatever `build_data.py` last built.
- **RFT / rejection sampling** — in `build_data.py`, set `source` to the student and
  `n_samples>1`; keep its correct samples; then SFT (`DATASET="auto"`).
- **Distillation** — in `build_data.py`, set `source` to a teacher (`anthropic:claude-…` or
  `ollama:qwen3.6:27b`); keep its correct chain-of-thought; then SFT.
- **GRPO / DPO** — `METHOD="grpo"` uses the pack's graded `reward()` as a verifiable reward;
  `METHOD="dpo"` trains on preference pairs.

**Pipelines hand off through checkpoints.** Each run saves to `outputs/<STAGE>/` and updates
`outputs/best.json`. To polish an SFT model with RL: run SFT (`STAGE="sft"`), then run again
with `METHOD="grpo"`, `INIT_FROM="outputs/sft"`, `STAGE="grpo"`. The reference recipe —
*teacher CoT → reject-sample → SFT → GRPO* — is just: edit `build_data.py` (distill) → run →
`train.py METHOD=sft DATASET=auto` → `train.py METHOD=grpo INIT_FROM=outputs/sft`.

## What to vary (high-leverage first)

Subject to the current **algorithm card** in STATUS.md — vary only the card's movable knobs
for the active stage; everything below is the generic menu.

- **Data** (`build_data.py`) — teacher vs student source, #samples, prompt/CoT style, size,
  filtering. Often the biggest lever, and the one dimension the card always leaves open.
- **Method** — sft → rft → grpo. With a verifiable scorer, RFT/GRPO usually beat tuning SFT.
- **Prompt / format**, **LoRA** (r, alpha, dropout, target_modules), **optimization** (lr,
  schedule, warmup, batch/grad-accum), **steps** within the time budget.

## Rules

- **One change per run**, in one recipe file. You can't attribute a result to two changes.
- **SPEC.md is the constitution — read it before proposing anything.** The **algorithm
  card** (SPEC §1) says which knob is even eligible per stage (the `data:` section /
  `build_data.py`; everything `frozen:` is human spec-review, never a loop move — the
  stage entry point refuses to run on drift). The **ban list** (SPEC §2) is binding: a
  banned technique enters only with dashboard-shown pathology AND human approval. The
  **environment lock** (SPEC §4): you have no authority to change dependency versions.
- **Algorithm changes must beat the null hypothesis (SPEC §3).** Before any *algorithmic*
  change (method, loss, schedule, RL trick — anything that is not a data recipe) is kept,
  it must beat the control: **the unchanged baseline trained for the same additional
  compute**. "Adding X gained 2 points" only counts if "no X, equal extra steps" does not
  also gain 2 points; log the control like any run (variable="null-control: ...").
  Data-recipe changes compete at equal wall-clock as usual, no control needed.
- **Never edit `gym/`** (tasks + verifiers + runner — the single definition of correct),
  **`packs/*/scorer.py`** (thin shims over gym), **or `lib/*`.** Using a gym verifier to
  *filter training data* is allowed and expected — it's not cheating, because
  **evaluation always runs on the held-out split with the same verifier.** What you must
  never do is change how the metric itself is computed.
- **Honor the time box.** Comparisons are only fair at equal wall-clock.
- **Log everything**, including failures. `LOG.md` is the memory across the whole search.
- Built datasets are **cached + provenance-tracked** (`data/<id>/provenance.json`): same
  recipe reuses the cache, a changed recipe makes a new artifact. Don't regenerate by hand.
- **Best metric wins.** Push the `METRIC` number, nothing else.

## Starting

Pick (or be told) an experiment, then: *"Read `LOG.md`, `train.py`, and `build_data.py`;
propose one change; run it."* If `LOG.md` is empty, the first run establishes the
**baseline** — make no change, run `train.py` with `DATASET="render"`, record the number.

## Dashboard

Before a run, start the local dashboard so the human can watch if they want:

```bash
cd dashboard-ui && npm install && npm run dev    # serves http://localhost:5273
```

Tell them the URL once. Training streams loss (SFT) or reward/kl (GRPO) plus the final eval
metric; every run shows up in the leaderboard. It's optional telemetry, not part of the
loop's decisions — don't block on it.

## Current targets

**Read `SPEC.md` (the constitution) and `STATUS.md` (the current phase) at the repo
root.** SPEC binds every decision (card / ban list / null hypothesis / environment lock);
STATUS names the active experiment, the phase metric, and the next levers. Stage entry
points are `python -m studio.stages.<cpt|sft|rlvr|opd> --config configs/<stage>.yaml`
(`--dry-run` validates the frozen section). Durable protocol that does not change with
the phase:

- For matrices of runs (N models × M treatments), drive them with
  `python tools/campaign.py <spec.json>` — declarative, resumable, results into the ledger —
  instead of ad-hoc shell loops.
- Every eval/train result lands in `experiments/<name>/results.jsonl` (lib/results.py);
  `LOG.md` stays the narrative of hypotheses and verdicts.
- When a metric has a frozen split (e.g. the bench `test` split), it is milestone-only,
  never keep/revert. Compare against the base model, and report paired flips (+n/−n), not
  just aggregates — deltas are usually small relative to eval noise.
- Bootstrap: `experiments/granite-4.1-3b-gsm8k/` — public pack, works on a bare clone;
  proves the loop end-to-end.
- `gsm8k` is a **public bootstrap** that proves the loop. It will later swap for a
  building-ontology pack with **no change to this skill, the loop, or the recipe files'
  structure** — only the pack import changes.
- To serve a winner: `python serve/to_ollama.py --exp <name> --name <ollama-name>` exports
  `outputs/best` to GGUF/Ollama, which `nekaise-edge` then runs (`OLLAMA_MODEL=<ollama-name>`).
- `nekaise_bench` is **milestone-only in every loop** (decoupling reform 2026-07-20): never a
  keep/revert signal. At a phase gate, run `tools/eval_bench.py` (frozen `test` needs
  `--milestone`; full official harness via `--model <ollama-name>` after export, for deployment
  parity) and report paired flips vs base. Results record the bench dataset version — compare
  only within one version. The loop metric is whatever STATUS.md names for the phase (pure-CPT:
  `corpus_probes` dev-absorption via `tools/eval_probes.py`; its `frozen` split is likewise
  milestone-only). Checkpoint-mode and Ollama-mode numbers are not interchangeable; compare
  like with like.
