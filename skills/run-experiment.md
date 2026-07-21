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
6. **Decide**: better than the best in `LOG.md` → keep + log (✅). Worse/equal → revert the
   edit + log anyway (❌ — negative results stop you re-trying dead ends).
7. **Repeat.**

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
- **Algorithm changes must beat the null hypothesis.** Before any *algorithmic* change
  (method, loss, schedule, RL trick — anything that is not a data recipe) is kept, it must
  beat the control: **the unchanged baseline trained for the same additional compute**.
  "Adding X gained 2 points" only counts if "no X, equal extra steps" does not also gain
  2 points; log the control like any run. Data-recipe changes (`build_data.py`) compete at
  equal wall-clock as usual, no control needed. The current **algorithm card in STATUS.md**
  says which knobs are even eligible per stage; its frozen parts are human spec-review
  territory, never loop moves, and its **ban list** is binding — a banned technique enters
  only when the dashboard shows the specific pathology it treats.
- **Never edit `packs/*/{scorer,prepare}.py` or `lib/*`.** That's the fixed referee/plumbing.
  Using the scorer to *filter training data* in `build_data.py` is allowed and expected —
  it's not cheating, because **evaluation always runs on the held-out test split with the
  same fixed scorer.** What you must never do is change how the metric itself is computed.
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

**Read `STATUS.md` at the repo root** — it names the current phase, the active experiment,
the phase metric, and the next levers. This skill describes the loop; STATUS.md says where
to point it today. Durable protocol that does not change with the phase:

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
