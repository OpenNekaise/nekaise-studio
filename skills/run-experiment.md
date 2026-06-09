# Skill: run-experiment

You are running an **autoresearch loop** (after Karpathy's `autoresearch`) to improve a
small language model's score on a **task pack**, by fine-tuning it with Unsloth and editing
a single training script. You are the researcher; the human only edits *this skill*.

## The setup

An **experiment** lives in `experiments/<name>/`:

- `train.py` — **the one file you edit.** It loads a base model, fine-tunes it with Unsloth,
  evaluates on the task pack, and prints a metric line. Change architecture knobs,
  hyperparameters, LoRA config, prompt/format, data handling — anything here.
- `LOG.md` — the running journal of every experiment: hypothesis, change, metric, kept/reverted.

A **task pack** lives in `packs/<pack>/`:

- `scorer.py` and `prepare.py` — **fixed. Never edit these.** They define the dataset and the
  metric. Editing them would be cheating the fitness function.
- `pack.md` — what the pack measures and how it's scored.

## The loop

Repeat, one change at a time:

1. **Read** `experiments/<name>/LOG.md` (full history) and the current `train.py`.
2. **Hypothesize** one concrete, minimal change likely to raise the metric. State it in one sentence.
3. **Edit** only `train.py` to make that single change. Keep it reviewable.
4. **Run** it: `python experiments/<name>/train.py`. It is **time-boxed** — respect the budget;
   don't remove the time limit to train longer.
5. **Read the metric** from the printed `METRIC ...` line.
6. **Decide**:
   - **Better** than the best in `LOG.md` → keep the change. Append an entry to `LOG.md`
     (hypothesis, diff summary, metric, ✅ kept). Optionally `git commit`.
   - **Worse or equal** → revert the edit. Append the entry anyway (❌ reverted) — negative
     results are data; they stop you re-trying dead ends.
7. **Repeat** with the next hypothesis.

## What to vary (high-leverage knobs first)

- **Prompt / format** — system prompt, how the answer is delimited, chat template usage.
- **Data** — subset, ordering, packing, max sequence length, how examples are rendered.
- **LoRA** — rank `r`, `alpha`, dropout, which `target_modules`.
- **Optimization** — learning rate, schedule, warmup, weight decay, optimizer, batch / grad-accum.
- **Steps** — within the time budget, more steps vs. bigger batches.

## Rules

- **One change per run.** You can't attribute a result to two changes at once.
- **Never edit `packs/*/scorer.py` or `prepare.py`.** That's the referee.
- **Honor the time box.** Comparisons are only fair at equal wall-clock.
- **Log everything**, including failures. `LOG.md` is the memory across the whole search.
- **Best metric wins.** The job is to push the `METRIC` number, nothing else.

## Starting

Pick (or be told) an experiment, then: *"Read `LOG.md` and the current `train.py`, propose one
change, and kick off a new experiment."* If `LOG.md` is empty, your first run just establishes
the **baseline** metric — make no change, run it, and record the number.

## Dashboard

Before kicking off a run, start the local dashboard so the human can watch if they want:

```bash
python dashboard/server.py    # serves http://localhost:8765
```

Tell them the URL once. `train.py` streams loss + the final metric to it automatically (via
`dashboard/runlog.py`), and every run shows up in the leaderboard — they can ignore it or click
in. Don't block on it; it's optional telemetry, not part of the loop's decisions.

## Current targets

- Active experiment: `experiments/granite-4.1-3b-gsm8k/` (base: `unsloth/granite-4.1-3b`, pack: `gsm8k`).
- Planned: Granite-4.1-8B, Gemma-4, Qwen-3.5, and sub-1B Granite (350m/1b). One folder per model.
- The `gsm8k` pack is a **public bootstrap** — it proves the loop works. It will later be swapped
  for a building-ontology pack with no change to this skill or the loop.
