# Nekaise Studio

> An **agentic training platform** for small language models (<8B). You don't run scripts —
> you talk to Claude Code or Codex, and it improves the models for you, autoresearch-style.

Nekaise Studio is the **model factory** behind [OpenNekaise](https://github.com/OpenNekaise/opennekaise).
It fine-tunes small models (1B–8B, and smaller) that can eventually run on-prem in a building
via [`nekaise-edge`](https://github.com/OpenNekaise/nekaise-edge) — no frontier cloud dependency.

**The mission:** distill a frontier *senior building engineer* (Claude / Opus) into a small
*junior* that understands a real building's systems well enough to work on-site — cheaply, with
no cloud. The bar is generalization: the junior is graded on a **building it never trained on**.

The twist: the whole apprenticeship is **run by an AI**. Following Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch), the platform is a tight loop —
**propose a change → fine-tune → score → keep or revert → repeat** — that Claude Code or Codex
drives by itself. And it isn't only the loop: the AI also **prepares the training data** (as the
senior engineer) and **judges** its quality. Every role is a **skill**, not commands you type.

## The core idea: just dump data into `nekaise_data/`

The most important design decision in Nekaise Studio: **there is one folder you feed, and the
AI does everything else.** Drop a building's real data into `nekaise_data/<building>/` — HVAC
and control PDFs, control cards, the ontology / semantic model (Brick / ASHRAE 223P `.ttl`),
tag lists, BIM, sensor exports — one subfolder per building, and that's it. You don't clean it,
label it, or hand-write any training data.

From there the agent does the rest. Reading the **full** corpus — ontology, control cards, trend
data, alarms — Claude Code (as a senior engineer, via the `prepare-trainset` skill) **designs the
dataset itself**: grounded Q&A that teach a junior to operate *that* building. A second skill,
`judge`, throws out anything not grounded in the real data. Then the autoresearch loop below
fine-tunes a small model and scores it on a **held-out building** it never saw.

> ⚠️ **`nekaise_data/` is git-ignored and never committed.** It holds proprietary, real
> building data — it stays on your machine. Only code and prompts live in this repo. Likewise,
> API keys (e.g. the teacher's `ANTHROPIC_API_KEY`) are read from a local `.env` and are never
> committed.

## How it works

```
  You: drop a building into nekaise_data/<building>/   ·   "train a junior on it"
            │
            ▼
  Claude Code / Codex  ──drive──►  three markdown skills (the program):
            │
            ├─ prepare-trainset ─► senior engineer reads the WHOLE corpus
            │                      (ontology + control cards + trends), writes grounded Q&A
            ├─ judge (gate) ─────► drops any ungrounded / hallucinated example
            │
            ▼   then the autoresearch loop (skills/run-experiment.md), run autonomously:
   ┌─────────────────────────────────────────────────────────────┐
   │  1. propose ONE change to a recipe (train.py / build_data.py)│
   │  2. fine-tune the small model with Unsloth (time-boxed)      │
   │  3. score on packs/building/scorer.py — the FIXED referee,   │
   │     over a HELD-OUT building it never trained on             │
   │  4. better than best?  keep + log.  worse?  revert + log.    │
   │  5. repeat — hundreds of experiments overnight               │
   └─────────────────────────────────────────────────────────────┘
            │
            ├─ judge (eval) ─────► advisory score on open-ended questions
            ▼
   serve/to_ollama.py ─► GGUF in Ollama ─► nekaise-edge  (runs on-prem, in the building)
```

(The public **`gsm8k`** pack runs the exact same loop with a one-line scorer — it's the bootstrap
that proves the machinery before the building data is in play.)

## Layout

| Path | What it is |
|------|-----------|
| `nekaise_data/<building>/` | **The one folder you feed.** Raw building data — PDFs, ontologies (`.ttl`), tag lists, BIM, sensor exports. One subfolder per building. **Git-ignored; never committed.** |
| `skills/` | **The product.** Three driver-agnostic skills: `prepare-trainset` (senior engineer writes grounded data), `judge` (gate bad data + open-ended eval), `run-experiment` (the loop). |
| `.claude/skills/`, `AGENTS.md` | Thin adapters so **both Claude Code and Codex** use the same skills. |
| `experiments/<model>-<pack>/` | Two editable recipe files — `build_data.py` (WHAT data: distill / rejection-sample) + `train.py` (HOW: sft/grpo) — plus `LOG.md`. Built datasets (`data/`) and checkpoints (`outputs/`) are git-ignored. |
| `packs/<pack>/` | A **task pack** = data + `scorer.py` (the fixed referee: `load_split`/`is_correct`/`reward`). **Never edited.** `packs/building/` scores ontology/topology Q&A from `nekaise_data`. |
| `lib/` | Fixed plumbing: `pack.py` (load a pack), `datakit.py` (dataset cache + provenance), `llm.py` (teacher backends: ollama / anthropic / openai). |
| `serve/` | Export a winning model to GGUF → Ollama (the handoff to `nekaise-edge`). |

## Design principles

- **The agent is the user.** No human runs Python; you direct agents through skills — for data,
  judging, and the training loop alike.
- **Edit the recipe, not the referee.** The agent edits the experiment's recipes (`train.py`,
  `build_data.py`) and authors data via the `prepare-trainset` skill. The scorer, the held-out
  split, and the frozen exam are fixed — so the metric can't be gamed.
- **The metric is the boss.** A deterministic scorer decides keep/revert; the LLM `judge` is an
  advisory second opinion, never the deciding vote.
- **Generalize, don't memorize.** Train on some buildings, grade on a **held-out** one — the real
  "approach Opus" bar.
- **Proprietary data stays local.** Real building data, built datasets, and the exam are
  git-ignored and never named in tracked files; only code and prompts live in this repo.
- **The mission pack is here; gsm8k just proves the loop.** `packs/gsm8k/` is the public
  bootstrap; `packs/building/` is the real target — same loop, same skills, swap the pack.
- **Small only.** Target <8B: Granite 4.1 (3B/8B), Gemma 4, Qwen 3.5, down to sub-1B Granite.
  Starting point: **`unsloth/granite-4.1-3b`**.

## Dashboard

A zero-dependency, fully-local web dashboard ships by default. The agent starts it when it runs
an experiment; you click in only if you want to watch:

```bash
python dashboard/server.py        # http://localhost:8765
```

It's a live view of the **loop**, not just one curve — training-loss chart, before/after vs the
baseline, run status, and a leaderboard of every experiment (kept vs reverted). Fed by a tiny
`events.jsonl` each run writes; no TensorBoard, no wandb, no CDN.

## Stack

- **Training:** [Unsloth](https://unsloth.ai) — fast LoRA fine-tuning on a single GPU.
- **Serving:** [Ollama](https://ollama.com) (built on llama.cpp) — what `nekaise-edge` runs.
- **Drivers:** Claude Code and Codex, via the skills in `skills/`.

## Getting started

You mostly don't run things by hand. Point Claude Code or Codex at the repo and say:

> *"Read `skills/run-experiment.md`, then look at `experiments/granite-4.1-3b-gsm8k/` — establish
> the baseline, then start improving it."*

(For the first run, prime the dataset cache once: `python packs/gsm8k/prepare.py`. Requires a
CUDA GPU + `pip install -r requirements.txt`.)

## Related

- [opennekaise](https://github.com/OpenNekaise/opennekaise) — the cloud Nekaise Agent.
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the loop this is modeled on.

## License

MIT — see [LICENSE](LICENSE).
