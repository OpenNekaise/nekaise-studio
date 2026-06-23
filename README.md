# Nekaise Studio

> An **agentic training platform** for small language models (<8B). You don't run scripts —
> you talk to Claude Code or Codex, and it improves the models for you, autoresearch-style.

Nekaise Studio is the **model factory** behind [OpenNekaise](https://github.com/OpenNekaise/opennekaise).
It fine-tunes small models (1B–8B, and smaller) that can eventually run on-prem in a building
via [`nekaise-edge`](https://github.com/OpenNekaise/nekaise-edge) — no frontier cloud dependency.

The twist: the interface *is* an AI agent. Following Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch), the platform is a tight loop —
**propose a change → fine-tune → score → keep or revert → repeat** — that Claude Code or Codex
drives by itself. Everything you'd normally do as an ML researcher lives in a **skill**, not in
commands you type.

## The core idea: just dump data into `nekaise_data/`

The most important design decision in Nekaise Studio: **there is one folder you feed, and the
AI does everything else.** Drop a building's real data into `nekaise_data/<building>/` — HVAC
and control PDFs, control cards, the ontology / semantic model (Brick / ASHRAE 223P `.ttl`),
tag lists, BIM, sensor exports — one subfolder per building, and that's it. You don't clean it,
label it, or hand-write any training data.

From there the agent does the rest: it reads the data, **designs the dataset itself** (e.g.
Opus 4.8 prompted as a senior building engineer, authoring English Q&A grounded in each
building's real ontology to train a junior engineer), fine-tunes a small model on it, and
scores the result on a held-out building — the autoresearch loop below.

> ⚠️ **`nekaise_data/` is git-ignored and never committed.** It holds proprietary, real
> building data — it stays on your machine. Only code and prompts live in this repo. Likewise,
> API keys (e.g. the teacher's `ANTHROPIC_API_KEY`) are read from a local `.env` and are never
> committed.

## How it works

```
  You: "improve granite-3b on the gsm8k pack"
            │
            ▼
  Claude Code / Codex  ──reads──►  skills/run-experiment.md   (the program — a markdown skill)
            │
            ▼   the loop, run autonomously:
   ┌─────────────────────────────────────────────────────────────┐
   │  1. read LOG.md + train.py                                   │
   │  2. propose ONE change, edit experiments/<m>/train.py        │
   │  3. run it  →  Unsloth fine-tune (time-boxed)                │
   │  4. score on packs/<pack>/scorer.py   (the fixed referee)    │
   │  5. better than best?  keep + log.  worse?  revert + log.    │
   │  6. repeat — hundreds of experiments overnight               │
   └─────────────────────────────────────────────────────────────┘
            │
            ▼
   serve/to_ollama.py  ──►  GGUF in Ollama  ──►  nekaise-edge
```

## Layout

| Path | What it is |
|------|-----------|
| `nekaise_data/<building>/` | **The one folder you feed.** Raw building data — PDFs, ontologies (`.ttl`), tag lists, BIM, sensor exports. One subfolder per building. **Git-ignored; never committed.** |
| `skills/` | **The product.** Driver-agnostic markdown skills — the loop's instructions. |
| `.claude/skills/`, `AGENTS.md` | Thin adapters so **both Claude Code and Codex** use the same skills. |
| `experiments/<model>-<pack>/` | Two editable recipe files — `build_data.py` (WHAT data: distill / rejection-sample) + `train.py` (HOW: sft/grpo) — plus `LOG.md`. Built datasets (`data/`) and checkpoints (`outputs/`) are git-ignored. |
| `packs/<pack>/` | A **task pack** = data + `scorer.py` (the fixed referee: `load_split`/`is_correct`/`reward`). **Never edited.** `packs/building/` scores ontology/topology Q&A from `nekaise_data`. |
| `lib/` | Fixed plumbing: `pack.py` (load a pack), `datakit.py` (dataset cache + provenance), `llm.py` (teacher backends: ollama / anthropic / openai). |
| `serve/` | Export a winning model to GGUF → Ollama (the handoff to `nekaise-edge`). |

## Design principles

- **The agent is the user.** No human runs Python; you direct an agent through a skill.
- **One mutable file.** The agent only edits `train.py`. Everything else (data prep, scorer) is
  fixed so it can't game the metric.
- **The metric is the boss.** Each task pack defines an automatic score; the loop only climbs it.
- **Task packs are swappable.** v0 is the public **`gsm8k`** pack (proves the loop works). Later
  it swaps for a **building-ontology** pack — *the loop, skill, and `train.py` don't change.*
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
