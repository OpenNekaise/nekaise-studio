# Nekaise Studio

> A **self-extending, AI-run training platform** for small language models. You don't run
> scripts and you don't write the training logic. You **dump data into one folder**, point
> Claude Code (or Codex) at the repo, and it trains a small, edge-deployable model for the
> **building / building-energy** domain by *following* skills — and, over time, *writing* them.

Nekaise Studio is the **model factory** behind [OpenNekaise](https://github.com/OpenNekaise/opennekaise).
It fine-tunes small models (1B–8B, and smaller) that run on-prem in a building via
[`nekaise-edge`](https://github.com/OpenNekaise/nekaise-edge) — no frontier cloud dependency.

**The mission:** distill a frontier *senior building engineer* (Claude / Opus) into a small
*junior* that understands real building systems well enough to work on-site — cheaply, offline,
grounded in the building's own data. The long-term target is **one general building-energy small
model**, locally adaptable to any specific building.

**The bigger idea (why this is more than autoML):** the apprenticeship is **run by an AI**, and
the AI doesn't only search model-space — it edits its own operating instructions. We (the
maintainers) ship a **bootloader**: a seed of skills, an eval harness, and a training recipe.
From there Claude Code decides *what to do, which experiments to run, and which new skills to
write*. We are the bootloader; the system is meant to extend itself. That is the recursive part —
bounded, today, by the frontier model driving it (which means a better Claude makes the whole
system better for free).

## The core idea: just dump data into `nekaise_data/`

The most important design decision: **there is one folder you feed, and the AI does everything
else.** Drop a building's real data into `nekaise_data/<building>/` — HVAC and control PDFs,
control cards, the ontology / semantic model (Brick / ASHRAE 223P `.ttl`), tag lists, BIM, sensor
exports — one subfolder per building, and that's it. You don't clean it, label it, or hand-write
any training data.

A second kind of dump raises the **general** ceiling: `nekaise_data/hvac_corpus/` is a maintained,
provenance-tracked corpus of public building / HVAC / building-energy reference material (Wikipedia,
arXiv, DOE/PNNL/LBNL reports, ASHRAE guides) across five topics — controls/BAS, equipment & systems,
building energy, commissioning & FDD, standards & protocols. Building-specific dumps close the
*gap* (this building); the corpus raises the *ceiling* (the domain).

> ⚠️ **`nekaise_data/` is git-ignored and never committed.** It holds proprietary, real building
> data — it stays on your machine. API keys (e.g. the teacher's `ANTHROPIC_API_KEY`) are read from
> a local `.env` and are never committed.

## How it works

```
  You: drop a building into nekaise_data/<building>/   ·   "train a junior on it"
            │
            ▼
  Claude Code / Codex  ──drive──►  skills (the program it executes AND extends):
            │
            ├─ prepare-trainset ─► senior engineer reads the WHOLE corpus
            │                      (ontology + control cards + trends), writes grounded Q&A
            ├─ judge (gate) ─────► drops any ungrounded / hallucinated example
            │
            ▼   then the autoresearch loop (skills/run-experiment), run autonomously:
   ┌─────────────────────────────────────────────────────────────┐
   │  1. propose ONE change to a recipe (train.py / build_data.py)│
   │  2. fine-tune the small model with Unsloth (cpt/sft/grpo/dpo)│
   │  3. score on the FIXED eval harness — over a HELD-OUT        │
   │     building it never trained on (+ a closed-book domain quiz)│
   │  4. better than best?  keep + log.  worse?  revert + log.    │
   │  5. crystallize what worked into a skill; repeat             │
   └─────────────────────────────────────────────────────────────┘
            │
            ▼
   serve/to_ollama.py ─► GGUF in Ollama ─► nekaise-edge  (runs on-prem, in the building)
```

(The public **`gsm8k`** pack runs the exact same loop with a one-line scorer — the bootstrap that
proves the machinery before building data is in play.)

## Architecture: bootloader vs workspace

Like an OS, the repo splits into an **immutable kernel** we maintain and a **mutable userland**
that stays on each user's machine:

| | **Bootloader** (pushed, versioned, ours) | **Workspace** (local, git-ignored, the agent's) |
|---|---|---|
| What | seed skills, the eval harness, recipe templates, `lib/`, guardrails | data dumps, generated datasets, checkpoints, run telemetry, journals, **agent-written skills** |
| Paths | `skills/` (core), `packs/`, `lib/`, `experiments/<exp>/train.py` | `nekaise_data/`, `experiments/**/{data,outputs,runs,LOG.md}` |
| Who changes it | maintainers (+ promoted contributions) | each user's Claude Code, freely |

**Skills split into two tiers.** *Core skills* (pushed, stable) are the kernel of expertise.
*Emergent skills* are what each user's Claude Code writes for its data — kept local, never pushed,
different for everyone. A skill earns its way into the bootloader through a **promotion path**:
validated against the eval harness, then reviewed as a PR. Local skills are the *mutation pool*;
promotion is the *selection* that feeds the shared kernel — without it the kernel freezes and the
project loses its network effect.

**One general model, local adaptation.** The commons is a general building-energy base model the
maintainers train from the shared corpus via the recipe. Each user's Claude Code adapts it to
*their* building locally (private, not pushed). Generic, validated improvements — skills,
benchmark items, anonymized signal — flow back by opt-in promotion and make the shared base
stronger over time. That is how private data and a shared general model coexist.

## The eval harness is the fitness function

In a self-extending loop the metric is everything: if the agent optimizes a flawed measure, it
will Goodhart it *confidently* across iterations. So the harness is the bootloader's most
load-bearing part, and it is deliberately **multi-metric** — no single number to game:

- **`building_judge` (the gap).** Answer a *frozen, realistic operator exam* about a **held-out
  building**, open-book over its data; a blind frontier judge scores each answer by the fraction
  of its required **anchors** (values, vendor tags, file paths, component names, time windows)
  present. Measures whether the junior can *use this building's data*. (`eval_judge.py` /
  `skills/judge`.)
- **`domain_quiz` (the ceiling).** A **closed-book** multiple-choice exam over general
  building/HVAC/energy knowledge (easy + hard tiers). Measures what's in the *weights*, with no
  retrieval — exactly what `building_judge` cannot see. (`eval_domain.py`.)

Two hard-won lessons, now part of the method: **perplexity is not knowledge** (continued
pretraining can cut held-out perplexity sharply while adding ~zero closed-book accuracy — it buys
fluency, not facts), and **a single building-specific score can't tell you if the model got
domain-smarter** — you need the ceiling probe alongside the gap probe.

## Layout

| Path | What it is |
|------|-----------|
| `nekaise_data/<building>/` | **The one folder you feed** — raw building data per building. **Git-ignored.** |
| `nekaise_data/hvac_corpus/` | Maintained general building/HVAC/energy corpus (ceiling material); `build_corpus.py` + `manifest.jsonl`. |
| `skills/` | **The product.** Driver-agnostic skills: `prepare-trainset`, `judge`, `run-experiment` (core today; meta-skills for crystallize/prune are the next seeds). |
| `.claude/skills/`, `AGENTS.md` | Thin adapters so **both Claude Code and Codex** use the same skills. |
| `experiments/<model>-<pack>/` | Editable recipes — `build_data.py` (WHAT data) + `train.py` (HOW: cpt/sft/grpo/dpo) — plus `eval_judge.py`, `eval_domain.py`, `domain_quiz*.jsonl`. `data/`, `outputs/`, `runs/`, `LOG.md` are git-ignored. |
| `packs/<pack>/` | A **task pack** = data + `scorer.py` (the fixed referee) + the frozen exam. **Never edited.** |
| `lib/` | Fixed plumbing: `pack.py`, `datakit.py` (dataset cache + provenance), `corpus.py` (load + retrieve), `llm.py` (teacher backends). |
| `dashboard-ui/` | Zero-config Vite live dashboard reading `experiments/**/runs/`. |
| `serve/` | Export a winning model to GGUF → Ollama (handoff to `nekaise-edge`). |

## Methods

The recipe is a *seed*, not a fixed pipeline — the agent improves it. What's validated so far:

- **CPT** — continued (next-token) pretraining on the corpus to raise the domain ceiling. LoRA or
  **full-parameter** (a 3B fits ~26 GB on a 48 GB card with 8-bit Adam + grad checkpointing). Must
  be followed by SFT to restore instruction-following. *Lesson: it lifts fluency, not closed-book
  knowledge — the base is already domain-strong.*
- **SFT** — supervised fine-tune on grounded, judge-gated teacher demos.
- **GRPO** — anchor-recall reinforcement learning: verifiable reward = fraction of required anchors
  the answer hits. No reward model, no reward hacking on the fact itself.
- **DPO** — preference optimization (available).

**The validated chain so far:** grounded gated SFT → anchor-recall GRPO closed **67% of the
student→teacher gap** on a held-out building (`building_judge` 0.36 → 0.54). Whether ceiling-raising
CPT adds to that, and where to invest (ceiling vs gap), is what the harness now decides per dataset.

## Design principles

- **The agent is the user.** No human runs Python; you direct agents through skills.
- **We are the bootloader.** We seed skills + harness + recipe; the agent extends them. The durable
  core we guard is the **eval harness**, not any one training trick.
- **Edit the recipe, not the referee.** Recipes (`train.py`, `build_data.py`) and emergent skills
  are mutable; scorers, held-out split, and frozen exams are fixed so the metric can't be gamed.
- **The metric is the boss — and it's plural.** A gap probe *and* a ceiling probe; perplexity is
  never the success metric.
- **Generalize, don't memorize.** Train on some buildings, grade on a **held-out** one. Bake the
  general domain into weights; ground the specific building at inference.
- **Proprietary data stays local; validated skills can flow back.** Real data and exams are
  git-ignored; emergent skills are local until promoted by PR.
- **Small only.** Target <8B: Granite 4.1 (3B/8B), Gemma, Qwen, down to sub-1B. Start:
  **`unsloth/granite-4.1-3b`**.

## Dashboard

A zero-config, fully-local live dashboard (Vite + React, Scandinavian light theme) reads the
per-run telemetry directly off disk — no separate backend, no wandb, no CDN:

```bash
cd dashboard-ui && npm install && npm run dev    # http://localhost:5273
```

Run list with live status, training-loss / reward curves, before→after vs baseline, and CPT
perplexity / domain results. Reachable over Tailscale by adding your MagicDNS name to
`server.allowedHosts`.

## Stack

- **Training:** [Unsloth](https://unsloth.ai) — fast LoRA / full fine-tuning on a single GPU.
- **Serving:** [Ollama](https://ollama.com) (llama.cpp) — what `nekaise-edge` runs.
- **Drivers:** Claude Code and Codex, via the skills in `skills/`.

## Getting started

You mostly don't run things by hand. Point Claude Code or Codex at the repo and say:

> *"Read `skills/run-experiment`, then look at `experiments/granite-4.1-3b-gsm8k/` — establish the
> baseline, then start improving it."*

(First run: prime the cache once with `python packs/gsm8k/prepare.py`. Needs a CUDA GPU +
`pip install -r requirements.txt`.)

## Method

[`docs/METHOD.md`](docs/METHOD.md) — the playbook for driving the loop toward **teacher parity**:
diagnose before you optimize (measure the teacher *and* the untrained base), separate ceiling from
gap, open-book = data-in-hand + retrieval, task-aligned teacher distillation, and the pitfalls.

## Related

- [opennekaise](https://github.com/OpenNekaise/opennekaise) — the cloud Nekaise Agent.
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the loop this is modeled on.

## License

MIT — see [LICENSE](LICENSE).
