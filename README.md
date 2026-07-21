# Nekaise Studio

**An AI agent that fine-tunes small LLMs — no human runs Python here.**

[![ci](https://github.com/OpenNekaise/nekaise-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenNekaise/nekaise-studio/actions/workflows/ci.yml)

Point Claude Code (or Codex) at this repo and it runs the whole training lifecycle for
small (≤8B) **building-energy** models: builds the data, trains, measures, keeps or
reverts, and writes down what it learned — then improves its own instructions. The
resulting models run on-prem via [nekaise-edge](https://github.com/OpenNekaise/nekaise-edge),
no frontier cloud dependency.

## How it works

```
      ┌────────────────────────────────────────────────────────┐
      │  propose ONE change            (data recipe only)      │
      │      ↓                                                 │
      │  train      python -m studio.stages.<stage>            │
      │      ↓                                                 │
      │  measure    gym verifiers → METRIC + noise band        │
      │      ↓                                                 │
      │  keep ⇔ effect > noise band          (else revert)     │
      │      ↓                                                 │
      │  log → replicate ×2 → crystallize into a new skill ────┼──▶ the loop improves
      └────────────────────────────────────────────────────────┘      itself
```

The algorithms are deliberately boring and **frozen** — five stages (CPT → SFT → RLVR →
OPD → agentic) on stock Unsloth/TRL trainers, JustRL-style single recipes, no tricks.
That's the philosophy: an agent-run loop converges only when the knobs are few, so all
creativity is spent where complexity compounds — **data recipes, verifiers, and
measurement** — and none where it doesn't.

## Try it

```bash
python tools/doctor.py                                            # preflight
python -m studio.stages.cpt --config configs/cpt.yaml --dry-run   # validate the card
# then point Claude Code at the repo — it reads AGENTS.md and drives the loop itself
```

## Map

| where | what |
|---|---|
| [SPEC.md](SPEC.md) | the constitution: algorithm card, ban list, null-hypothesis rule, environment lock |
| [BOUNDARY.md](BOUNDARY.md) | what studio is, what gym is, and which contains which |
| [STATUS.md](STATUS.md) | current phase + next levers — the only part that changes |
| [gym/](gym/) | the examination hall: tasks + verifiers + runner ([own README](gym/README.md)) |
| [studio/stages/](studio/stages/) + [configs/](configs/) | five frozen training entry points |
| [skills/](skills/) | the agent's operating manual |
| [docs/RESULTS.md](docs/RESULTS.md) | published, reproducible results (public assets only) |

Part of [OpenNekaise](https://opennekaise.com/). MIT.

*AI agents: start at [AGENTS.md](AGENTS.md), not here.*
