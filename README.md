# Nekaise Studio

**An AI agent that fine-tunes small LLMs — no human runs Python here.**

[![ci](https://github.com/OpenNekaise/nekaise-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenNekaise/nekaise-studio/actions/workflows/ci.yml)

Point Claude Code (or Codex) at this repo and it runs the whole training lifecycle for
small (≤8B) **building-energy** models: builds the data, trains, measures, keeps or
reverts, and writes down what it learned — then improves its own instructions. The
resulting models run on-prem via [nekaise-edge](https://github.com/OpenNekaise/nekaise-edge),
no frontier cloud dependency.

## How it works — nekaise-coapt

The training algorithm is **CoAPT** (Co-Adaptive Pretraining and Tuning): the current
student's measured state selects the data, the agent teaches against real corpus text,
and the trained student re-selects the next round's data.

```
      ┌──────────────────────────────────────────────────────────────┐
      │  diagnose    student NLL + closed-book probes over the pool  │
      │      ↓                                                       │
      │  teach       student drafts / answers → agent corrects,      │
      │              grounded ONLY in the source document (gated)    │
      │      ↓                                                       │
      │  mix         raw + teacher CPT + QA-text + anchor            │
      │              (token-ledgered shares)                         │
      │      ↓                                                       │
      │  train       python -m studio.cli train cpt (frozen stage)   │
      │      ↓                                                       │
      │  measure     probe referee → METRIC + noise band             │
      │      ↓                                                       │
      │  keep ⇔ effect > band, guardrail intact       (else revert)  │
      │      ↓                                                       │
      │  re-diagnose the NEW student → next round ───────────────────┼──▶ the loop
      └──────────────────────────────────────────────────────────────┘     closes
```

The trainer is deliberately boring and **frozen** — one stock Unsloth full-parameter CPT
recipe, no tricks. That's the philosophy: an agent-run loop converges only when the knobs
are few, so all creativity is spent where complexity compounds — **the data the loop
generates, verifiers, and measurement** — and none where it doesn't.

## Try it

```bash
python tools/doctor.py                                            # preflight
python -m studio.cli integrity                                   # query-store preflight
# then point Claude Code at the repo — it reads AGENTS.md and drives the loop itself
```

For a clean open-source installation, keep agent-created work outside the bootloader:

```bash
python -m studio.cli workspace init ../nekaise-user-workspace
python -m studio.cli --workspace ../nekaise-user-workspace integrity
```

See [docs/WORKSPACE.md](docs/WORKSPACE.md) for the overlay, isolation, and promotion model.

## Map

| where | what |
|---|---|
| [SPEC.md](SPEC.md) | the constitution: algorithm card, ban list, null-hypothesis rule, environment lock |
| [BOUNDARY.md](BOUNDARY.md) | what studio is, what gym is, and which contains which |
| [STATUS.md](STATUS.md) | current phase + next levers — the only part that changes |
| [gym/](gym/) | the examination hall: tasks + verifiers + runner ([own README](gym/README.md)) |
| [studio/stages/](studio/stages/) + [configs/](configs/) | frozen training entry points (cpt, sft) + the CoAPT round recipe |
| [studio/cli.py](studio/cli.py) + [lib/runstore.py](lib/runstore.py) | JSON agent API, SQLite run index, immutable artifacts |
| [docs/RUNSTORE.md](docs/RUNSTORE.md) | run lifecycle, artifact and retention protocol |
| [docs/WORKSPACE.md](docs/WORKSPACE.md) | isolated user overlays for configs, recipes, skills, runs, and artifacts |
| [skills/](skills/) | the agent's operating manual |
| [STATUS.md](STATUS.md) | active bring-up phase and the next executable checks |

Part of [OpenNekaise](https://opennekaise.com/). MIT.

*AI agents: start at [AGENTS.md](AGENTS.md), not here.*
