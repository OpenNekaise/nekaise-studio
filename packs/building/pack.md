# Task pack: building

The real target. Measures a small model's ability to do what Opus does in OpenNekaise —
**ontology/topology understanding of a building** — grounded in `nekaise_data/`.

| | |
|---|---|
| **Source** | `nekaise_data/<building>/*.ttl` (Brick / ASHRAE 223P / REC semantic models) |
| **Buildings** | one folder per building; ≥1 ontology each (parsed by [`prepare.py`](prepare.py)) |
| **Split** | **cross-building** — held-out building (`NEKAISE_HOLDOUT`, default `rio10`) = test; rest = train |
| **Metric** | `building_acc` — fraction of deterministically-verifiable questions answered correctly |
| **Scorer** | [`scorer.py`](scorer.py) — **fixed; the agent must not edit it** |

## What it measures (v0: ontology/topology)

Questions are minted from the parsed graph, so the gold is exact and ungameable:

- **type_of** — given an entity's description, name its ontology class (`brick:Air_Handler_Unit`, …).
- **count** — how many entities of a class are in the building.
- **connections** — what an entity is connected to (`s223:cnx`) — i.e. topology.

Grading: exact for counts, any-correct-class for type, fraction-of-connections for topology
(the graded `reward()` doubles as the GRPO reward).

## Why cross-building

Training and testing on the *same* building only proves memorization. Holding out a whole
building tests **generalization to an unseen building** — the real bar for "approach Opus."
With N buildings you can rotate the holdout (leave-one-building-out).

## How knowledge vs. skill is split

The pack does **not** ask the model to memorize a building. At inference the model has the
ontology available (RAG / file tools, as in nekaise-edge); fine-tuning teaches it to *use*
that grounding — read the model, answer with the right class/topology, in the right form.
Training data (teacher chain-of-thought, rejection-sampled by this scorer) is built in the
experiment's `build_data.py`; this pack only defines the fixed, verifiable evaluation.

## Future question kinds

setpoint/spec lookup with citations, tag-name normalization (Brick/223P), control-sequence
interpretation, BIM/spatial queries — each adds templates here + scorer logic, no loop change.
