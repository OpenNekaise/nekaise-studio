# Task pack: building

The real target. Measures a small model's ability to do what Opus does in OpenNekaise —
**ontology/topology understanding of a building** — grounded in `nekaise_data/`.

| | |
|---|---|
| **Source** | `nekaise_data/<building>/*.ttl` (Brick / ASHRAE 223P / REC semantic models) |
| **Buildings** | one folder per building; ≥1 ontology each (parsed by [`prepare.py`](prepare.py)) |
| **Split** | **cross-building** — held-out building (`NEKAISE_HOLDOUT`) = test; rest = train |
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

## Open-ended eval (judge track)

The deterministic scorer above only grades synthetic graph questions (type / count / connections)
— **not how a real user asks**. The **primary** metric is the **realistic exam**: the questions a
building engineer or operator actually asks, graded by the **`judge` skill** against a **frozen
exam** `packs/building/eval_open.jsonl` (git-ignored; derives from the proprietary holdout
building). One JSON object per line:

```json
{"id": "...", "persona": "engineer|operator", "intent": "<free-form>", "question": "...", "ground_truth": "<correct grounded answer>", "anchors": ["<must-match fact: value / tag / file path / component / window / checklist item>", "..."], "source": "<file / entity>"}
```

(No fixed category taxonomy — `intent` is free-form, for coverage/reporting only.) It is authored
**once** by the `prepare-trainset` skill (holdout building only) and then **frozen** like this
scorer. The judge scores each answer by the **fraction of its `anchors`** present (strict on the
fact — values/tags/paths/components/windows — lenient on phrasing), so multi-part answers get
honest partial credit; `building_judge ∈ [0,1]` is reported with per-`intent` means. The
deterministic `building_acc` above is demoted to a **cheap sanity check**.

## Future question kinds

`action` tasks (fetch/align time series, fit a model), multi-step troubleshooting, BIM/spatial
queries — each adds realistic templates to the exam; the loop and skills don't change.
