# example-building — a fully synthetic building you can train against

Everything in this folder is **fictional** (the "Norrsken Demo Building", invented tags,
generated trend values). It exists so a fresh clone can walk the ENTIRE building pipeline —
`prepare-trainset` → `judge` → `train.py` → `eval_judge.py` — without any proprietary data.

Use it by copying it into the (git-ignored) data folder:

```bash
cp -r examples/example-building nekaise_data/
python packs/building/prepare.py     # index it; scorer questions come from the ontology
```

Contents mirror what a real dump looks like:

| File | Real-world counterpart |
|------|------------------------|
| `ontology.ttl` | Brick / ASHRAE 223P semantic model export |
| `points.csv` | BMS point list (vendor tags) |
| `trends/supply_air_temp.csv` | sensor trend export |
| `alarms.txt` | alarm/event log export |
| `control_card_ahu1.md` | control card / sequence of operations (text) |
| `control_card_ahu1.pdf` | the same card as a PDF (exercises the PDF text-layer path) |

Never mix this folder with real building data in training runs you care about — it is
demo material, not domain truth.
