# Experiments

The repository carries two experiments: **`coapt/`** (the active nekaise-coapt loop — its
`build_data.py` is the round toolkit: `init-pool`, `emit-docs`, `select-frontier`,
`make-drafts`, `build`) and **`cpt/`** (the raw-corpus stream builder, kept as the
equal-token control for the null-hypothesis rule, SPEC.md §3).

An experiment owns its mutable data recipe (`build_data.py`) and ignored runtime state:
immutable `data/objects/` and `runs/<run_id>/` records. The global ignored
`experiments/.studio/` directory contains the SQLite query index and checkpoint CAS. The training algorithm and
hyperparameters do not live in the experiment; they are frozen in `studio/stages/` and
asserted against the corresponding `configs/<stage>.yaml` `frozen:` section.

For a CoAPT round (driven by `skills/coapt-round.md`):

```bash
python -m studio.cli build coapt --config configs/coapt.yaml
python -m studio.cli train cpt --config configs/coapt.yaml
```

For the raw-CPT control:

```bash
python experiments/cpt/build_data.py
python -m studio.cli train cpt --config configs/cpt.yaml
```

With an external user workspace, run the same flow through the overlay:

```bash
python -m studio.cli --workspace ../nekaise-user-workspace build cpt \
  --config configs/cpt.yaml
python -m studio.cli --workspace ../nekaise-user-workspace train cpt \
  --config configs/cpt.yaml
```

The workspace's `experiments/<name>/build_data.py` overrides the core recipe when it
exists; otherwise `build` falls back to this directory. Data, runs, and artifacts always
land in the active workspace.

Direct stage entry points are implementation/debug interfaces, not the agent control
surface. During the autoresearch loop, the agent may change only
`experiments/cpt/build_data.py` or `configs/cpt.yaml`'s `data:` section.
