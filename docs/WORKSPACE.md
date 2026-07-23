# User workspace

Nekaise Studio separates the shared framework from each installation's mutable work.
The Git repository is the reviewed **bootloader**; a user workspace is an untracked
**overlay** with its own identity and complete scientific state.

Initialize one once:

```bash
python -m studio.cli workspace init ../nekaise-user-workspace
python -m studio.cli --workspace ../nekaise-user-workspace workspace show
```

The same selection can be made with
`NEKAISE_WORKSPACE=/absolute/path/to/nekaise-user-workspace`. If neither is set, the
legacy in-repository layout remains active, so existing campaigns require no migration.

## Layout

```text
nekaise-user-workspace/
  .nekaise-workspace.json       stable workspace ID + schema
  .studio/                      SQLite index, artifact CAS, pointers, exports
  configs/                      user-owned overlays copied from core templates
  nekaise_data/                 proprietary local building inputs
  experiments/<name>/
    build_data.py               optional user recipe overriding the core recipe
    data/                       immutable dataset objects and plans
    runs/                       manifests, state, events, metrics, decisions, logs
  skills/local/                 emergent, evidence-gated skills
  extensions/                   private integrations; never auto-imported
  scratch/                      disposable eval output and temporary work
  AGENTS.md                     generated workspace-specific agent boundary
```

Config and recipe lookup is an overlay: workspace first, then the core repository.
Runtime output always goes to the active workspace. A run records the workspace ID and
schema, while checkpoint and dataset paths are stored relative to the workspace so the
whole directory can move to another disk.

Useful agent commands:

```bash
python -m studio.cli --workspace ../nekaise-user-workspace build cpt \
  --config configs/cpt.yaml
python -m studio.cli --workspace ../nekaise-user-workspace train cpt \
  --config configs/cpt.yaml
python -m studio.cli --workspace ../nekaise-user-workspace list --limit 20
python -m studio.cli --workspace ../nekaise-user-workspace integrity
```

`build` chooses `experiments/<name>/build_data.py` from the workspace when present and
otherwise uses the core recipe. Stage implementations, gym verifiers, and frozen config
sections always come from the bootloader.

## Extension boundary

Agents may freely create recipes, configs, skills, notes, and private integrations in the
workspace. Nothing under it is part of the core project or appears in the core Git status.
`extensions/` is intentionally not auto-imported: executing local code must be an explicit
recipe/config/command choice, so opening an unfamiliar workspace cannot silently inject
code into the trainer or referee.

Promotion back to the shared project is a separate action: copy only the minimal,
sanitized source change into a branch, run the fixed tests/privacy scan, and open a
human-reviewed PR. Runtime data, paths, secrets, checkpoints, and local skills never move
implicitly.

## Single-machine scale

Every workspace has one SQLite/WAL index and content-addressed artifact store. `list` and
`show` stay bounded and do not scan run folders, so tens of thousands of experiments add
rows and small record files rather than process memory. Large checkpoint retention is
handled independently by pointer-aware `gc`; `reindex --reset` rebuilds SQLite from the
immutable run and artifact records.
