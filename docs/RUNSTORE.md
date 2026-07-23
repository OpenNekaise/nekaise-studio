# Agent run store

This is a single-machine protocol for coding agents, not a human experiment UI. Its goals
are stable identity, bounded JSON queries, crash evidence, immutable artifacts, and cheap
retention across tens of thousands of runs.

## Contract

One attempt gets one collision-resistant, time-sortable `run_id`. The lifecycle is:

    queued → running → trained → evaluating → succeeded
                        └───────────────→ failed
    running → aborted (timebox, checkpoint may exist)

Training commits an immutable checkpoint and stops at `trained`. Gym evaluates that exact
run, storing evaluator version and split hash. `studio.cli decide` applies the noise-band
rule and moves the run to `succeeded`. A retry is a new run, never a reset of the old one.

## Storage

In an external [user workspace](WORKSPACE.md), all paths below are rooted there. With
no workspace selected, the legacy repository paths remain unchanged:

    .studio/                                # external workspace
    experiments/.studio/                    # legacy equivalent
      runs.sqlite3                         rebuildable SQLite/WAL query index
      artifacts/checkpoints/<sha256>/     immutable model/tokenizer directories
      pointers/<experiment>/*.json        recovery copies of latest/best pointers
    experiments/<experiment>/
      data/objects/<digest>/              immutable JSONL dataset + provenance
      data/specs/<spec-id>.json           recipe-cache pointer
      data/plans/<name>/                   immutable campaign membership/order snapshot
      data/LATEST_SPEC                     active recipe identity (not just content identity)
      runs/<run_id>/
        manifest.json                     immutable launch identity
        state.json                        atomically replaced current state
        events.jsonl                      append-only scalar trainer events
        metrics/*.json                    idempotent evaluation records
        decision.json                     one typed adjudication
        process.log                       complete launcher output

SQLite is the only query surface but not the only copy. Rebuild it from those files with
`python -m studio.cli reindex --reset`. Artifact files are never placed inside SQLite.
The `dataset_specs` index is many-to-one: two recipes may intentionally produce identical
bytes while retaining distinct provenance. A named stream plan isolates a campaign from
concurrent corpus ingestion, so scale prefixes remain reproducible while collectors append
new documents.

## Agent commands

All result-bearing commands emit JSON. Agents must preserve `run_id` rather than infer an
experiment from filenames.

    python -m studio.cli train cpt --config configs/cpt.yaml
    python -m studio.cli list --experiment cpt --status trained --limit 50
    python -m studio.cli show <run_id> --events-tail 20 --log-tail 40
    python -m studio.cli show <run_id> --full       # provenance/config only when needed
    python -m studio.cli resolve --run-id <run_id>
    $NEKAISE_EVAL_PYTHON tools/eval_probes.py --run-id <run_id> --split dev
    python -m studio.cli decide <run_id> --metric corpus_transfer_macro_dev --noise-band 0.01
    python -m studio.cli integrity
    python -m studio.cli gc --keep-last 2            # plan only
    python -m studio.cli gc --keep-last 2 --apply    # delete eligible weights

Add `--workspace <path>` before the subcommand, or set `NEKAISE_WORKSPACE`, to bind every
command in the train/eval/decide/export chain to the same isolated store. Run provenance
contains the stable workspace ID; database paths use workspace-relative references rather
than machine-specific absolute paths.

`latest:<experiment>` is allowed only as a downstream stage input. Conclusions, evals,
exports, comparisons, and skill evidence use an exact run ID.

`list` and `show` deliberately return compact records so a large campaign does not consume
the agent's context window. `show --full` is the explicit escape hatch for provenance,
environment, and complete configuration.

`train` writes two NDJSON control records to stdout: `train_started` immediately exposes
the allocated run ID, then `train_result` reports the terminal launch outcome. Child-process
output goes to `process.log`; add `--follow` only for live debugging.

## Retention

Run manifests, state, scalar events, metrics, and decisions are retained. They are small
and are the scientific record. Full checkpoints dominate disk, so GC keeps every pinned
or pointer-referenced checkpoint plus the latest N checkpoints per experiment. Deleting
weights changes the artifact state to `deleted` but leaves the producing run and digest.

Successful training scratch is removed after the checkpoint commits. Failed run scratch
and `process.log` remain for diagnosis. Dataset objects are immutable; future dataset GC
must remove only objects unreferenced by any run and not pinned.

## Scale boundary

SQLite/WAL and local files are intentional: one machine has at most a small number of
concurrent GPU writers. If execution becomes multi-host, replace only the index/artifact
backends with PostgreSQL/object storage; keep the run/artifact protocol unchanged.
