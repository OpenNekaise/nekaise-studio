"""JSON-first control surface for coding agents.

Examples:
    python -m studio.cli train cpt --config configs/cpt.yaml
    python -m studio.cli list --experiment cpt --limit 20
    python -m studio.cli show <run_id> --events-tail 10
    python -m studio.cli resolve --run-id <run_id>
    python -m studio.cli integrity
    python -m studio.cli gc --keep-last 2          # dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))

from runstore import RunStore, TERMINAL, new_run_id  # noqa: E402
from workspace import ENV as WORKSPACE_ENV  # noqa: E402
from workspace import REPO_ENV, Workspace  # noqa: E402


def emit(value) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _train_result(store: RunStore, run_id: str) -> dict:
    """Return the small launch result; `show` owns the detailed representation."""
    run = store.get_run(run_id)
    result = {
        "type": "train_result",
        "run_id": run_id,
        "experiment": run["experiment"],
        "stage": run["stage"],
        "status": run["status"],
        "exit_code": run.get("exit_code"),
        "dataset_id": run.get("dataset_id"),
        "checkpoint_digest": run.get("checkpoint_digest"),
        "checkpoint": (
            str(store.resolve_stored_path(run["checkpoint_path"]))
            if run.get("checkpoint_path") else None
        ),
        "event_count": run.get("event_count", 0),
    }
    if run.get("error"):
        result["error"] = run["error"]
    return result


def _show_summary(store: RunStore, run_id: str) -> dict:
    run = store.get_run(run_id, related=True)
    fields = (
        "run_id", "experiment", "stage", "kind", "status", "created", "updated",
        "started", "ended", "model", "model_revision", "dataset_id", "seed",
        "error", "exit_code", "checkpoint_digest", "checkpoint_path", "latest_step",
        "event_count", "last_metrics", "parent_run_id", "retry_of",
    )
    result = {field: run.get(field) for field in fields}
    result["metrics"] = [{
        key: metric.get(key) for key in (
            "name", "value", "n", "split", "split_hash", "evaluator",
            "evaluator_version", "diagnostic", "created",
        )
    } for metric in run["metrics"]]
    result["artifacts"] = [{
        key: artifact.get(key) for key in (
            "digest", "kind", "role", "path", "size_bytes", "state", "pinned",
            "created",
        )
    } for artifact in run["artifacts"]]
    decision = run.get("decision")
    if decision:
        result["decision"] = {
            key: decision.get(key) for key in (
                "baseline_run_id", "metric", "value", "best_before", "noise_band",
                "verdict", "reason", "created",
            )
        }
    else:
        result["decision"] = None
    return result


def train(args) -> int:
    from studio.stages import _common

    workspace = Workspace.resolve(REPO)
    config_path = workspace.resolve_config(args.config)
    cfg = _common.load_config(config_path)
    run = cfg["run"]
    experiment = run["experiment"]
    run_id = new_run_id()
    store = RunStore(REPO)
    stage_module = f"studio.stages.{args.stage}"
    command = [sys.executable, "-m", stage_module, "--config", str(config_path)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.dataset_id:
        command += ["--dataset-id", args.dataset_id]
    store.create_run(
        experiment=experiment, stage=run.get("stage", args.stage),
        kind=run.get("kind", "experiment"),
        model=str(run.get("init_from") or run.get("base_model") or ""),
        seed=args.seed if args.seed is not None else run.get("seed"),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        command=command, metadata={"launcher": "studio.cli"},
        run_id=run_id, status="queued",
    )
    run_dir = store.run_dir(experiment, run_id)
    process_log = run_dir / "process.log"
    emit({"type": "train_started", "run_id": run_id, "experiment": experiment,
          "stage": run.get("stage", args.stage), "process_log": str(process_log)})
    sys.stdout.flush()
    env = {**os.environ, "NEKAISE_RUN_ID": run_id}
    try:
        process = subprocess.Popen(
            command, cwd=REPO, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        with process_log.open("a") as log:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if args.follow and not args.quiet:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        code = process.wait()
    except BaseException as exc:
        if store.get_run(run_id)["status"] not in TERMINAL:
            store.fail(run_id, exc)
        raise

    current = store.get_run(run_id)
    if code != 0:
        if current["status"] not in TERMINAL:
            store.fail(run_id, f"stage process exited with code {code}", exit_code=code)
        else:
            store.update_run(run_id, exit_code=code)
    elif current["status"] in {"queued", "running"}:
        store.fail(run_id, "stage exited 0 without committing a checkpoint", exit_code=code)
        code = 1
    else:
        store.update_run(run_id, exit_code=code)
    emit(_train_result(store, run_id))
    return code


def build(args) -> int:
    """Run a workspace recipe, falling back to the shared core recipe."""
    workspace = Workspace.resolve(REPO)
    script = workspace.resolve_experiment_file(args.experiment, "build_data.py")
    if not script.is_file():
        raise SystemExit(
            f"no build_data.py for experiment {args.experiment!r} in workspace or core")
    config = workspace.resolve_config(args.config)
    command = [sys.executable, str(script), "--config", str(config)]
    env = {
        **os.environ,
        REPO_ENV: str(REPO),
        "PYTHONPATH": os.pathsep.join(
            [str(REPO), str(REPO / "lib"), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
    }
    completed = subprocess.run(
        command, cwd=workspace.root, env=env, capture_output=True, text=True)
    emit({
        "type": "build_result",
        "experiment": args.experiment,
        "workspace_id": workspace.workspace_id,
        "script": str(script),
        "config": str(config),
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    })
    return completed.returncode


def workspace_init(args) -> int:
    workspace = Workspace.initialize(
        REPO, args.path, copy_configs=not args.no_copy_configs)
    emit({"type": "workspace", "action": "initialized", **workspace.describe()})
    return 0


def workspace_show(args) -> int:
    emit({"type": "workspace", "action": "show", **Workspace.resolve(REPO).describe()})
    return 0


def decide(args) -> int:
    from studio.tools import explog

    store = RunStore(REPO)
    run = store.get_run(args.run_id)
    measured = store.get_metric(args.run_id, args.metric, args.split)
    if measured is None or measured["value"] is None:
        raise SystemExit(f"run {args.run_id} has no non-diagnostic {args.metric}/{args.split}")
    if args.baseline_run_id:
        baseline = store.get_run(args.baseline_run_id)
        if baseline["experiment"] != run["experiment"]:
            raise SystemExit("baseline run belongs to a different experiment")
        previous = store.get_metric(args.baseline_run_id, args.metric, args.split)
        if previous is None or previous["value"] is None:
            raise SystemExit(
                f"baseline {args.baseline_run_id} has no non-diagnostic "
                f"{args.metric}/{args.split}")
    else:
        previous = store.best_metric(
            run["experiment"], args.metric, args.split, exclude_run_id=args.run_id,
            direction=args.direction)
    if previous is None:
        verdict = {"verdict": "baseline", "delta": None, "reason": "first measured run"}
    else:
        if args.direction == "max":
            verdict = explog.decide(measured["value"], previous["value"], args.noise_band)
        else:
            verdict = explog.decide(-measured["value"], -previous["value"], args.noise_band)
            if verdict.get("delta") is not None:
                verdict["delta"] = round(previous["value"] - measured["value"], 6)
    store.record_decision(
        args.run_id, metric=args.metric, value=measured["value"],
        verdict=verdict["verdict"], reason=verdict["reason"],
        best_before=previous["value"] if previous else None,
        noise_band=args.noise_band,
        baseline_run_id=args.baseline_run_id,
        metadata={"split": args.split, "delta": verdict.get("delta"),
                  "direction": args.direction, "finding": args.finding,
                  "hypothesis": args.hypothesis},
    )
    if verdict["verdict"] in {"baseline", "keep"} and run.get("checkpoint_digest"):
        store.set_pointer(run["experiment"], f"best:{args.metric}", args.run_id,
                          run["checkpoint_digest"])
    if run["status"] in {"trained", "evaluating"}:
        store.transition(args.run_id, "succeeded")
    emit({"type": "decision", "run_id": args.run_id, "metric": args.metric,
          "split": args.split,
          "baseline_run_id": previous["run_id"] if previous else None, **verdict})
    return 0


def invalidate(args) -> int:
    """Replace a run's decision with an explicit invalidation; never delete evidence."""
    store = RunStore(REPO)
    run = store.get_run(args.run_id, related=True)
    previous = run.get("decision") or {}
    metric = args.metric or previous.get("metric") or "invalid_run"
    store.record_decision(
        args.run_id, metric=metric, value=previous.get("value"), verdict="invalid",
        reason=args.reason, best_before=previous.get("best_before"),
        noise_band=previous.get("noise_band"),
        baseline_run_id=previous.get("baseline_run_id"),
        metadata={"invalidated_by": "studio.cli", "previous_verdict": previous.get("verdict")},
    )
    emit({"type": "decision", "run_id": args.run_id, "metric": metric,
          "verdict": "invalid", "reason": args.reason})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", metavar="PATH",
        help="use an initialized external user workspace (or set NEKAISE_WORKSPACE)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("workspace", help="initialize or inspect a user workspace")
    workspace_sub = p.add_subparsers(dest="workspace_command", required=True)
    q = workspace_sub.add_parser("init", help="create a non-destructive workspace overlay")
    q.add_argument("path")
    q.add_argument("--no-copy-configs", action="store_true")
    q.set_defaults(func=workspace_init)
    q = workspace_sub.add_parser("show", help="describe the active workspace")
    q.set_defaults(func=workspace_show)

    p = sub.add_parser("build", help="build immutable data through a workspace recipe")
    p.add_argument("experiment")
    p.add_argument("--config", required=True)
    p.set_defaults(func=build)

    p = sub.add_parser("train", help="launch a stage with captured process output")
    p.add_argument("stage", choices=("cpt", "sft"))
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--dataset-id",
                   help="train this immutable dataset object instead of the mutable "
                        "LATEST pointer (the id a build prints as DATASET_ID)")
    p.add_argument("--follow", action="store_true",
                   help="stream raw stage output between the JSON start/result records")
    p.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=train)

    p = sub.add_parser("list", help="query runs; emits one JSON array")
    p.add_argument("--experiment")
    p.add_argument("--stage")
    p.add_argument("--status")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p.set_defaults(func=lambda a: (emit(RunStore(REPO).list_runs(
        experiment=a.experiment, stage=a.stage, status=a.status,
        limit=a.limit, offset=a.offset)), 0)[1])

    p = sub.add_parser("show", help="query one run and related records")
    p.add_argument("run_id")
    p.add_argument("--events-tail", type=int, default=0)
    p.add_argument("--log-tail", type=int, default=0)
    p.add_argument("--full", action="store_true",
                   help="include full config, provenance, environment, and metadata")
    def show(a):
        store = RunStore(REPO)
        result = (store.get_run(a.run_id, related=True) if a.full
                  else _show_summary(store, a.run_id))
        if a.events_tail:
            result["events_tail"] = store.tail_events(a.run_id, a.events_tail)
        if a.log_tail:
            result["process_log_tail"] = store.tail_process_log(a.run_id, a.log_tail)
        emit(result)
        return 0
    p.set_defaults(func=show)

    p = sub.add_parser("resolve", help="resolve a run or experiment pointer to a checkpoint")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--run-id")
    who.add_argument("--experiment")
    p.add_argument("--pointer", default="latest")
    def resolve(a):
        store = RunStore(REPO)
        if a.run_id:
            run_id, path = a.run_id, store.resolve_checkpoint(a.run_id)
        else:
            run_id, path = store.resolve_pointer(a.experiment, a.pointer)
        emit({"run_id": run_id, "checkpoint": str(path)})
        return 0
    p.set_defaults(func=resolve)

    p = sub.add_parser("decide", help="apply the noise-band keep/revert rule")
    p.add_argument("run_id")
    p.add_argument("--metric", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--noise-band", type=float)
    p.add_argument("--direction", choices=("max", "min"), default="max")
    p.add_argument("--baseline-run-id")
    p.add_argument("--finding", help="reusable finding slug for crystallization evidence")
    p.add_argument("--hypothesis", help="short machine-readable hypothesis")
    p.set_defaults(func=decide)

    p = sub.add_parser("invalidate", help="mark a run invalid without deleting evidence")
    p.add_argument("run_id")
    p.add_argument("--reason", required=True)
    p.add_argument("--metric")
    p.set_defaults(func=invalidate)

    p = sub.add_parser("integrity", help="check SQLite and artifact references")
    p.set_defaults(func=lambda a: (emit(RunStore(REPO).integrity()), 0)[1])

    p = sub.add_parser("reindex", help="rebuild SQLite from run/artifact record files")
    p.add_argument("--reset", action="store_true",
                   help="replace the existing index; record/artifact files are untouched")
    p.set_defaults(func=lambda a: (emit(RunStore(REPO).reindex(reset=a.reset)), 0)[1])

    p = sub.add_parser("gc", help="plan/apply checkpoint retention")
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--apply", action="store_true")
    p.set_defaults(func=lambda a: (emit(RunStore(REPO).gc(
        keep_last=a.keep_last, apply=a.apply)), 0)[1])

    args = parser.parse_args(argv)
    if args.workspace:
        os.environ[WORKSPACE_ENV] = str(Path(args.workspace).expanduser().resolve())
    Workspace.resolve(REPO).apply_environment()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
