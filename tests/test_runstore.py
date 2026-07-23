from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from lib.runstore import RunStore, new_run_id
from lib.workspace import Workspace


def test_ids_are_sortable_and_collision_resistant():
    ids = {new_run_id(1_700_000_000.123456) for _ in range(1000)}
    assert len(ids) == 1000
    assert all(run_id.startswith("20231114T221320123456Z-") for run_id in ids)


def test_lifecycle_metrics_events_and_query(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run(
        experiment="cpt", stage="cpt", kind="experiment", model="base",
        dataset_id="data-1", seed=7, metadata={"hypothesis": "x"},
    )
    store.log_event(run_id, 1, {"loss": 2.0})
    store.log_event(run_id, 2, {"loss": 1.5})
    store.transition(run_id, "evaluating")
    store.record_metric(
        run_id, name="acc", value=0.4, n=10, split="dev",
        split_hash="split", evaluator="unit", evaluator_version="v1",
    )
    store.transition(run_id, "succeeded")

    run = store.get_run(run_id, related=True)
    assert run["status"] == "succeeded" and run["event_count"] == 2
    assert run["metrics"][0]["value"] == 0.4
    assert store.tail_events(run_id, 1)[0]["step"] == 2
    listed = store.list_runs(experiment="cpt", limit=1)[0]
    assert listed["run_id"] == run_id
    assert "metadata" not in listed and listed["last_metrics"]["loss"] == 1.5
    directory = store.run_dir("cpt", run_id)
    assert json.loads((directory / "manifest.json").read_text())["run_id"] == run_id
    assert json.loads((directory / "state.json").read_text())["status"] == "succeeded"


def test_artifacts_are_immutable_deduplicated_and_pointer_resolved(tmp_path):
    store = RunStore(tmp_path)
    run_ids = [store.create_run(experiment="x", stage="cpt", kind="experiment")
               for _ in range(2)]
    artifacts = []
    for run_id in run_ids:
        temporary = store.artifact_temp("checkpoint", run_id)
        (temporary / "model.safetensors").write_bytes(b"same weights")
        artifact = store.commit_artifact(
            temporary, kind="checkpoint", run_id=run_id, role="checkpoint")
        artifacts.append(artifact)
    assert artifacts[0]["digest"] == artifacts[1]["digest"]
    assert artifacts[0]["path"] == artifacts[1]["path"]
    store.set_pointer("x", "latest", run_ids[1], artifacts[1]["digest"])
    resolved_id, resolved = store.resolve_pointer("x", "latest")
    assert resolved_id == run_ids[1]
    assert (resolved / "model.safetensors").read_bytes() == b"same weights"
    store.reindex(reset=True)
    assert all(store.get_run(run_id, related=True)["artifacts"] for run_id in run_ids)


def test_concurrent_run_creation_has_no_collisions(tmp_path):
    store = RunStore(tmp_path)

    def create(_):
        return store.create_run(experiment="x", stage="cpt", kind="experiment")

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(create, range(100)))
    assert len(set(ids)) == 100
    assert len(store.list_runs(experiment="x", limit=200)) == 100


def test_gc_keeps_pointers_and_latest_n(tmp_path):
    store = RunStore(tmp_path)
    made = []
    for i in range(4):
        run_id = store.create_run(experiment="x", stage="cpt", kind="experiment")
        temporary = store.artifact_temp("checkpoint", run_id)
        (temporary / "weights").write_bytes(f"weights-{i}".encode())
        artifact = store.commit_artifact(
            temporary, kind="checkpoint", run_id=run_id, role="checkpoint")
        made.append((run_id, artifact))
    store.set_pointer("x", "best:acc", made[0][0], made[0][1]["digest"])
    plan = store.gc(keep_last=1, apply=False)
    candidates = {item["digest"] for item in plan["candidates"]}
    assert made[0][1]["digest"] not in candidates
    assert made[-1][1]["digest"] not in candidates
    assert len(candidates) == 2


def test_index_rebuilds_from_record_files(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.create_run(experiment="x", stage="cpt", kind="experiment")
    temporary = store.artifact_temp("checkpoint", run_id)
    (temporary / "weights").write_bytes(b"weights")
    artifact = store.commit_artifact(
        temporary, kind="checkpoint", run_id=run_id, role="checkpoint")
    store.set_pointer("x", "latest", run_id, artifact["digest"])
    store.record_metric(run_id, name="acc", value=0.5, split="dev", evaluator="unit")
    store.record_decision(run_id, metric="acc", value=0.5, verdict="baseline",
                          reason="first")

    result = store.reindex(reset=True)
    rebuilt = store.get_run(run_id, related=True)
    assert result["integrity"]["ok"]
    assert rebuilt["metrics"][0]["value"] == 0.5
    assert rebuilt["decision"]["verdict"] == "baseline"
    assert store.resolve_pointer("x", "latest")[0] == run_id


def test_external_workspace_isolated_and_relocatable(tmp_path):
    repo = tmp_path / "bootloader"
    (repo / "configs").mkdir(parents=True)
    workspace = Workspace.initialize(repo, tmp_path / "user-workspace")
    store = RunStore(repo, workspace=workspace)
    run_id = store.create_run(experiment="x", stage="cpt", kind="experiment")
    temporary = store.artifact_temp("checkpoint", run_id)
    (temporary / "weights").write_bytes(b"external")
    artifact = store.commit_artifact(
        temporary, kind="checkpoint", run_id=run_id, role="checkpoint")

    assert store.root == workspace.root / ".studio"
    assert store.run_dir("x", run_id).is_relative_to(workspace.root)
    assert artifact["path"].startswith("workspace:")
    assert (store.resolve_checkpoint(run_id) / "weights").read_bytes() == b"external"
    assert not (repo / "experiments" / ".studio").exists()
    assert store.reindex(reset=True)["integrity"]["ok"]


def test_agent_controlled_ids_cannot_escape_store(tmp_path):
    store = RunStore(tmp_path)
    with pytest.raises(ValueError):
        store.create_run(experiment="../outside", stage="cpt", kind="experiment")
    with pytest.raises(ValueError):
        store.create_run(
            experiment="safe", stage="cpt", kind="experiment", run_id="../../outside")
