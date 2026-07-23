from __future__ import annotations

import json

import pytest

from lib.workspace import ENV, MANIFEST, Workspace, validate_name


def _repo(tmp_path):
    repo = tmp_path / "bootloader"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "cpt.yaml").write_text("frozen: {}\ndata: {}\nrun: {}\n")
    (repo / "experiments" / "cpt").mkdir(parents=True)
    (repo / "experiments" / "cpt" / "build_data.py").write_text("# core recipe\n")
    return repo


def test_legacy_layout_is_backward_compatible(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    repo = _repo(tmp_path)
    workspace = Workspace.resolve(repo)
    assert not workspace.external
    assert workspace.experiments_dir == repo / "experiments"
    assert workspace.store_dir == repo / "experiments" / ".studio"
    assert workspace.local_skills_dir == repo / "skills" / "local"


def test_initialize_is_non_destructive_and_resolves_overlays(tmp_path):
    repo = _repo(tmp_path)
    root = tmp_path / "user-workspace"
    workspace = Workspace.initialize(repo, root)
    assert json.loads((root / MANIFEST).read_text())["workspace_id"] == workspace.workspace_id
    assert (root / "configs" / "cpt.yaml").is_file()
    assert (root / "nekaise_data").is_dir()
    assert (root / "skills" / "local").is_dir()
    assert workspace.resolve_config("configs/cpt.yaml") == root / "configs" / "cpt.yaml"
    assert workspace.resolve_experiment_file("cpt", "build_data.py") == (
        repo / "experiments" / "cpt" / "build_data.py")

    (root / "configs" / "cpt.yaml").write_text("user-owned\n")
    Workspace.initialize(repo, root)
    assert (root / "configs" / "cpt.yaml").read_text() == "user-owned\n"


def test_invalid_experiment_names_cannot_escape_workspace():
    for value in ("../core", "/absolute", "", "a/b"):
        with pytest.raises(ValueError):
            validate_name(value, "experiment")


def test_unignored_workspace_inside_bootloader_is_refused(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="git-ignored"):
        Workspace.initialize(repo, repo / "accidental-local-state")
