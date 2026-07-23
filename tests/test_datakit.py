"""datakit — the dataset cache + provenance layer every recipe writes through."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import datakit  # noqa: E402
from lib.runstore import RunStore
from lib.workspace import ENV, Workspace


def test_dataset_id_is_stable_and_order_independent():
    a = datakit.dataset_id({"x": 1, "y": [1, 2]})
    b = datakit.dataset_id({"y": [1, 2], "x": 1})
    assert a == b and len(a) == 12
    assert datakit.dataset_id({"x": 2, "y": [1, 2]}) != a


def test_write_read_roundtrip_with_provenance(tmp_path):
    spec = {"kind": "sft", "seed": 1}
    rows = [{"messages": [{"role": "user", "content": "q"}]}] * 3
    d = datakit.write(tmp_path, spec, rows, stats={"kept": 3})
    assert datakit.read(tmp_path, spec) == rows
    prov = datakit.provenance(d)
    assert prov["spec"] == spec and prov["n"] == 3 and prov["stats"] == {"kept": 3}
    assert prov["dataset_id"] == prov["content_sha256"][:16]
    assert d.parent.name == "objects"
    assert datakit.latest_dir(tmp_path) == d
    assert datakit.latest_provenance(tmp_path) == prov
    assert datakit.data_file(d) == d / "data.jsonl"
    assert list(datakit.iter_dir(d)) == rows


def test_same_spec_new_content_never_overwrites_old_object(tmp_path):
    spec = {"kind": "sft", "seed": 1}
    first = datakit.write(tmp_path, spec, [{"x": 1}])
    second = datakit.write(tmp_path, spec, [{"x": 2}])
    assert first != second
    assert datakit.read_dir(first) == [{"x": 1}]
    assert datakit.read_dir(second) == [{"x": 2}]
    assert datakit.latest_dir(tmp_path) == second


def test_same_content_can_have_multiple_spec_provenance_aliases(tmp_path):
    rows = [{"text": "same bytes"}]
    first = datakit.write(tmp_path, {"scale": "4m"}, rows, stats={"tokens": 4})
    second = datakit.write(tmp_path, {"scale": "40m"}, rows, stats={"tokens": 4})
    assert first == second
    assert datakit.provenance(first)["spec"] == {"scale": "4m"}
    assert datakit.latest_provenance(tmp_path)["spec"] == {"scale": "40m"}
    datakit.activate(tmp_path, {"scale": "4m"})
    assert datakit.latest_provenance(tmp_path)["spec"] == {"scale": "4m"}


def test_cached_builds_once(tmp_path):
    spec = {"kind": "sft", "v": 1}
    calls = []

    def builder():
        calls.append(1)
        return [{"messages": []}]

    d1 = datakit.cached(tmp_path, spec, builder)
    d2 = datakit.cached(tmp_path, spec, builder)
    assert d1 == d2 and len(calls) == 1          # second call hits the cache
    datakit.cached(tmp_path, {"kind": "sft", "v": 2}, builder)
    assert len(calls) == 2                       # changed spec = new artifact


def test_external_workspace_dataset_registers_in_its_own_index(tmp_path, monkeypatch):
    workspace = Workspace.initialize(datakit.REPO, tmp_path / "user-workspace",
                                     copy_configs=False)
    monkeypatch.setenv(ENV, str(workspace.root))
    exp_dir = workspace.experiment_dir("local-exp")
    artifact = datakit.write(exp_dir, {"kind": "local"}, [{"x": 1}])
    store = RunStore(datakit.REPO)
    assert store.root == workspace.store_dir
    assert store.integrity()["counts"]["datasets"] == 1
    assert artifact.is_relative_to(workspace.root)
