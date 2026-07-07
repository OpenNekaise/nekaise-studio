"""datakit — the dataset cache + provenance layer every recipe writes through."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import datakit  # noqa: E402


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
    assert datakit.latest_dir(tmp_path) == d


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
