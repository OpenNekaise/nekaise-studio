"""Explicit dataset binding: `train --dataset-id` pins an immutable object even after a
later build moves the mutable LATEST pointer."""
from __future__ import annotations

import sys

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "lib"))

import datakit  # noqa: E402
from studio.stages.cpt import load_data_paths, resolve_dataset_dir  # noqa: E402

CFG = {"data": {"sources": [{"dataset": "auto", "weight": 1}]}}


def test_dataset_id_pins_object_across_latest_moves(tmp_path):
    exp = tmp_path / "exp"
    first = datakit.write(exp, {"kind": "t", "v": 1},
                          iter([{"text": "hello world"}]), stats={"content_tokens": 5})
    dataset_id = first.name

    paths, tokens, provenances = load_data_paths(CFG, exp, dataset_id=dataset_id)
    assert paths[0].is_file() and tokens == 5
    assert provenances[0]["dataset_id"] == dataset_id

    # A second build moves LATEST; the explicit id still resolves the first object.
    datakit.write(exp, {"kind": "t", "v": 2},
                  iter([{"text": "x y"}]), stats={"content_tokens": 2})
    _, latest_tokens, _ = load_data_paths(CFG, exp)
    assert latest_tokens == 2
    pinned_paths, pinned_tokens, _ = load_data_paths(CFG, exp, dataset_id=dataset_id)
    assert pinned_tokens == 5 and pinned_paths[0] == paths[0]


def test_unknown_dataset_id_refuses(tmp_path):
    exp = tmp_path / "exp"
    datakit.write(exp, {"kind": "t"}, iter([{"text": "a b"}]),
                  stats={"content_tokens": 2})
    with pytest.raises(SystemExit):
        resolve_dataset_dir(exp, "deadbeef00000000")
