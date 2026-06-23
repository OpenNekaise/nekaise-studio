"""datakit — content-addressed dataset cache + provenance. FIXED plumbing.

The autoresearch loop needs a place for *generated* training data (teacher CoT,
rejection-sampled solutions, preference pairs) that is:
  - expensive to build (frontier API / GPU sampling) → built once, then reused,
  - reproducible → every artifact records exactly the spec that produced it,
  - swappable → a changed recipe makes a new artifact, not a silent overwrite.

A dataset artifact is a folder `experiments/<exp>/data/<dataset_id>/` with:
  - data.jsonl        : the built rows (schema set by `kind`: "sft" | "preference")
  - provenance.json   : {dataset_id, spec, n, stats, built}

`dataset_id` = short stable hash of the build SPEC, so the same recipe hits the
cache and a changed recipe gets a fresh id. build_data.py is the (mutable) recipe;
this module is the (fixed) plumbing it calls.

Row schemas:
  kind="sft"        -> {"messages": [{"role","content"}, ...]}
  kind="preference" -> {"prompt": str, "chosen": str, "rejected": str}
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable


def dataset_id(spec: dict) -> str:
    """Short stable id for a build spec (order-independent)."""
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def data_root(exp_dir: Path | str) -> Path:
    return Path(exp_dir) / "data"


def artifact_dir(exp_dir: Path | str, spec: dict) -> Path:
    return data_root(exp_dir) / dataset_id(spec)


def exists(exp_dir: Path | str, spec: dict) -> bool:
    return (artifact_dir(exp_dir, spec) / "data.jsonl").exists()


def write(exp_dir: Path | str, spec: dict, rows: Iterable[dict],
          stats: dict | None = None) -> Path:
    """Write rows + provenance for `spec`; update the LATEST pointer. Returns the dir."""
    d = artifact_dir(exp_dir, spec)
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with (d / "data.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    prov = {
        "dataset_id": dataset_id(spec),
        "spec": spec,
        "n": n,
        "stats": stats or {},
        "built": time.time(),
    }
    (d / "provenance.json").write_text(json.dumps(prov, indent=2))
    (data_root(exp_dir) / "LATEST").write_text(dataset_id(spec))
    return d


def read(exp_dir: Path | str, spec: dict) -> list[dict]:
    d = artifact_dir(exp_dir, spec)
    return read_dir(d)


def read_dir(d: Path | str) -> list[dict]:
    p = Path(d) / "data.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def provenance(d: Path | str) -> dict:
    return json.loads((Path(d) / "provenance.json").read_text())


def latest_dir(exp_dir: Path | str) -> Path | None:
    """The most recently built artifact dir, or None."""
    p = data_root(exp_dir) / "LATEST"
    if not p.exists():
        return None
    d = data_root(exp_dir) / p.read_text().strip()
    return d if (d / "data.jsonl").exists() else None


def cached(exp_dir: Path | str, spec: dict,
           builder: Callable[[], Iterable[dict]],
           stats: Callable[[list[dict]], dict] | None = None) -> Path:
    """Return the artifact dir for `spec`, building (and caching) it if absent."""
    if exists(exp_dir, spec):
        return artifact_dir(exp_dir, spec)
    rows = list(builder())
    return write(exp_dir, spec, rows, stats(rows) if stats else None)
