"""Immutable dataset artifacts with spec-cache pointers and full provenance.

``spec_id`` answers "have I built this recipe before?".  ``dataset_id`` identifies the
actual JSONL bytes.  Changing builder code or output can therefore never silently
overwrite an earlier dataset, even when the declarative spec was unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[1]


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dataset_id(spec: dict) -> str:
    """Stable short ID for a build specification (the cache key, not content ID)."""
    return hashlib.sha256(_canonical(spec).encode()).hexdigest()[:12]


def data_root(exp_dir: Path | str) -> Path:
    return Path(exp_dir) / "data"


def _objects(exp_dir: Path | str) -> Path:
    return data_root(exp_dir) / "objects"


def _spec_pointer(exp_dir: Path | str, spec: dict) -> Path:
    return data_root(exp_dir) / "specs" / f"{dataset_id(spec)}.json"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _managed_store(exp_dir: Path | str):
    """Return the active workspace's store when this is one of its experiments."""
    try:
        from workspace import Workspace
    except ModuleNotFoundError:  # pragma: no cover - package import in unit tests
        from lib.workspace import Workspace
    workspace = Workspace.resolve(REPO)
    directory = Path(exp_dir).resolve()
    try:
        directory.relative_to(workspace.experiments_dir)
    except ValueError:
        return None
    try:
        from runstore import RunStore
    except ModuleNotFoundError:  # pragma: no cover - package import in unit tests
        from lib.runstore import RunStore
    return RunStore(REPO, workspace=workspace)


def artifact_dir(exp_dir: Path | str, spec: dict) -> Path:
    pointer = _spec_pointer(exp_dir, spec)
    if pointer.is_file():
        info = json.loads(pointer.read_text())
        return _objects(exp_dir) / info["dataset_id"]
    # Legacy compatibility for artifacts created by the old spec-addressed store.
    return data_root(exp_dir) / dataset_id(spec)


def exists(exp_dir: Path | str, spec: dict) -> bool:
    return (artifact_dir(exp_dir, spec) / "data.jsonl").exists()


def write(
    exp_dir: Path | str, spec: dict, rows: Iterable[dict],
    stats: dict | None = None, *, recipe_path: str | Path | None = None,
) -> Path:
    """Commit immutable JSONL bytes, then atomically move spec/LATEST pointers."""
    root = data_root(exp_dir)
    temporary = root / f".tmp-{os.getpid()}-{secrets.token_hex(6)}"
    temporary.mkdir(parents=True, exist_ok=False)
    content_hash = hashlib.sha256()
    n = 0
    size_bytes = 0
    data_path = temporary / "data.jsonl"
    try:
        with data_path.open("wb") as handle:
            for row in rows:
                payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                handle.write(payload)
                content_hash.update(payload)
                size_bytes += len(payload)
                n += 1
        content_sha256 = content_hash.hexdigest()
        content_id = content_sha256[:16]
        recipe_sha256 = None
        if recipe_path:
            recipe_bytes = Path(recipe_path).read_bytes()
            recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
        provenance_record = {
            "schema_version": 2,
            "dataset_id": content_id,
            "spec_id": dataset_id(spec),
            "content_sha256": content_sha256,
            "recipe_sha256": recipe_sha256,
            "spec": spec,
            "n": n,
            "size_bytes": size_bytes,
            "stats": stats or {},
            "built": time.time(),
        }
        (temporary / "provenance.json").write_text(
            json.dumps(provenance_record, ensure_ascii=False, indent=2, sort_keys=True))
        destination = _objects(exp_dir) / content_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Same content is a cache hit; the committed object remains immutable.
            import shutil
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        pointer = {"spec_id": dataset_id(spec), "dataset_id": content_id,
                   "content_sha256": content_sha256, "recipe_sha256": recipe_sha256,
                   "provenance": provenance_record}
        _atomic_json(_spec_pointer(exp_dir, spec), pointer)
        _atomic_text(root / "LATEST", content_id)
        _atomic_text(root / "LATEST_SPEC", dataset_id(spec))

        # Register only datasets owned by the active workspace. Unit-test temp dirs stay
        # standalone and external workspaces never contaminate the legacy index.
        store = _managed_store(exp_dir)
        if store is not None:
            store.register_dataset(
                dataset_id=content_id, spec_id=dataset_id(spec),
                content_sha256=content_sha256, recipe_sha256=recipe_sha256,
                path=destination, rows=n, size_bytes=size_bytes, spec=spec,
                metadata={"stats": stats or {}},
            )
        return destination
    except BaseException:
        # A uniquely named temp directory is safe forensic evidence. It is never returned
        # by latest_dir/exists and can be removed by maintenance after diagnosis.
        raise


def read(exp_dir: Path | str, spec: dict) -> list[dict]:
    return read_dir(artifact_dir(exp_dir, spec))


def read_dir(directory: Path | str) -> list[dict]:
    path = Path(directory) / "data.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def data_file(directory: Path | str) -> Path:
    """Return the immutable JSONL payload without reading it into process memory."""
    path = Path(directory) / "data.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def iter_dir(directory: Path | str):
    """Stream rows from an artifact; intended for validation and small transforms."""
    with data_file(directory).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def provenance(directory: Path | str) -> dict:
    return json.loads((Path(directory) / "provenance.json").read_text())


def activate(exp_dir: Path | str, spec: dict) -> Path:
    """Activate a spec alias, preserving provenance when content is deduplicated."""
    pointer_path = _spec_pointer(exp_dir, spec)
    if not pointer_path.is_file():
        raise FileNotFoundError(pointer_path)
    pointer = json.loads(pointer_path.read_text())
    directory = _objects(exp_dir) / pointer["dataset_id"]
    if not (directory / "data.jsonl").is_file():
        raise FileNotFoundError(directory / "data.jsonl")
    if "provenance" not in pointer:
        # Migrate a schema-v2 pointer created before spec aliases carried their own
        # provenance. Content statistics remain valid; identity fields belong to spec.
        record = provenance(directory)
        record.update({
            "spec_id": dataset_id(spec), "spec": spec,
            "recipe_sha256": pointer.get("recipe_sha256"),
        })
        pointer["provenance"] = record
        _atomic_json(pointer_path, pointer)
    root = data_root(exp_dir)
    _atomic_text(root / "LATEST", pointer["dataset_id"])
    _atomic_text(root / "LATEST_SPEC", pointer["spec_id"])
    store = _managed_store(exp_dir)
    if store is not None:
        record = pointer["provenance"]
        store.register_dataset(
            dataset_id=record["dataset_id"], spec_id=record["spec_id"],
            content_sha256=record["content_sha256"],
            recipe_sha256=record.get("recipe_sha256"), path=directory,
            rows=record["n"], size_bytes=record["size_bytes"], spec=record["spec"],
            metadata={"stats": record.get("stats", {})},
        )
    return directory


def latest_provenance(exp_dir: Path | str) -> dict:
    """Provenance for the active spec, not merely the first producer of its bytes."""
    root = data_root(exp_dir)
    active = root / "LATEST_SPEC"
    if active.is_file():
        pointer = root / "specs" / f"{active.read_text().strip()}.json"
        if pointer.is_file():
            value = json.loads(pointer.read_text())
            if value.get("provenance"):
                return value["provenance"]
    directory = latest_dir(exp_dir)
    return provenance(directory) if directory else {}


def latest_dir(exp_dir: Path | str) -> Path | None:
    pointer = data_root(exp_dir) / "LATEST"
    if not pointer.exists():
        return None
    object_id = pointer.read_text().strip()
    modern = _objects(exp_dir) / object_id
    legacy = data_root(exp_dir) / object_id
    for candidate in (modern, legacy):
        if (candidate / "data.jsonl").exists():
            return candidate
    return None


def cached(
    exp_dir: Path | str, spec: dict, builder: Callable[[], Iterable[dict]],
    stats: Callable[[list[dict]], dict] | None = None,
    *, recipe_path: str | Path | None = None,
) -> Path:
    if exists(exp_dir, spec):
        return activate(exp_dir, spec)
    rows = list(builder())
    return write(exp_dir, spec, rows, stats(rows) if stats else None,
                 recipe_path=recipe_path)
