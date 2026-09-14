"""Immutable content-addressed JSON artifacts. Disk first, database commit second."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_checkpoint(result: dict) -> None:
    root = Path(result["checkpoint"])
    manifest_path = root / "checkpoint.json"
    if result.get("trained") is False and not manifest_path.exists() and result["manifest"].get("kind") == "unchanged_parent":
        manifest = result["manifest"]
    else:
        manifest = json.loads(manifest_path.read_text())
    if manifest != result["manifest"] or not manifest.get("files"):
        raise ValueError("Checkpoint manifest is missing or differs from its stage artifact")
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Checkpoint manifest contains an invalid file path")
        h = hashlib.sha256()
        with (root/name).open("rb") as handle:
            for block in iter(lambda: handle.read(4*1024*1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"Checkpoint integrity failed: {name}")


def unchanged_checkpoint(checkpoint, dataset_hash):
    root = Path(checkpoint)
    manifest_path = root/"checkpoint.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        files = {}
        for path in sorted(root.iterdir()):
            if path.is_file():
                h = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(4*1024*1024), b""):
                        h.update(block)
                files[path.name] = h.hexdigest()
        manifest = {"kind": "unchanged_parent", "files": files}
    return {"checkpoint": checkpoint, "manifest": manifest, "trained": False, "dataset_hash": dataset_hash, "reason": "Teacher selected no training targets; parent checkpoint retained"}


class Artifacts:
    def __init__(self, workspace: Path):
        self.root = workspace / "artifacts"

    def put(self, value: object) -> str:
        key = digest(value)
        path = self.root / key[:2] / f"{key}.json"
        if not path.exists():
            atomic_write(path, canonical(value))
        return key

    def get(self, key: str):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact digest")
        data = (self.root / key[:2] / f"{key}.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != key:
            raise ValueError(f"Artifact integrity failed: {key}")
        return json.loads(data)
