"""Immutable content-addressed JSON artifacts. Disk first, database commit second."""
from __future__ import annotations

import hashlib
import json
import os
import re
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


def verify_checkpoint(result: dict, *, require_optimizer: bool = True) -> None:
    root = Path(result["checkpoint"])
    manifest_path = root / "checkpoint.json"
    if result.get("trained") is False and not manifest_path.exists() and result["manifest"].get("kind") == "unchanged_parent":
        manifest = result["manifest"]
    else:
        manifest = json.loads(manifest_path.read_text())
    if manifest != result["manifest"] or not manifest.get("files"):
        raise ValueError("Checkpoint manifest is missing or differs from its stage artifact")
    from .checkpoint_retention import receipt
    retired = receipt(root) if manifest_path.exists() else None
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Checkpoint manifest contains an invalid file path")
        # Inference never opens Adam state. Keep the manifest identity, but avoid
        # reading gigabytes of optimizer bytes on a weight-only diagnostic path.
        # Training/resumption continues to verify every file with the default.
        if not (root/name).exists() and retired and retired.get('deleted', {}).get(name) == expected:
            if name == "training_state.pt" and not require_optimizer:
                continue
            raise ValueError(f"Checkpoint file intentionally retired ({name}); see retained summary and choose an available checkpoint")
        if name == "training_state.pt" and not require_optimizer and (root/name).is_file():
            continue
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


def verified_snapshot(checkpoint: str) -> dict:
    """Read a pinned local HF snapshot, verifying each content-addressed cache blob.

    This records current byte identity, not a retroactive pre-training checksum.
    No downloads, model loading, or writes to the cache occur here.
    """
    root = Path(checkpoint)
    if (not root.is_absolute() or root.resolve() != root or root.parent.name != "snapshots"
            or not root.parent.parent.name.startswith("models--")
            or not re.fullmatch(r"[0-9a-f]{40}", root.name)):
        raise ValueError("Input comparison requires a pinned local cache snapshot")
    blobs = root.parent.parent / "blobs"
    files = {}
    for path in sorted(root.iterdir()):
        target = path.resolve(strict=True)
        if (not path.is_symlink() or target.parent != blobs or not target.is_file()
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", target.name)):
            raise ValueError("Snapshot file must link to its content-addressed cache blob")
        sha256 = hashlib.sha256()
        blob_hash = hashlib.sha1(f"blob {target.stat().st_size}\0".encode()) if len(target.name) == 40 else hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(4*1024*1024), b""):
                sha256.update(block)
                blob_hash.update(block)
        if blob_hash.hexdigest() != target.name:
            raise ValueError(f"Snapshot integrity failed: {path.name}")
        files[path.name] = sha256.hexdigest()
    if not {"config.json", "tokenizer_config.json", "tokenizer.json"} <= files.keys():
        raise ValueError("Snapshot is missing model/tokenizer configuration")
    if "model.safetensors.index.json" in files:
        index = json.loads((root/"model.safetensors.index.json").read_text())
        shards = set(index.get("weight_map", {}).values())
        if not shards or not shards <= files.keys() or any(not s.endswith(".safetensors") for s in shards):
            raise ValueError("Snapshot weight index has missing or invalid shards")
    elif "model.safetensors" not in files:
        raise ValueError("Snapshot is missing safetensors weights")
    return {"kind": "local_hub_snapshot", "revision": root.name, "files": files}


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
