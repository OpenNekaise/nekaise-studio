"""Reproducibility metadata for run records and checkpoint provenance."""
from __future__ import annotations

import hashlib
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprints(repo: Path, relative_paths: list[str]) -> dict[str, str]:
    return {path: sha256_file(repo / path) for path in relative_paths}


def git_state(repo: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        capture_output=True, text=True, check=True).stdout
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def environment() -> dict[str, str]:
    packages = ("unsloth", "trl", "transformers", "torch", "datasets", "accelerate")
    out = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "missing"
    return out
