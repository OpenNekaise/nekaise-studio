"""Shared test plumbing. Tests are CPU-only, offline, and never import unsloth/torch."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_uniq = itertools.count()


def load_module(path: Path, name: str | None = None):
    """Import a repo file as a fresh module instance (no sys.modules cache)."""
    name = name or f"_test_mod_{next(_uniq)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
