"""pack — load a task pack's referee module. FIXED plumbing (do not edit per-experiment).

A task pack lives in `packs/<name>/` and its `scorer.py` exposes the contract:

    load_split(split, n) -> list[dict]   # rows with at least {"question", "answer"}
    is_correct(pred, gold) -> bool       # eval metric / rejection-sampling filter
    reward(pred, gold) -> float          # graded signal in [0,1] for RL
    extract_answer(text) -> str | None   # shared helper

Usage (from build_data.py / train.py):

    from pack import load
    pack = load("gsm8k")
    rows = pack.load_split("train", 500)
    ok   = pack.is_correct(pred, rows[0]["answer"])

Loading by name (not by path) keeps experiments decoupled from where packs live,
so swapping the pack changes one string, not the training code.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]

# The fixed contract every pack scorer must provide.
_REQUIRED = ("load_split", "is_correct", "reward", "extract_answer")


def load(name: str) -> ModuleType:
    """Import and return packs/<name>/scorer.py as a module, verifying the contract."""
    path = REPO / "packs" / name / "scorer.py"
    if not path.exists():
        raise FileNotFoundError(f"pack '{name}' has no scorer at {path}")
    spec = importlib.util.spec_from_file_location(f"pack_{name}_scorer", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [fn for fn in _REQUIRED if not hasattr(mod, fn)]
    if missing:
        raise AttributeError(f"pack '{name}' scorer is missing {missing}")
    return mod
