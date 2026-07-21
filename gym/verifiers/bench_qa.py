"""bench_qa — grade against the EXTERNAL nekaise-bench harness (milestone referee).

meta: {"gold": {"track": "mcq"|"open", "answer": ..., "choices"?: [...], "aliases"?: [...]}}
      (or the same dict JSON-encoded as a string — legacy pack rows)

Grading is IMPORTED from the bench's own eval module — never reimplemented — so the
external benchmark stays the single grading truth for itself. Needs a local clone
(NEKAISE_BENCH_DIR, default ../nekaise-bench). This is an external-repo import, not a
studio import; the one-way dependency rule (R1) is intact.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parents[2].parent   # …/Code (repo's parent dir)
_harness_mod = None


def bench_dir() -> Path:
    return Path(os.environ.get("NEKAISE_BENCH_DIR", _REPO_PARENT / "nekaise-bench"))


def harness():
    """The bench repo's own eval module (prompts + grading)."""
    global _harness_mod
    if _harness_mod is None:
        runner = bench_dir() / "eval_ollama.py"
        if not runner.exists():
            raise FileNotFoundError(
                f"nekaise-bench not found at {bench_dir()} — clone "
                f"https://github.com/OpenNekaise/nekaise-bench there, or set NEKAISE_BENCH_DIR")
        spec = importlib.util.spec_from_file_location("nekaise_bench_eval", runner)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _harness_mod = mod
    return _harness_mod


def verify(prompt: str, response: str, meta: dict) -> float:
    g = meta["gold"]
    if isinstance(g, str):
        g = json.loads(g)
    h = harness()
    ans = h.strip_think(str(response))
    if g["track"] == "mcq":
        pred = h.extract_letter(ans, len(g["choices"]), g["choices"])
        return 1.0 if pred == g["answer"] else 0.0
    return 1.0 if h.grade_open(ans, g["answer"], g.get("aliases")) else 0.0
