"""R4 done-when: no transformers `.generate(` on rollout / eval / datagen paths.

The sanctioned call sites are the vLLM offline engine itself (gym/runner/generate.py)
and the CoAPT student-inference tool (tools/student.py) — both are `self.llm.generate`
on a vLLM engine, not an HF fallback.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SANCTIONED = (REPO / "gym" / "runner" / "generate.py",
              REPO / "tools" / "student.py")


def _hits(root: Path) -> list[str]:
    out = []
    for py in sorted(root.rglob("*.py")):
        if "unsloth_compiled_cache" in py.parts:
            continue
        for i, line in enumerate(py.read_text().splitlines(), 1):
            # prose mentions (docstrings/comments) use backticks or '#'; code doesn't
            if ".generate(" in line and "`" not in line and not line.lstrip().startswith("#"):
                out.append(f"{py.relative_to(REPO)}:{i}: {line.strip()}")
    return out


def test_no_hf_generate_on_hot_paths():
    hits = _hits(REPO / "gym") + _hits(REPO / "studio") + _hits(REPO / "tools")
    prefixes = tuple(str(p.relative_to(REPO)) for p in SANCTIONED)
    rogue = [h for h in hits if not h.startswith(prefixes)]
    assert not rogue, "HF .generate( on a banned path (R4):\n" + "\n".join(rogue)
    sanctioned = [h for h in hits if h.startswith(prefixes)]
    for h in sanctioned:
        assert "self.llm.generate(" in h, f"unexpected generate call: {h}"
