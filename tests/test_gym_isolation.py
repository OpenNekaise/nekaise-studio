"""R1 done-when: gym/ never imports the studio side. Dependency is one-way."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+(?:studio|lib|packs|tools|experiments|attic)\b", re.M)


def test_gym_never_imports_studio_side():
    offenders = []
    for py in sorted((REPO / "gym").rglob("*.py")):
        for m in FORBIDDEN.finditer(py.read_text()):
            offenders.append(f"{py.relative_to(REPO)}: {m.group(0).strip()}")
    assert not offenders, "gym must not import the studio side (R1):\n" + "\n".join(offenders)


def test_same_verifier_serves_reward_filter_and_eval():
    """The done-when's second half: reward wrapper, pack shim (filter), and eval runner
    all import THE SAME gym verifier module — no duplicated grading logic."""
    reward_src = (REPO / "studio" / "stages" / "_common.py").read_text()
    assert "from gym import tasks as gym_tasks" in reward_src          # TRL reward wrapper
    shim = (REPO / "packs" / "corpus_probes" / "scorer.py").read_text()
    assert "from gym.verifiers import numeric_cloze" in shim           # legacy filter path
    ev = (REPO / "gym" / "runner" / "evaluate.py").read_text()
    assert "from gym.tasks import" in ev                               # eval runner
