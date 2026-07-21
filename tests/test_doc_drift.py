"""Docs may not advertise ban-listed techniques as available (SPEC §2).

Born from a real drift: README said "DPO — preference optimization (available)." while
SPEC.md bans DPO stacking — one repo simultaneously said available and banned. Guard: in
tracked markdown, a ban-list term must not co-occur with availability language on a line,
unless that line itself marks the legacy/banned status.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BAN_TERMS = re.compile(
    r"\b(?:DPO|PPO|curriculum learning|process reward|MCTS|model merging)\b", re.I)
AVAILABLE = re.compile(r"\b(?:available|supported|enabled|in service)\b|可用|开箱", re.I)
LEGACY_MARK = re.compile(r"legacy|banned|ban[- ]list|archived|禁用|存档", re.I)
SKIP = {"SPEC.md", "REFACTOR.md", "BOUNDARY.md",
        "REFACTOR-NOTES.md"}                              # they DISCUSS the ban itself


def tracked_markdown() -> list[Path]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.md"],
                         capture_output=True, text=True, check=True).stdout
    return [REPO / f for f in out.splitlines() if Path(f).name not in SKIP]


def test_banned_techniques_never_advertised():
    offenders = []
    for md in tracked_markdown():
        for i, line in enumerate(md.read_text().splitlines(), 1):
            if BAN_TERMS.search(line) and AVAILABLE.search(line) \
                    and not LEGACY_MARK.search(line):
                offenders.append(f"{md.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, \
        "ban-listed technique advertised as available (SPEC §2):\n" + "\n".join(offenders)
