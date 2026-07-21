"""R9: crystallize admission is code-enforced — ≥2 independent experiments or no skill."""
from __future__ import annotations

import json

from studio.tools import crystallize_gate as gate


def _log(root, exp, records):
    d = root / "experiments" / exp
    d.mkdir(parents=True)
    (d / "log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))


def test_gate_requires_two_experiments(tmp_path):
    _log(tmp_path, "exp-a", [{"finding": "wsd-wins", "verdict": "keep"}])
    ok, msg = gate.check("wsd-wins", root=tmp_path)
    assert not ok and "need ≥2" in msg

    _log(tmp_path, "exp-b", [{"finding": "wsd-wins", "verdict": "keep"},
                             {"finding": "other", "verdict": "keep"}])
    ok, msg = gate.check("wsd-wins", root=tmp_path)
    assert ok and "PASS" in msg


def test_reverts_do_not_count(tmp_path):
    _log(tmp_path, "exp-a", [{"finding": "x", "verdict": "keep"}])
    _log(tmp_path, "exp-b", [{"finding": "x", "verdict": "revert"}])
    ok, _ = gate.check("x", root=tmp_path)
    assert not ok


def test_mark_unverified(tmp_path):
    md = tmp_path / "skill.md"
    md.write_text("---\nfinding: x\n---\n# Skill\n")
    gate.mark_unverified(md)
    assert "status: unverified" in md.read_text()
    gate.mark_unverified(md)                                   # idempotent
    assert md.read_text().count("status: unverified") == 1
