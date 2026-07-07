"""Repo invariants: the skills mirror stays in sync and adapters carry valid frontmatter."""
import re

from conftest import REPO

SKILLS = REPO / "skills"
ADAPTERS = REPO / ".claude" / "skills"


def canonical_names():
    return {p.stem for p in SKILLS.glob("*.md")}


def adapter_names():
    return {p.parent.name for p in ADAPTERS.glob("*/SKILL.md")}


def test_every_canonical_skill_has_a_claude_adapter():
    assert canonical_names() == adapter_names()


def test_adapter_frontmatter_names_match_their_directory():
    for d in sorted(ADAPTERS.glob("*/SKILL.md")):
        head = d.read_text()
        m = re.match(r"---\s*\nname:\s*(\S+)\s*\n(?:.*\n)*?description:", head)
        assert m, f"{d}: missing name/description frontmatter"
        assert m.group(1) == d.parent.name, f"{d}: frontmatter name != directory"


def test_adapters_point_at_existing_canonical_files():
    for d in sorted(ADAPTERS.glob("*/SKILL.md")):
        name = d.parent.name
        assert (SKILLS / f"{name}.md").exists(), f"adapter {name} has no canonical skills/{name}.md"
