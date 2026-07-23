#!/usr/bin/env python3
"""crystallize_gate — code-enforced admission for crystallized skills (R9).

A finding may become a local skill ONLY when it replicated across ≥2 independent
experiments (distinct experiment names with SQLite decision records tagged by
``metadata.finding`` and verdict=keep). The crystallize-skill runs this gate FIRST and aborts on
exit 1; prune-skills uses --audit to re-review the existing library and demote what
lacks evidence (status: unverified → hypothesis, not deleted).

    python -m studio.tools.crystallize_gate --finding wsd-beats-cosine
    python -m studio.tools.crystallize_gate --audit
    python -m studio.tools.crystallize_gate --mark-unverified skills/local/foo.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MIN_EXPERIMENTS = 2


def evidence(finding: str, root: Path | None = None) -> dict[str, int]:
    """{experiment name: #keep decisions tagged with this finding}."""
    import sys
    repo = (root or REPO).resolve()
    sys.path.insert(0, str(repo / "lib"))
    from runstore import RunStore

    out: dict[str, int] = {}
    for decision in RunStore(repo).list_decisions(verdict="keep"):
        if decision.get("metadata", {}).get("finding") == finding:
            exp = decision["experiment"]
            out[exp] = out.get(exp, 0) + 1
    return out


def check(finding: str, min_experiments: int = MIN_EXPERIMENTS,
          root: Path | None = None) -> tuple[bool, str]:
    ev = evidence(finding, root)
    ok = len(ev) >= min_experiments
    detail = ", ".join(f"{e}×{n}" for e, n in sorted(ev.items())) or "no tagged keeps"
    return ok, (f"finding '{finding}': replicated in {len(ev)} experiment(s) "
                f"[{detail}] — {'PASS' if ok else f'need ≥{min_experiments}'}")


def skill_findings(path: Path) -> list[str]:
    """`finding:` slugs referenced in a local skill's frontmatter/body."""
    return re.findall(r"^finding:\s*([\w-]+)\s*$", path.read_text(), re.M)


def mark_unverified(path: Path) -> None:
    text = path.read_text()
    if re.search(r"^status:\s*unverified\s*$", text, re.M):
        return
    if text.startswith("---\n"):
        text = text.replace("---\n", "---\nstatus: unverified\n", 1)
    else:
        text = f"---\nstatus: unverified\n---\n{text}"
    path.write_text(text)


def audit() -> int:
    sys.path.insert(0, str(REPO / "lib"))
    from workspace import Workspace
    local = sorted(Workspace.resolve(REPO).local_skills_dir.glob("*.md"))
    if not local:
        print(f"{Workspace.resolve(REPO).local_skills_dir} is empty — nothing to audit")
        return 0
    bad = 0
    for p in local:
        slugs = skill_findings(p)
        if not slugs:
            print(f"  {p.name}: NO `finding:` tag — unverifiable → demote to hypothesis")
            bad += 1
            continue
        for s in slugs:
            ok, msg = check(s)
            print(f"  {p.name}: {msg}")
            bad += 0 if ok else 1
    print(f"audit: {len(local)} skill(s), {bad} lacking evidence "
          f"(mark with --mark-unverified; prune-skills demotes them)")
    return 0


def local_skill_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    sys.path.insert(0, str(REPO / "lib"))
    from workspace import Workspace
    workspace = Workspace.resolve(REPO)
    if path.parts[:2] == ("skills", "local"):
        return workspace.local_skills_dir.joinpath(*path.parts[2:])
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--finding", help="slug to check for ≥2-experiment replication")
    g.add_argument("--audit", action="store_true", help="re-review skills/local/")
    g.add_argument("--mark-unverified", metavar="SKILL_MD")
    ap.add_argument("--min-experiments", type=int, default=MIN_EXPERIMENTS)
    args = ap.parse_args(argv)

    if args.audit:
        return audit()
    if args.mark_unverified:
        path = local_skill_path(args.mark_unverified)
        mark_unverified(path)
        print(f"marked unverified: {path}")
        return 0
    ok, msg = check(args.finding, args.min_experiments)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
