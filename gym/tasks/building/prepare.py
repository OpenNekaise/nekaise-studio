#!/usr/bin/env python3
"""building pack — data prep: parse nekaise_data ontologies into a normalized index. FIXED.

Each building is a subfolder of `nekaise_data/` with one or more Turtle (.ttl) ontologies
(Brick / ASHRAE 223P / REC). This parses them into a per-building index of entities —
{uri, name, types, comment, connections, properties} — that `scorer.py` uses to mint
deterministically-verifiable ontology/topology questions. Unparseable files are skipped
with a warning (real building exports aren't always clean Turtle).

    python gym/tasks/building/prepare.py        # (re)build the index, print a summary

The index is written under `nekaise_data/_index/` so it stays git-ignored with the raw
(proprietary) data. Set NEKAISE_DATA to point elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("NEKAISE_DATA", REPO / "nekaise_data"))
INDEX_DIR = DATA / "_index"


def _local(uri) -> str:
    """Local name of a URI: the bit after the last '#' or '/'."""
    s = str(uri)
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def building_dirs() -> list[Path]:
    """Subfolders of nekaise_data that contain at least one .ttl (one folder = one building)."""
    out = []
    for d in sorted(p for p in DATA.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if d.name == "documentations":
            continue
        if list(d.rglob("*.ttl")):
            out.append(d)
    return out


def default_holdout() -> str | None:
    """Building reserved for eval when NEKAISE_HOLDOUT is unset: the first folder by name.

    Kept positional (not a literal name) so no specific, proprietary building name is
    hardcoded in tracked source. Set NEKAISE_HOLDOUT to choose a different holdout.
    """
    dirs = building_dirs()
    return dirs[0].name if dirs else None


EXAM_PATH = REPO / "packs" / "building" / "eval_open.jsonl"


def exam_building(exam_path: Path | None = None) -> str | None:
    """Which building the frozen exam is about — inferred at runtime, never hardcoded (privacy).

    Matches the known building folder names against each exam row's source / anchors /
    ground_truth. Returns None when there is no exam, no local data, or no clear
    (majority-of-rows) match.
    """
    p = Path(exam_path) if exam_path else EXAM_PATH
    if not p.exists():
        return None
    names = [d.name for d in building_dirs()]
    if not names:
        return None
    rows = [l for l in p.read_text().splitlines() if l.strip()]
    counts: Counter = Counter()
    for line in rows:
        try:
            r = json.loads(line)
        except Exception:
            continue
        hay = " ".join((str(r.get("source", "")), str(r.get("ground_truth", "")),
                        " ".join(str(a) for a in r.get("anchors", []))))
        for n in names:
            if n in hay:
                counts[n] += 1
    if not counts:
        return None
    name, hits = counts.most_common(1)[0]
    return name if hits * 2 >= len(rows) else None


def require_holdout_matches_exam(holdout: str | None, exam_path: Path | None = None) -> str | None:
    """Fail LOUDLY when the configured holdout is not the frozen exam's building.

    The exam grades one specific building. A holdout that silently points elsewhere breaks
    building_judge (context from the wrong building) AND leaks the exam building into
    training. Set NEKAISE_ALLOW_HOLDOUT_MISMATCH=1 to proceed knowingly (e.g. a
    scorer-only experiment with a deliberately different split).
    """
    exam_b = exam_building(exam_path)
    if exam_b is None or holdout == exam_b:
        return holdout
    if os.environ.get("NEKAISE_ALLOW_HOLDOUT_MISMATCH") == "1":
        print(f"WARNING: holdout '{holdout}' != exam building '{exam_b}' "
              f"(NEKAISE_ALLOW_HOLDOUT_MISMATCH=1, proceeding)", file=sys.stderr)
        return holdout
    raise SystemExit(
        f"holdout mismatch: NEKAISE_HOLDOUT resolves to '{holdout}' but the frozen exam "
        f"(packs/building/eval_open.jsonl) is about '{exam_b}'.\n"
        f"This silently collapses building_judge and leaks the exam building into training.\n"
        f"Fix: set NEKAISE_HOLDOUT={exam_b} (e.g. in .env), or set "
        f"NEKAISE_ALLOW_HOLDOUT_MISMATCH=1 if the mismatch is intentional.")


def parse_building(d: Path) -> list[dict]:
    """Merge a building's .ttl files into one graph and extract a flat entity list."""
    import rdflib
    from rdflib import RDF

    g = rdflib.Graph()
    for ttl in sorted(d.rglob("*.ttl")):
        try:
            g.parse(str(ttl), format="turtle")
        except Exception as e:
            print(f"  ! {d.name}: skip {ttl.name} ({str(e).splitlines()[0][:70]})", file=sys.stderr)

    ents: dict[str, dict] = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        e = ents.setdefault(str(s), {"uri": str(s), "name": _local(s), "types": [],
                                     "comment": "", "connections": [], "properties": []})
        t = _local(o)
        if t not in e["types"]:
            e["types"].append(t)

    for s, p, o in g:
        su = str(s)
        if su not in ents:
            continue
        pl = _local(p)
        if pl in ("comment", "label") and not ents[su]["comment"]:
            ents[su]["comment"] = str(o)
        elif pl == "cnx":
            ents[su]["connections"].append(_local(o))
        elif pl == "hasProperty":
            ents[su]["properties"].append(_local(o))
    return list(ents.values())


def build_index() -> dict:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    idx = {}
    for d in building_dirs():
        ents = parse_building(d)
        idx[d.name] = ents
        (INDEX_DIR / f"{d.name}.json").write_text(json.dumps(ents, ensure_ascii=False, indent=2))
    (INDEX_DIR / "buildings.json").write_text(json.dumps(list(idx.keys())))
    return idx


def load_index(rebuild: bool = False) -> dict:
    """Load the cached index, building it on first use."""
    if rebuild or not (INDEX_DIR / "buildings.json").exists():
        return build_index()
    idx = {}
    for b in json.loads((INDEX_DIR / "buildings.json").read_text()):
        idx[b] = json.loads((INDEX_DIR / f"{b}.json").read_text())
    return idx


def main() -> None:
    idx = build_index()
    print(f"indexed {len(idx)} buildings -> {INDEX_DIR}")
    for b, ents in idx.items():
        c = Counter(t for e in ents for t in e["types"])
        top = ", ".join(f"{k}×{v}" for k, v in c.most_common(4))
        n_comment = sum(1 for e in ents if e["comment"])
        n_cnx = sum(1 for e in ents if e["connections"])
        print(f"  {b:18s} {len(ents):4d} entities ({n_comment} with comment, {n_cnx} with cnx); top: {top}")


if __name__ == "__main__":
    main()
