#!/usr/bin/env python3
"""building pack — data prep: parse nekaise_data ontologies into a normalized index. FIXED.

Each building is a subfolder of `nekaise_data/` with one or more Turtle (.ttl) ontologies
(Brick / ASHRAE 223P / REC). This parses them into a per-building index of entities —
{uri, name, types, comment, connections, properties} — that `scorer.py` uses to mint
deterministically-verifiable ontology/topology questions. Unparseable files are skipped
with a warning (real building exports aren't always clean Turtle).

    python packs/building/prepare.py        # (re)build the index, print a summary

The index is written under `nekaise_data/_index/` so it stays git-ignored with the raw
(proprietary) data. Set NEKAISE_DATA to point elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
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
