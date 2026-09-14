"""Read-only, paginated access to the complete teaching archive and corpus.

Usage: python -m nekaise_loop.teacher_tools CONTEXT_JSON QUERY_JSON
Pagination bounds one response, never the set of accessible history or sources.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys

from .artifacts import Artifacts
from .config import ROOT
from .corpus import search_sources, read_source


@contextmanager
def archive(workspace):
    db = sqlite3.connect(f"file:{Path(workspace)/'loop.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        yield db
    finally:
        db.close()


def latest_strategy(workspace, campaign_id):
    with archive(workspace) as db:
        visited = set()
        while campaign_id and campaign_id not in visited:
            visited.add(campaign_id)
            stage = db.execute("SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='adapt' AND s.status='complete' ORDER BY r.number DESC,s.id DESC LIMIT 1", (campaign_id,)).fetchone()
            if stage:
                return Artifacts(Path(workspace)).get(stage["artifact"])
            row = db.execute("SELECT parent_campaign_id,context_artifact FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            campaign_id = row["parent_campaign_id"] if row else None
        return {}


def replay_lesson(workspace, round_id, lesson_id):
    with archive(workspace) as db:
        row = db.execute("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='gate' AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id,)).fetchone()
    if not row:
        raise ValueError(f"No completed teacher lesson artifact in {round_id}")
    lessons = Artifacts(Path(workspace)).get(row["artifact"])["lessons"]
    for lesson in lessons:
        if lesson["id"] == lesson_id:
            return {**lesson, "origin_round_id": round_id, "origin_artifact": row["artifact"]}
    raise ValueError(f"Unknown lesson {round_id}/{lesson_id}")


def query(context, request):
    op = request.get("op", "help")
    offset, limit = int(request.get("offset", 0)), int(request.get("limit", 20))
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError("Use offset>=0 and page limit 1..200; continue paging for all results")
    workspace = Path(context["workspace"])
    if op == "help":
        return {"operations": {
            "campaigns": "All campaigns, config and lineage; offset/limit",
            "rounds": "All rounds; optional campaign_id; offset/limit",
            "records": "Lessons/evaluations/gaps; optional round_id, campaign_id, kind, query; offset/limit",
            "stages": "Stage attempts and artifact hashes; optional round_id; offset/limit",
            "metrics": "Actual optimizer metrics; optional round_id; offset/limit",
            "events": "Activity and failures; optional campaign_id; offset/limit",
            "calls": "Teacher call metadata; optional campaign_id; offset/limit; full prompts/results in runs/",
            "artifact": "Read immutable JSON: hash",
            "lesson": "Read a historical teacher lesson: round_id, lesson_id",
            "sources": "Search the entire corpus: query, optional prefix, offset/limit (prefix hint is not a restriction)",
            "source": "Read verified source: document_id, start (default 0), length (0=all)"
        }, "current_campaign_id": context["campaign_id"], "workspace": str(workspace), "corpus_path": context["corpus_path"], "latest_strategy": latest_strategy(workspace, context["campaign_id"])}
    if op == "artifact":
        return Artifacts(workspace).get(request["hash"])
    if op == "lesson":
        return replay_lesson(workspace, request["round_id"], request["lesson_id"])
    if op == "sources":
        return search_sources((ROOT/context["corpus_path"]).resolve(), request.get("query", ""), request.get("prefix", ""), offset, limit)
    if op == "source":
        return read_source((ROOT/context["corpus_path"]).resolve(), request["document_id"], int(request.get("start", 0)), int(request.get("length", 0)))
    specs = {
        "campaigns": ("campaigns x", "x.*", "x.created_at DESC,x.id", {"campaign_id": "x.id"}),
        "rounds": ("rounds x", "x.*", "x.created_at DESC,x.id", {"campaign_id": "x.campaign_id", "round_id": "x.id"}),
        "records": ("records x JOIN rounds r ON r.id=x.round_id", "x.*,r.campaign_id,r.number AS round_number", "r.created_at DESC,x.position,x.id", {"round_id":"x.round_id", "campaign_id":"r.campaign_id", "kind":"x.kind"}),
        "stages": ("stage_runs x", "x.*", "x.id DESC", {"round_id":"x.round_id"}),
        "metrics": ("metrics x", "x.*", "x.id DESC", {"round_id":"x.round_id"}),
        "events": ("events x", "x.*", "x.id DESC", {"campaign_id":"x.campaign_id"}),
        "calls": ("teacher_calls x", "x.*", "x.id DESC", {"campaign_id":"x.campaign_id"})
    }
    if op not in specs:
        raise ValueError(f"Unknown archive operation: {op}")
    table, columns, order, filters = specs[op]
    where, args = [], []
    for key, column in filters.items():
        if key in request:
            where.append(f"{column}=?")
            args.append(request[key])
    if op == "records" and request.get("query"):
        where.append("x.data LIKE ?")
        args.append("%"+request["query"]+"%")
    with archive(workspace) as db:
        rows = [dict(r) for r in db.execute(f"SELECT {columns} FROM {table} WHERE {' AND '.join(where) or '1'} ORDER BY {order} LIMIT ? OFFSET ?", (*args, limit+1, offset))]
    more = len(rows)>limit
    for row in rows[:limit]:
        for key in ("config", "data", "usage"):
            if key in row:
                row[key] = json.loads(row[key])
    return {"rows": rows[:limit], "next_offset": offset+limit if more else None}


if __name__ == "__main__":
    context = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(query(context, json.loads(sys.argv[2])), ensure_ascii=False))
