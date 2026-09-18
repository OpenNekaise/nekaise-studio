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


def operational_context(workspace, campaign_id):
    """Supply applied operations alongside, without rewriting, teaching notes.

    Pending proposals and decisions applied to sibling continuations are not a
    handoff to this campaign. All reports remain separately accessible.
    """
    requests = operator_review_requests(workspace, campaign_id)
    with archive(workspace) as db:
        campaign = db.execute("SELECT operator_hold FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        action = db.execute("SELECT id,kind,actor,reason,handled_at FROM actions WHERE campaign_id=? ORDER BY id DESC LIMIT 1", (campaign_id,)).fetchone()
        latest, visited, child_id = None, set(), None
        while campaign_id and campaign_id not in visited:
            visited.add(campaign_id)
            row = db.execute("""SELECT id,campaign_id,status,decision,continuation_id,updated_at
                FROM recoveries WHERE campaign_id=? AND status='resolved' AND decision IS NOT NULL
                AND (continuation_id IS NULL OR continuation_id=?) ORDER BY id DESC LIMIT 1""",
                (campaign_id, child_id)).fetchone()
            if row and (latest is None or row["id"] > latest["id"]):
                latest = dict(row)
                latest["decision"] = json.loads(latest["decision"])
            parent = db.execute("SELECT parent_campaign_id FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            child_id, campaign_id = campaign_id, parent["parent_campaign_id"] if parent else None
    return {"operator_hold": campaign["operator_hold"] if campaign else None,
            "latest_action": dict(action) if action else None, "latest_applied_review": latest,
            "operator_review_requests": requests}


def operator_review_requests(workspace, campaign_id):
    """Retain exact operator investigation requests across the ancestor chain.

    Queue handling or recovery resolution is not evidence that an investigation
    succeeded. Keep original requests available and let reports explain their
    outcomes; this read never clears a hold or queues execution.
    """
    requests, visited = [], set()
    with archive(workspace) as db:
        while campaign_id and campaign_id not in visited:
            visited.add(campaign_id)
            requests.extend(dict(row) for row in db.execute("""SELECT id,campaign_id,kind,actor,reason,created_at,handled_at
                FROM actions WHERE campaign_id=? AND actor='operator' AND kind='review' ORDER BY id""", (campaign_id,)))
            parent = db.execute("SELECT parent_campaign_id FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            campaign_id = parent["parent_campaign_id"] if parent else None
    return {"requests": sorted(requests, key=lambda r: r["id"]),
            "interpretation": "Historical operator review requests, not new commands. Read their reports for findings and remaining work. Handled commands and resolved recoveries do not establish that an underlying investigation is complete. Later explicit operator controls retain precedence."}


def read_report(workspace, request):
    """Reuse the dashboard report reader over a strictly read-only connection."""
    from types import SimpleNamespace
    from .reports import catalog, detail
    with archive(workspace) as db:
        def rows(sql, args=()):
            return [dict(row) for row in db.execute(sql, args)]
        def one(sql, args=()):
            result = rows(sql, args)
            return result[0] if result else None
        service = SimpleNamespace(settings=SimpleNamespace(workspace=Path(workspace)),
                                  store=SimpleNamespace(query=rows, one=one))
        if request["op"] == "report":
            return detail(service, int(request["recovery_id"]))
        return catalog(service, before=request.get("before"), limit=int(request.get("limit", 20)))


def read_experiment(workspace, request):
    from types import SimpleNamespace
    from . import experiments
    with archive(workspace) as db:
        def rows(sql, args=()):
            return [dict(row) for row in db.execute(sql, args)]
        def one(sql, args=()):
            result = rows(sql, args)
            return result[0] if result else None
        store, artifacts = SimpleNamespace(query=rows, one=one), Artifacts(Path(workspace))
        if request["op"] == "experiment":
            return experiments.detail(store, artifacts, request["round_id"])
        if request["op"] == "strategy":
            return experiments.strategy(store, artifacts, request["version"])
        return experiments.catalog(store, campaign_id=request.get("campaign_id"),
            strategy_version=request.get("strategy_version"), before=int(request["before"]) if request.get("before") is not None else None,
            limit=int(request.get("limit", 20)))


def replay_lesson(workspace, round_id, lesson_id):
    with archive(workspace) as db:
        row = db.execute("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='gate' AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id,)).fetchone()
    if not row:
        raise ValueError(f"No completed teacher lesson artifact in {round_id}")
    lessons = Artifacts(Path(workspace)).get(row["artifact"])["lessons"]
    for lesson in lessons:
        if lesson["id"] == lesson_id:
            result = {**lesson, "origin_round_id": round_id, "origin_artifact": row["artifact"]}
            if lesson.get("training_tokenization") == "chat_response":
                # The diagnostic prompt may have used raw text. Bind replay to
                # the template actually used for training, not that prompt's mode.
                with archive(workspace) as db:
                    frozen = db.execute("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='freeze' AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id,)).fetchone()
                rows = Artifacts(Path(workspace)).get(frozen["artifact"])["rows"] if frozen else []
                trained = next((r for r in rows if r["id"] == lesson_id and r["stream"] in {"cpt", "sft"}), None)
                if trained is None:
                    raise ValueError("Chat replay requires a frozen training row; re-author an unprepared lesson explicitly")
                result["training_template_sha256"] = trained["serialization"]["template_sha256"]
                result["origin_freeze_artifact"] = frozen["artifact"]
            return result
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
            "log_summaries": "Orchestrator summaries and recovery locations of cleaned raw logs; offset/limit",
            "reports": "All operational reports including proposals and outcomes; limit/before; follow next_before",
            "report": "Full operational report, decision turns, host checks and outcomes: recovery_id",
            "experiments": "All recorded teaching experiments, including interrupted/archived runs; optional campaign_id, strategy_version, limit/before; follow next_before",
            "experiment": "Pre-training plan, immutable strategy, later Teacher judgment and actual teaching evidence: round_id; null if no plan was recorded",
            "strategy": "Read an immutable teaching strategy definition: version (full content hash)",
            "artifact": "Read immutable JSON: hash",
            "material_candidates": "Read all exact material-author candidates: round_id, offset/limit, optional candidate_id; full sources and provenance included",
            "author_jobs": "All material-author jobs including partial failures; optional round_id, author_id; offset/limit; saved input/result artifact hashes",
            "author_calls": "Material-author attempts, reservations, reported usage and response artifact hashes; optional round_id; offset/limit",
            "lesson": "Read a historical teacher lesson: round_id, lesson_id",
            "sources": "Search the entire corpus: query, optional prefix, offset/limit (prefix hint is not a restriction)",
            "source": "Read verified source: document_id, start (default 0), length (0=all)"
        }, "current_campaign_id": context["campaign_id"], "workspace": str(workspace), "corpus_path": context["corpus_path"], "latest_strategy": latest_strategy(workspace, context["campaign_id"]), "operational_context": operational_context(workspace, context["campaign_id"])}
    if op in {"reports", "report"}:
        return read_report(workspace, request)
    if op in {"experiments", "experiment", "strategy"}:
        return read_experiment(workspace, request)
    if op == "artifact":
        return Artifacts(workspace).get(request["hash"])
    if op == "material_candidates":
        with archive(workspace) as db:
            stage = db.execute("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='expand' AND status='complete' ORDER BY id DESC LIMIT 1", (request["round_id"],)).fetchone()
        if not stage:
            raise ValueError("No completed material expansion for this round")
        result = Artifacts(workspace).get(stage["artifact"])
        rows = result["candidates"]
        if request.get("candidate_id"):
            rows = [r for r in rows if r["id"] == request["candidate_id"]]
        return {"manifest_hash": result["manifest_hash"], "rows": rows[offset:offset+limit],
                "next_offset": offset+limit if offset+limit < len(rows) else None, "total": len(rows)}
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
        "calls": ("teacher_calls x", "x.*", "x.id DESC", {"campaign_id":"x.campaign_id"}),
        "author_jobs": ("material_jobs x", "x.*", "x.created_at DESC,x.id", {"round_id":"x.round_id", "author_id":"x.author_id"}),
        "author_calls": ("material_calls x JOIN material_jobs j ON x.job_id=j.id", "x.*,j.round_id,j.author_id", "x.id DESC", {"round_id":"j.round_id"}),
        "log_summaries": ("log_cleanup x", "x.*", "x.id DESC", {})
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
        if op in {"author_jobs", "author_calls"} and not db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='material_jobs'").fetchone():
            return {"rows": [], "next_offset": None}
        rows = [dict(r) for r in db.execute(f"SELECT {columns} FROM {table} WHERE {' AND '.join(where) or '1'} ORDER BY {order} LIMIT ? OFFSET ?", (*args, limit+1, offset))]
    more = len(rows)>limit
    for row in rows[:limit]:
        if op == 'rounds' and row.get('checkpoint'):
            from .checkpoint_retention import receipt
            row['checkpoint_retention'] = receipt(Path(row['checkpoint']))
        for key in ("config", "data", "usage"):
            if key in row:
                row[key] = json.loads(row[key])
    return {"rows": rows[:limit], "next_offset": offset+limit if more else None}


if __name__ == "__main__":
    context = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(query(context, json.loads(sys.argv[2])), ensure_ascii=False))
