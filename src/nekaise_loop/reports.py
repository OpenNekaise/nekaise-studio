"""Read-only operational reports shared by the dashboard and orchestrator.

Decisions are proposals until the durable controller state records their outcome.
Live observations are kept separate from dated agent narratives.
"""
from __future__ import annotations

import json

from .artifacts import Artifacts
from .learning_work import latest_work
from .storage import now


def catalog(service, *, before=None, limit=30):
    where, args = ("WHERE r.id < ?", [before]) if before is not None else ("", [])
    rows = service.store.query(f"""SELECT r.id,r.campaign_id,r.kind,r.status,r.error,r.retry_at,
        r.created_at,r.updated_at,r.continuation_id,r.decision,
        COALESCE(NULLIF(t.label,''),c.name) AS campaign_name
        FROM recoveries r JOIN campaigns c ON c.id=r.campaign_id
        LEFT JOIN run_retention t ON t.campaign_id=c.id {where}
        ORDER BY r.id DESC LIMIT ?""", (*args, limit + 1))
    more = len(rows) > limit
    rows = rows[:limit]
    for row in rows:
        decision = json.loads(row.pop("decision") or "{}")
        row["action"] = decision.get("action")
        row["reason"] = decision.get("reason") or row["error"]
    return {"items": rows, "next_before": rows[-1]["id"] if more else None}


def current_status(service, *, campaign_id=None):
    store = service.store
    where, args = ("WHERE c.id=?", (campaign_id,)) if campaign_id is not None else ("", ())
    campaign = store.one(f"""SELECT c.id,c.status,c.error,c.updated_at,c.operator_hold,
        COALESCE(NULLIF(t.label,''),c.name) AS name FROM campaigns c
        LEFT JOIN run_retention t ON t.campaign_id=c.id {where}
        ORDER BY (c.status IN ('running','queued','recovering','waiting','pausing','stopping')) DESC,
        c.created_at DESC LIMIT 1""", args)
    if not campaign:
        return {"observed_at": now(), "campaign": None, "summary": "No runs recorded.", "next_action": "No work is scheduled."}
    cid = campaign["id"]
    round_ = store.one("SELECT id,number,status,stage,updated_at FROM rounds WHERE campaign_id=? ORDER BY number DESC LIMIT 1", (cid,))
    completed = store.one("SELECT COUNT(*) AS n FROM rounds WHERE campaign_id=? AND status='complete'", (cid,))["n"]
    recovery = store.one("SELECT id,status,kind,error,retry_at,updated_at FROM recoveries WHERE campaign_id=? AND status IN ('pending','running','waiting','decided') ORDER BY id DESC LIMIT 1", (cid,))
    last_action = store.one("SELECT id,kind,actor,reason,created_at FROM actions WHERE campaign_id=? ORDER BY id DESC LIMIT 1", (cid,))
    status = campaign["status"]
    summary = f"{campaign['name']} is {status}. {completed} {'iteration' if completed == 1 else 'iterations'} completed in this run."
    if round_ and status == "running":
        summary += f" Iteration {round_['number']} is at {round_['stage'] or 'preparation'}."
    config = store.campaign(cid)["config"]
    if campaign["operator_hold"]:
        next_action = f"Your explicit {campaign['operator_hold']} remains in effect; automatic execution is suspended."
    elif recovery:
        if not config.get("auto_recover", True):
            next_action = "Automatic orchestration is disabled in this run's configuration."
        elif recovery["status"] == "decided":
            next_action = "The controller will apply the orchestrator's recorded decision."
        elif recovery["retry_at"]:
            next_action = "The scheduled automatic retry or review will run when the wait expires."
        elif recovery["kind"] == "budget":
            next_action = "The configured execution allowance is exhausted; the limit remains in force."
        else:
            next_action = "The orchestrator will inspect the reports and decide the next action."
    elif status in {"running", "queued"}:
        next_action = "Training continues under the teacher's curriculum; interruptions wake the orchestrator."
    elif status in {"paused", "stopped", "pausing", "stopping"}:
        next_action = ("Execution is suspended; the supervisor will wake the orchestrator to review the interruption."
                       if config.get("auto_recover", True) else "Automatic orchestration is disabled in this run's configuration.")
    else:
        next_action = "No further work is queued for this run."
    review = store.one("SELECT next_review_round FROM history_reviews ORDER BY applied_at DESC LIMIT 1")
    total = store.one("SELECT COUNT(*) AS n FROM rounds WHERE status='complete'")["n"]
    return {"observed_at": now(), "campaign": campaign, "round": round_,
            "completed_rounds": completed, "summary": summary, "next_action": next_action,
            "learning_work": latest_work(store, Artifacts(service.settings.workspace), cid),
            "recovery": recovery, "last_action": last_action,
            "review_in_rounds": max(0, review["next_review_round"] - total) if review and config.get("manage_history", True) and config.get("auto_recover", True) else None}


def detail(service, recovery_id):
    store = service.store
    row = store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    if not row:
        raise KeyError(recovery_id)
    row["decision"] = json.loads(row["decision"] or "null")
    # Only fixed paths under this recovery's artifact directory are read. Raw
    # provider logs can contain credentials and are never served over HTTP.
    root = service.settings.workspace / "recoveries" / str(recovery_id)
    turns = []
    attempts = sorted((p for p in root.glob("attempt-*") if p.name[8:].isdigit()), key=lambda p: int(p.name[8:]))
    for attempt in attempts:
        directories = sorted((p for p in attempt.glob("turn-*") if p.name[5:].isdigit()), key=lambda p: int(p.name[5:])) or [attempt]
        for directory in directories:
            turn = {"attempt": int(attempt.name[8:]), "turn": int(directory.name[5:]) if directory != attempt else 1}
            for name in ("decision", "checks"):
                path = directory / f"{name}.json"
                if not path.exists():
                    continue
                try:
                    if not path.resolve().is_relative_to(root.resolve()) or not path.resolve().is_relative_to(service.settings.workspace.resolve()):
                        raise ValueError("Report path leaves its recovery directory")
                    turn[name] = json.loads(path.read_text())
                except (ValueError, OSError) as exc:
                    turn[f"{name}_error"] = f"Recorded report unavailable: {exc}"
            if len(turn) > 2:
                turns.append(turn)
    events = store.query("SELECT id,kind,message,data,created_at FROM events WHERE campaign_id=? AND json_extract(data,'$.recovery_id')=? ORDER BY id", (row["campaign_id"], recovery_id))
    for event in events:
        event["data"] = json.loads(event["data"])
    history = store.one("SELECT * FROM history_reviews WHERE recovery_id=?", (recovery_id,))
    if history:
        history["result"] = json.loads(history["result"])
    return {"recovery": row, "turns": turns, "events": events, "history": history}


def agent_context(service, *, campaign_id=None):
    # Full metadata index, without a recency cutoff; full narratives are available
    # through the same read-only CLI used by people and the API detail endpoint.
    count = service.store.one("SELECT COUNT(*) AS n FROM recoveries")["n"]
    # Recovery follows its incident even when another campaign is active. The
    # dashboard's default selection and complete report catalog stay unchanged.
    current = current_status(service, campaign_id=campaign_id)
    from .teacher_tools import operator_review_requests
    return {"current": current, "reports": catalog(service, limit=max(1, count))["items"],
            "operator_review_requests": operator_review_requests(service.settings.workspace, (current.get("campaign") or {}).get("id")),
            "read_report": "Use .venv/bin/nekaise-loop --workspace <workspace> report <id> to read all decision turns, host checks and recorded outcomes."}
