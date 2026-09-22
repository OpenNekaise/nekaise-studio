"""Explicit orchestrator funding for one unfinished author batch, never a usage reset."""
from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from .author_config import AuthorPool
from .storage import encode, now


class MaterialAllowance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_id: str
    budget_since: str
    expected_calls: int = Field(ge=0)
    expected_reserved_tokens: int = Field(ge=0)
    additional_calls: int = Field(ge=0)
    additional_output_tokens: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=3000)


def allowance_totals(db, round_id, since, pool):
    used = db.execute("""SELECT COUNT(*) AS calls,COALESCE(SUM(c.reserved_tokens),0) AS tokens,
        COALESCE(SUM(CASE WHEN json_type(c.usage,'$.completion_tokens')='integer'
            THEN MAX(0,json_extract(c.usage,'$.completion_tokens')-c.reserved_tokens) ELSE 0 END),0) AS overrun
        FROM material_calls c JOIN material_jobs j ON j.id=c.job_id
        WHERE j.round_id=? AND c.created_at>=?""", (round_id, since)).fetchone()
    grants = db.execute("""SELECT COALESCE(SUM(additional_calls),0) AS calls,
        COALESCE(SUM(additional_output_tokens),0) AS tokens FROM material_allowances
        WHERE round_id=? AND budget_since=?""", (round_id, since)).fetchone()
    return {"used_calls": used["calls"], "reserved_tokens": used["tokens"],
            "reported_output_overrun_tokens": used["overrun"],
            "granted_calls": grants["calls"], "granted_output_tokens": grants["tokens"],
            "remaining_calls": pool.max_calls_per_round + grants["calls"] - used["calls"],
            "remaining_output_tokens": pool.max_output_tokens_per_round + grants["tokens"] - used["tokens"] - used["overrun"]}


def allowance_status(store, artifacts, campaign_id, *, db=None):
    if db is None:
        with store.connect() as connection:
            return allowance_status(store, artifacts, campaign_id, db=connection)
    campaign = db.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
    current = db.execute("SELECT * FROM rounds WHERE campaign_id=? ORDER BY number DESC LIMIT 1", (campaign_id,)).fetchone()
    if not current or not db.execute("SELECT name FROM sqlite_master WHERE name='material_jobs'").fetchone():
        return None
    pool = AuthorPool.model_validate(json.loads(campaign["config"]).get("material_authors", {}))
    since = campaign["teacher_budget_since"] or campaign["created_at"]
    jobs = db.execute("SELECT * FROM material_jobs WHERE round_id=?", (current["id"],)).fetchall()
    if not jobs:
        return None
    pending = [job for job in jobs if job["status"] != "complete"]
    requirements_error = None
    try:
        needed_tokens = sum(artifacts.get(job["input_artifact"])["request"]["max_tokens"] for job in pending)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A missing/corrupt request is itself evidence for the recovery agent;
        # the allowance projection must not prevent that agent from starting.
        needed_tokens = None
        requirements_error = f"Cannot read unfinished author request artifacts ({type(exc).__name__}); inspect saved jobs"
    totals = allowance_totals(db, current["id"], since, pool)
    calls_by_status = [dict(row) for row in db.execute("""SELECT c.status,COUNT(*) AS calls,
        SUM(c.reserved_tokens) AS reserved_tokens FROM material_calls c JOIN material_jobs j ON j.id=c.job_id
        WHERE j.round_id=? AND c.created_at>=? GROUP BY c.status""", (current["id"], since))]
    return {"round_id": current["id"], "budget_since": since, "round_stage": current["stage"],
            "round_status": current["status"], **totals, "needed_calls": len(pending),
            "base_calls": pool.max_calls_per_round, "base_output_tokens": pool.max_output_tokens_per_round,
            "supplemental_calls_remaining": max(0, pool.max_calls_per_round - totals["granted_calls"]),
            "supplemental_output_tokens_remaining": max(0, pool.max_output_tokens_per_round - totals["granted_output_tokens"]),
            "calls_by_status": calls_by_status,
            "needed_output_tokens": needed_tokens,
            "requirements_error": requirements_error,
            "shortfall_calls": max(0, len(pending) - totals["remaining_calls"]),
            "shortfall_output_tokens": max(0, needed_tokens - totals["remaining_output_tokens"]) if needed_tokens is not None else None,
            "jobs": [{"id": j["id"], "status": j["status"], "error": j["error"]} for j in jobs],
            "automatic_time_renewal": False}


def apply_allowance(service, recovery, grant, db):
    """Caller atomically records this grant, resolves recovery and queues retry."""
    grant = MaterialAllowance.model_validate(grant)
    state = allowance_status(service.store, service.artifacts, recovery["campaign_id"], db=db)
    if (not state or state["round_id"] != grant.round_id or state["budget_since"] != grant.budget_since
            or recovery["round_id"] != grant.round_id or state["round_stage"] != "expand"
            or state["round_status"] not in {"failed", "waiting", "interrupted"}):
        raise ValueError("Material allowance does not match the current interrupted expansion and budget epoch")
    if service.store.operator_cancelled(recovery["campaign_id"], db=db):
        raise ValueError("Explicit operator hold prevents material allowance changes")
    if state["requirements_error"]:
        raise ValueError(state["requirements_error"])
    if db.execute("""SELECT c.id FROM material_calls c JOIN material_jobs j ON j.id=c.job_id
        WHERE j.round_id=? AND (c.status='running' OR c.process_pid IS NOT NULL) LIMIT 1""", (grant.round_id,)).fetchone():
        raise ValueError("Material allowance requires quiescent author calls")
    if (state["used_calls"], state["reserved_tokens"]) != (grant.expected_calls, grant.expected_reserved_tokens):
        raise ValueError("Material reservations changed after the orchestrator inspected them")
    if not state["used_calls"] or not (state["shortfall_calls"] or state["shortfall_output_tokens"]):
        raise ValueError("Material allowance requires a recorded attempt and an actual retry shortfall")
    if (grant.additional_calls, grant.additional_output_tokens) != (state["shortfall_calls"], state["shortfall_output_tokens"]):
        raise ValueError("Material allowance must fund exactly the current unfinished batch shortfall")
    if (grant.additional_calls > state["supplemental_calls_remaining"]
            or grant.additional_output_tokens > state["supplemental_output_tokens_remaining"]):
        raise ValueError("Cumulative material repair allowance is exhausted; investigate and repair the repeated failure before further execution")
    db.execute("""INSERT INTO material_allowances(recovery_id,campaign_id,round_id,budget_since,
        additional_calls,additional_output_tokens,decision,created_at) VALUES(?,?,?,?,?,?,?,?)""",
        (recovery["id"], recovery["campaign_id"], grant.round_id, grant.budget_since,
         grant.additional_calls, grant.additional_output_tokens, encode(grant.model_dump()), now()))
    service.store.event(recovery["campaign_id"], grant.round_id, "material_allowance",
                        "Orchestrator funded the unfinished author batch", {"recovery_id": recovery["id"],
                        **grant.model_dump(), "previous_allowance": state}, db=db)
