"""Durable coding-agent recovery, executed only while training is quiescent."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from .artifacts import atomic_write, canonical
from .config import CampaignConfig
from .failures import quota_kind
from .ownership import source_lock, source_fingerprint
from .processes import ProcessRunner, Cancelled, process_start, stop_owned
from .service import Service, Conflict
from .storage import now, encode
from .history import RunRetention, LogRemoval, inventory, apply_history
from .checkpoint_retention import CheckpointRetention
from .reports import agent_context


def restore_failed_source(settings, directory):
    """Keep the failed patch for inspection; leave the controller importable."""
    before, after = directory/"source-before", directory/"source-failed"
    for base in ("src/nekaise_loop", "prompts"):
        for path in (settings.root/base).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".txt"}:
                relative = path.relative_to(settings.root)
                saved = after/relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, saved)
                if not (before/relative).exists():
                    path.unlink()
    for path in before.rglob("*"):
        if path.is_file():
            target = settings.root/path.relative_to(before)
            if path.suffix == ".md" and target.exists():
                saved = after/path.relative_to(before)
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, saved)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


class ConfigChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["focus", "source_prefix", "lessons_per_round", "passage_chars", "max_seq_len", "learning_rate", "tokens_per_update", "train_epochs", "train_steps", "max_stage_seconds", "inherit_optimizer", "restore_base_from_round", "generation_batch_size", "generation_batch_tokens", "workload_guidance", "material_authors"]
    value: str = Field(description="JSON-encoded value for this field")


class RecoveryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["retry", "wait", "continue", "pause", "check"]
    reason: str = Field(min_length=1, max_length=3000)
    report: str = Field(min_length=1, max_length=12000, description="What was investigated or changed, observed validation results, why this action was chosen, and any outstanding work")
    retry_seconds: int = Field(ge=30, le=86400)
    config_updates: list[ConfigChange] = Field(max_length=12)
    run_retention: list[RunRetention] = Field(max_length=200)
    log_removals: list[LogRemoval] = Field(max_length=200)
    checkpoint_retention: list[CheckpointRetention] = Field(default_factory=list, max_length=200)
    history_review_after_rounds: int = Field(ge=1, le=1000)


def later(seconds):
    return (datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def operator_cancelled(store, campaign_id):
    return store.operator_cancelled(campaign_id)


def defer(service, row, reason, seconds, *, decision=None):
    with service.store.connect(immediate=True) as db:
        current = db.execute("SELECT status FROM recoveries WHERE id=?", (row["id"],)).fetchone()
        if not current or current["status"] not in {"pending", "running", "waiting", "decided"}:
            return
        if service.store.operator_cancelled(row["campaign_id"], db=db):
            db.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE id=?", (now(), row["id"]))
            return
        retry_at = later(seconds) if seconds is not None else None
        db.execute("UPDATE recoveries SET status='waiting',error=?,retry_at=?,decision=?,updated_at=? WHERE id=?", (reason[-3000:], retry_at, encode(decision) if decision else row.get("decision"), now(), row["id"]))
        db.execute("UPDATE campaigns SET status='waiting',error=?,updated_at=? WHERE id=?", (reason, now(), row["campaign_id"]))
        service.store.event(row["campaign_id"], row["round_id"], "campaign", "Campaign waiting", {"status": "waiting", "error": reason}, db=db)
        service.store.event(row["campaign_id"], row["round_id"], "recovery_wait", reason, {"recovery_id": row["id"], "retry_at": retry_at}, db=db)


def run_agent(settings, row, campaign, directory, runner):
    config = CampaignConfig.model_validate(campaign["config"])
    schema = RecoveryDecision.model_json_schema()
    schema["required"] = list(schema["properties"])  # Agents return every field; old saved decisions still default safely.
    snapshot = Service(settings).snapshot(campaign["id"])
    # Pass bounded operational context; lesson/source text stays in local artifacts.
    context = {"incident": row, "campaign": campaign, "rounds": [{k:r[k] for k in ("id", "number", "status", "stage", "model_before", "checkpoint", "error")} for r in snapshot["rounds"][:3]], "events": snapshot["events"][:20]}
    inventory_path = directory/"history-inventory.json"
    atomic_write(inventory_path, canonical(inventory(Service(settings))))
    context["history_inventory"] = str(inventory_path)
    reports_path = directory/"reports.json"
    atomic_write(reports_path, canonical(agent_context(Service(settings), campaign_id=row["campaign_id"])))
    context["reports"] = str(reports_path)
    atomic_write(directory/"incident.json", canonical(context))
    prompt = (settings.root/"prompts/orchestrator.txt").read_text() + "\n\n" + json.dumps({"workspace": str(settings.workspace), "incident_file": str(directory/"incident.json"), "repository": str(settings.root), "teacher_provider": config.teacher_provider, "teacher_model": config.teacher_model}, ensure_ascii=False)
    schema_path, result_path = directory/"schema.json", directory/"response.json"
    atomic_write(schema_path, canonical(schema))
    if config.orchestrator_provider == "codex":
        command = [settings.codex, "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "-C", str(settings.root), "-s", "workspace-write", "-m", config.orchestrator_model, "-c", 'model_reasoning_effort="high"', "--output-schema", str(schema_path), "--output-last-message", str(result_path), "--color", "never", "-"]
        if not settings.workspace.is_relative_to(settings.root):
            command[2:2] = ["--add-dir", str(settings.workspace)]
        runner.run(command, cwd=settings.root, log=directory/"agent.log", timeout=config.orchestrator_timeout, stdin=prompt)
        return RecoveryDecision.model_validate_json(result_path.read_text()).model_dump()
    command = [settings.claude, "-p", "--model", config.orchestrator_model, "--effort", "high", "--permission-mode", "acceptEdits", "--permission-prompts", "none", "--allowedTools", "Read,Edit,Write,Glob,Grep,Bash", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json", "--json-schema", json.dumps(schema)]
    output = runner.run(command, cwd=settings.root, log=directory/"agent.log", timeout=config.orchestrator_timeout, stdin=prompt)
    envelope = json.loads(output)
    if envelope.get("is_error"):
        raise RuntimeError(str(envelope.get("result", "Orchestrator failed")))
    payload = envelope.get("structured_output") or json.loads(envelope.get("result", "{}"))
    return RecoveryDecision.model_validate(payload).model_dump()


def run_checks(settings, directory, runner):
    """Return host observations to the orchestrator, including unsuccessful checks."""
    command = [sys.executable, "-m", "pytest", "-q"]
    result = {"actor": "host", "command": command, "cwd": str(settings.root),
              "source_hash": source_fingerprint(), "started_at": now(),
              "log": str(directory/"checks.log"), "status": "passed", "error": None}
    try:
        runner.run(command, cwd=settings.root, log=directory/"checks.log", timeout=600,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    except Cancelled as exc:
        result.update(status="cancelled", error=str(exc))
        raise
    except Exception as exc:
        # A failed test, timeout or unavailable environment is evidence for the
        # agent's next decision, not a host decision to pause or discard its fix.
        result.update(status="unsuccessful", error=str(exc)[-3000:])
    finally:
        result["finished_at"] = now()
        atomic_write(directory/"checks.json", canonical(result))
    return result


def write_decision(directory, decision):
    atomic_write(directory/"decision.json", canonical(decision))
    report = ("# Orchestrator decision\n\n"
              f"Action: {decision['action']}\n\n{decision['reason']}\n\n{decision['report']}\n")
    atomic_write(directory/"report.md", report.encode())


def handle_recovery(settings, recovery_id, *, agent=run_agent):
    service = Service(settings)
    with (settings.workspace/"recovery.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with source_lock(exclusive=True):
                _handle(service, recovery_id, agent)
        except BlockingIOError:
            return


def _handle(service, recovery_id, agent):
    settings, store = service.settings, service.store
    row = store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    if not row or row["status"] not in {"pending", "waiting", "running"}:
        return
    if operator_cancelled(store, row["campaign_id"]):
        store.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE id=?", (now(), recovery_id))
        return
    if row["process_pid"]:
        stop_owned(row["process_pid"], row["process_start"])
    campaign = store.campaign(row["campaign_id"])
    config = CampaignConfig.model_validate(campaign["config"])
    previous_attempts = store.one("SELECT COALESCE(SUM(x.attempts),0) AS n FROM recoveries x LEFT JOIN stage_runs s ON s.id=x.stage_id WHERE x.campaign_id=? AND x.round_id IS ? AND (s.stage IS (SELECT stage FROM stage_runs WHERE id=?))", (row["campaign_id"], row["round_id"], row["stage_id"]))["n"]
    if previous_attempts >= config.max_repair_attempts and row["status"] == "pending":
        defer(service, row, "Repeated recovery attempts; cooling down before the orchestrator reviews its reports again.", config.teacher_retry_seconds)
        return
    attempt = row["attempts"]+1
    base = settings.workspace/"recoveries"/str(recovery_id)
    execution = 1 + max((int(p.name.removeprefix("attempt-")) for p in base.glob("attempt-*") if p.name.removeprefix("attempt-").isdigit()), default=0)
    directory = base/f"attempt-{execution}"
    directory.mkdir(parents=True, exist_ok=False)
    before = source_fingerprint()
    for base in ("src/nekaise_loop", "prompts"):
        for path in (settings.root/base).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".txt"}:
                target = directory/"source-before"/path.relative_to(settings.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    handbook = settings.root/"docs/COAPT.md"
    target = directory/"source-before/docs/COAPT.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(handbook, target)
    with store.connect(immediate=True) as db:
        current = db.execute("SELECT status FROM recoveries WHERE id=?", (recovery_id,)).fetchone()
        if not current or current["status"] not in {"pending", "waiting", "running"}:
            return
        db.execute("UPDATE recoveries SET status='running',attempts=?,updated_at=? WHERE id=?", (attempt, now(), recovery_id))
        db.execute("UPDATE campaigns SET status='recovering',error=?,updated_at=? WHERE id=?", (row["error"], now(), campaign["id"]))
        store.event(campaign["id"], None, "campaign", "Campaign recovering", {"status": "recovering", "error": row["error"]}, db=db)
        store.event(campaign["id"], row["round_id"], "orchestrator", "Orchestrator started", {"recovery_id": recovery_id, "provider": config.orchestrator_provider, "model": config.orchestrator_model, "attempt": attempt}, db=db)
    runner = ProcessRunner(store, recovery_id, lambda: operator_cancelled(store, campaign["id"]), table="recoveries")
    try:
        feedback = None
        for turn in range(1, config.max_repair_attempts + 2):
            if operator_cancelled(store, campaign["id"]):
                raise Cancelled("Operator cancelled recovery")
            turn_directory = directory/f"turn-{turn}"
            turn_directory.mkdir()
            turn_before = source_fingerprint()
            context = {**row, "feedback": feedback, "attempt_directory": str(directory),
                       "previous_attempts": previous_attempts,
                       "remaining_check_requests": config.max_repair_attempts - turn + 1}
            decision = RecoveryDecision.model_validate(agent(settings, context, campaign, turn_directory, runner)).model_dump()
            decision["source_hash"] = source_fingerprint()
            decision["source_changed"] = before != decision["source_hash"]
            write_decision(turn_directory, decision)
            needs_checks = turn_before != decision["source_hash"] or decision["action"] == "check"
            store.event(campaign["id"], row["round_id"], "orchestrator_decision", decision["reason"],
                        {"recovery_id": recovery_id, "turn": turn, "action": decision["action"],
                         "report": str(turn_directory/"report.md"), "final": not needs_checks})
            if not needs_checks:
                break
            if turn > config.max_repair_attempts:
                raise RuntimeError("Orchestrator exhausted host check requests; all turn reports are preserved")
            checks = run_checks(settings, turn_directory, runner)
            feedback = {"decision": decision, "decision_file": str(turn_directory/"decision.json"),
                        "checks": checks, "checks_file": str(turn_directory/"checks.json")}
            # Passing checks never turn pause into continue. Failed checks never
            # choose pause. The agent sees the evidence and makes the next decision.
        if operator_cancelled(store, campaign["id"]):
            raise Cancelled("Operator cancelled recovery")
        write_decision(directory, decision)
        # A fresh Python process applies the decision so it sees any repaired code.
        # Keep the report even if a newer review cancelled this investigation.
        # Do not revive its decision after that command has been accepted.
        store.execute("UPDATE recoveries SET status='decided',decision=?,updated_at=? WHERE id=? AND status='running'", (encode(decision), now(), recovery_id))
    except Cancelled:
        if source_fingerprint() != before:
            restore_failed_source(settings, directory)
        store.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE id=?", (now(), recovery_id))
    except Exception as exc:
        if source_fingerprint() != before:
            restore_failed_source(settings, directory)
        unavailable = quota_kind(exc)
        if unavailable:
            store.execute("UPDATE recoveries SET attempts=attempts-1 WHERE id=?", (recovery_id,))
        cooldown = min(3600, 60 * 2 ** min(previous_attempts // config.max_repair_attempts, 6))
        defer(service, row, f"Orchestrator: {exc}", config.teacher_retry_seconds if unavailable else cooldown)


def apply_recovery(settings, recovery_id):
    service = Service(settings)
    try:
        with source_lock(exclusive=True):
            _apply(service, recovery_id)
    except BlockingIOError:
        return
    except Exception as exc:
        row = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
        if row:
            defer(service, row, f"Recovery decision could not be applied: {exc}", 60)


def _apply(service, recovery_id):
    row = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    if not row or row["status"] != "decided":
        return
    if operator_cancelled(service.store, row["campaign_id"]):
        service.store.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE id=?", (now(), recovery_id))
        return
    decision = json.loads(row["decision"])
    if "run_retention" in decision:
        apply_history(service, recovery_id, decision)
    if decision["action"] in {"wait", "pause"}:
        # Agent pause suspends training, while operations remain scheduled. Only
        # an explicit operator pause/stop cancels autonomous operation.
        defer(service, row, decision["reason"], decision["retry_seconds"], decision=decision)
        return
    updates = {item["field"]: json.loads(item["value"]) for item in decision["config_updates"]}
    campaign = service.store.campaign(row["campaign_id"])
    if updates or decision["action"] == "continue" or campaign.get("implementation_hash") != source_fingerprint():
        service.continue_campaign(campaign["id"], updates, decision["reason"], start=True, actor="orchestrator", recovery_id=recovery_id)
        return
    with service.store.connect(immediate=True) as db:
        recovery = db.execute("SELECT status FROM recoveries WHERE id=?", (recovery_id,)).fetchone()
        if not recovery or recovery["status"] != "decided":
            return
        current = db.execute("SELECT status FROM campaigns WHERE id=?", (campaign["id"],)).fetchone()
        if current["status"] not in {"recovering", "waiting"} or service.store.operator_cancelled(campaign["id"], db=db):
            return
        db.execute("UPDATE recoveries SET status='resolved',updated_at=? WHERE id=?", (now(), recovery_id))
        db.execute("UPDATE campaigns SET status='queued',error=NULL,teacher_budget_since=?,updated_at=? WHERE id=?", (now(), now(), campaign["id"]))
        db.execute("INSERT INTO actions(campaign_id,kind,created_at,actor,reason) VALUES(?,'resume',?,'orchestrator',?)", (campaign["id"], now(), decision["reason"]))
        service.store.event(campaign["id"], row["round_id"], "recovery_applied", "Orchestrator decision applied; training resume queued", {"recovery_id": recovery_id, "action": decision["action"]}, db=db)
        service.store.event(campaign["id"], row["round_id"], "recovery", "Orchestrator queued stage retry", {"recovery_id": recovery_id}, db=db)
