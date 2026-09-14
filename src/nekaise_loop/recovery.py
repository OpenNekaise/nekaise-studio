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
    field: Literal["focus", "source_prefix", "lessons_per_round", "passage_chars", "max_seq_len", "learning_rate", "tokens_per_update", "train_epochs", "train_steps", "max_stage_seconds", "inherit_optimizer"]
    value: str = Field(description="JSON-encoded value for this field")


class RecoveryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["retry", "wait", "continue", "pause"]
    reason: str = Field(min_length=1, max_length=3000)
    retry_seconds: int = Field(ge=30, le=86400)
    config_updates: list[ConfigChange] = Field(max_length=12)


def later(seconds):
    return (datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def operator_cancelled(store, campaign_id):
    return store.campaign(campaign_id)["status"] in {"pausing", "stopping", "paused", "stopped"} or bool(store.one("SELECT id FROM actions WHERE campaign_id=? AND handled_at IS NULL AND kind IN ('pause','stop')", (campaign_id,)))


def defer(service, row, reason, seconds, *, decision=None):
    if operator_cancelled(service.store, row["campaign_id"]):
        service.store.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE id=?", (now(), row["id"]))
        return
    retry_at = later(seconds) if seconds is not None else None
    service.store.execute("UPDATE recoveries SET status='waiting',error=?,retry_at=?,decision=?,updated_at=? WHERE id=?", (reason[-3000:], retry_at, encode(decision) if decision else row.get("decision"), now(), row["id"]))
    service.store.set_status(row["campaign_id"], "waiting", reason)
    service.store.event(row["campaign_id"], row["round_id"], "recovery_wait", reason, {"recovery_id": row["id"], "retry_at": retry_at})


def run_agent(settings, row, campaign, directory, runner):
    config = CampaignConfig.model_validate(campaign["config"])
    schema = RecoveryDecision.model_json_schema()
    snapshot = Service(settings).snapshot(campaign["id"])
    # Pass bounded operational context; lesson/source text stays in local artifacts.
    context = {"incident": row, "campaign": campaign, "rounds": [{k:r[k] for k in ("id", "number", "status", "stage", "model_before", "checkpoint", "error")} for r in snapshot["rounds"][:3]], "events": snapshot["events"][:20]}
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
    if previous_attempts >= config.max_repair_attempts:
        defer(service, row, "Automatic repair attempts used. Inspect recovery logs, then Resume to retry training.", None)
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
    store.execute("UPDATE recoveries SET status='running',attempts=?,updated_at=? WHERE id=?", (attempt, now(), recovery_id))
    store.set_status(campaign["id"], "recovering", row["error"])
    store.event(campaign["id"], row["round_id"], "orchestrator", "Orchestrator started", {"recovery_id": recovery_id, "provider": config.orchestrator_provider, "model": config.orchestrator_model, "attempt": attempt})
    runner = ProcessRunner(store, recovery_id, lambda: operator_cancelled(store, campaign["id"]), table="recoveries")
    try:
        decision = RecoveryDecision.model_validate(agent(settings, row, campaign, directory, runner)).model_dump()
        changed = before != source_fingerprint()
        if changed:
            runner.run([sys.executable, "-m", "pytest", "-q"], cwd=settings.root, log=directory/"checks.log", timeout=600, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            decision["action"] = "continue" if decision["action"] == "retry" else decision["action"]
        decision["source_changed"] = changed
        atomic_write(directory/"decision.json", canonical(decision))
        # A fresh Python process applies the decision so it sees any repaired code.
        store.execute("UPDATE recoveries SET status='decided',decision=?,updated_at=? WHERE id=?", (encode(decision), now(), recovery_id))
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
        defer(service, row, f"Orchestrator: {exc}", config.teacher_retry_seconds if unavailable else 60)


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
            defer(service, row, f"Recovery decision could not be applied: {exc}", None)


def _apply(service, recovery_id):
    row = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    if not row or row["status"] != "decided":
        return
    if operator_cancelled(service.store, row["campaign_id"]):
        service.store.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE id=?", (now(), recovery_id))
        return
    decision = json.loads(row["decision"])
    if decision["action"] in {"wait", "pause"}:
        defer(service, row, decision["reason"], decision["retry_seconds"] if decision["action"] == "wait" else None, decision=decision)
        return
    updates = {item["field"]: json.loads(item["value"]) for item in decision["config_updates"]}
    campaign = service.store.campaign(row["campaign_id"])
    if updates or decision["action"] == "continue" or campaign.get("implementation_hash") != source_fingerprint():
        service.continue_campaign(campaign["id"], updates, decision["reason"], start=True)
        return
    with service.store.connect(immediate=True) as db:
        current = db.execute("SELECT status FROM campaigns WHERE id=?", (campaign["id"],)).fetchone()
        if current["status"] not in {"recovering", "waiting"}:
            return
        db.execute("UPDATE recoveries SET status='resolved',updated_at=? WHERE id=?", (now(), recovery_id))
        db.execute("UPDATE campaigns SET status='queued',error=NULL,teacher_budget_since=?,updated_at=? WHERE id=?", (now(), now(), campaign["id"]))
        db.execute("INSERT INTO actions(campaign_id,kind,created_at) VALUES(?,'resume',?)", (campaign["id"], now()))
        service.store.event(campaign["id"], row["round_id"], "recovery", "Orchestrator queued stage retry", {"recovery_id": recovery_id}, db=db)
