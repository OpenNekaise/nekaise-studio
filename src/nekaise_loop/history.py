"""Orchestrator-owned run retention and recoverable cleanup of closed raw logs.

Teaching artifacts, datasets, metrics and lineage remain. Explicit checkpoint byte
retirement preserves summaries and immutable provenance; it is not reversible trash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import atomic_write, canonical
from .ownership import source_lock
from .storage import encode, now


ACTIVE = {"running", "queued", "pausing", "stopping", "waiting", "recovering"}


class RunRetention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_id: str
    disposition: Literal["keep", "archive"]
    label: str = Field(max_length=120, description="Concise display name; empty preserves the original name")
    reason: str = Field(min_length=1, max_length=2000)


class LogRemoval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern="^[a-f0-9]{64}$")
    summary: str = Field(min_length=1, max_length=6000)
    reason: str = Field(min_length=1, max_length=2000)


def completed_rounds(store):
    return store.one("SELECT COUNT(*) AS n FROM rounds WHERE status='complete'")["n"]


def review_due(store):
    last = store.one("SELECT next_review_round FROM history_reviews ORDER BY applied_at DESC,recovery_id DESC LIMIT 1")
    return last is None or completed_rounds(store) >= last["next_review_round"]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_path(workspace, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts or parts.parts[0] not in {"runs", "recoveries"} or parts.suffix != ".log":
        raise ValueError("Cleanup targets must be explicit workspace runs/recoveries .log files")
    path = workspace/relative
    if path.resolve() != path.absolute() or path.is_symlink():
        raise ValueError("Symlink log targets are not eligible for cleanup")
    return path


def inventory(service):
    store, workspace = service.store, service.settings.workspace
    campaigns = {r["id"]: r for r in store.query("SELECT id,name,status,parent_campaign_id,created_at,error FROM campaigns ORDER BY created_at DESC")}
    rounds = {r["id"]: r for r in store.query("SELECT id,campaign_id,number,status,stage,model_before,checkpoint FROM rounds ORDER BY created_at")}
    recoveries = {str(r["id"]): r for r in store.query("SELECT id,campaign_id,status,kind,error FROM recoveries")}
    retention = {r["campaign_id"]: r for r in store.query("SELECT * FROM run_retention")}
    for c in campaigns.values():
        c["rounds"] = [r for r in rounds.values() if r["campaign_id"] == c["id"]]
        c["children"] = [x["id"] for x in campaigns.values() if x["parent_campaign_id"] == c["id"]]
        c["retention"] = retention.get(c["id"])
        c["recoveries"] = [r for r in recoveries.values() if r["campaign_id"] == c["id"]]
    logs = []
    for base, owners in (("runs", rounds), ("recoveries", recoveries)):
        for path in sorted((workspace/base).rglob("*.log")):
            relative = path.relative_to(workspace).as_posix()
            owner = owners.get(PurePosixPath(relative).parts[1])
            if not owner or not path.is_file():
                continue
            try:
                log_path(workspace, relative)
            except ValueError:
                continue
            campaign = campaigns[owner["campaign_id"]]
            protected = campaign["status"] in ACTIVE or (base == "recoveries" and owner["status"] in {"pending", "running", "waiting", "decided"})
            logs.append({"path": relative, "campaign_id": campaign["id"], "bytes": path.stat().st_size,
                         "sha256": file_hash(path), "eligible": not protected,
                         "protected_reason": "Active campaign or recovery" if protected else None})
    from .checkpoint_retention import inventory as checkpoint_inventory
    return {"checkpoint_storage": checkpoint_inventory(service), "created_at": now(), "completed_rounds": completed_rounds(store),
            "campaigns": list(campaigns.values()), "logs": logs,
            "previous_reviews": store.query("SELECT * FROM history_reviews ORDER BY applied_at DESC"),
            "cleaned_logs": store.query("SELECT * FROM log_cleanup ORDER BY id"),
            "scope": "Keep/archive run presentation and remove explicit closed raw logs to recoverable trash. Teaching records, artifacts and lineage remain; checkpoint byte availability follows explicit retention decisions."}


def trash_path(workspace, recovery_id, relative):
    path = workspace/"trash/history"/str(recovery_id)/relative
    if path.resolve() != path.absolute():
        raise ValueError("Symlink trash destinations are not allowed")
    return path


def apply_history(service, recovery_id, decision):
    """Apply an agent's exact choices under the caller's exclusive source lock."""
    store, workspace = service.store, service.settings.workspace
    applied = store.one("SELECT result FROM history_reviews WHERE recovery_id=?", (recovery_id,))
    if applied:
        result = json.loads(applied["result"])
        atomic_write(workspace/"recoveries"/str(recovery_id)/"history-result.json", canonical(result))
        return result
    choices = [RunRetention.model_validate(x) for x in decision.get("run_retention", [])]
    removals = [LogRemoval.model_validate(x) for x in decision.get("log_removals", [])]
    interval = decision.get("history_review_after_rounds", 10)
    if not isinstance(interval, int) or not 1 <= interval <= 1000:
        raise ValueError("Next history review must be 1..1000 completed rounds away")
    if len({x.campaign_id for x in choices}) != len(choices) or len({x.path for x in removals}) != len(removals):
        raise ValueError("Duplicate history decisions")
    for choice in choices:
        campaign = store.campaign(choice.campaign_id)
        if choice.disposition == "archive" and campaign["status"] in ACTIVE:
            raise ValueError("An active campaign cannot be archived")
    # Validate the whole batch before touching any file or presentation metadata.
    candidates = {x["path"]: x for x in inventory(service)["logs"]} if removals else {}
    for removal in removals:
        source = log_path(workspace, removal.path)
        target = trash_path(workspace, recovery_id, removal.path)
        saved = store.one("SELECT * FROM log_cleanup WHERE recovery_id=? AND path=?", (recovery_id, removal.path))
        if saved and saved["sha256"] != removal.sha256:
            raise ValueError("Cleanup journal content differs from the decision")
        if saved and saved["status"] in {"removed", "restored"}:
            continue
        if saved and not source.exists() and target.is_file() and file_hash(target) == removal.sha256:
            continue  # recover a process exit between move and journal completion
        item = candidates.get(removal.path)
        if not item or not item["eligible"]:
            raise ValueError(f"Log is not a closed cleanup candidate: {removal.path}")
        if item["sha256"] != removal.sha256 or target.exists():
            raise ValueError(f"Log changed or trash destination exists: {removal.path}")
    from .checkpoint_retention import apply as apply_checkpoints
    checkpoint_results = apply_checkpoints(service, recovery_id, decision.get("checkpoint_retention", []))
    removed = []
    for removal in removals:
        source = log_path(workspace, removal.path)
        target = trash_path(workspace, recovery_id, removal.path)
        saved = store.one("SELECT * FROM log_cleanup WHERE recovery_id=? AND path=?", (recovery_id, removal.path))
        if not saved:
            store.execute("INSERT INTO log_cleanup(recovery_id,path,sha256,summary,reason,bytes,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'planned',?,?)",
                          (recovery_id, removal.path, removal.sha256, removal.summary, removal.reason, source.stat().st_size, now(), now()))
            saved = store.one("SELECT * FROM log_cleanup WHERE recovery_id=? AND path=?", (recovery_id, removal.path))
        if saved["status"] == "planned":
            if source.exists():
                if file_hash(source) != removal.sha256:
                    raise ValueError(f"Log changed during cleanup: {removal.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
            store.execute("UPDATE log_cleanup SET status='removed',updated_at=? WHERE id=?", (now(), saved["id"]))
        removed.append({"id": saved["id"], "path": removal.path, "bytes": saved["bytes"], "summary": removal.summary})
    result = {"checkpoint_retention": checkpoint_results, "run_retention": [x.model_dump() for x in choices], "log_removals": removed,
              "report": decision.get("report", ""), "recoverable": not any(x["deleted"] for x in checkpoint_results)}
    count = completed_rounds(store)
    with store.connect(immediate=True) as db:
        for choice in choices:
            db.execute("INSERT INTO run_retention(campaign_id,disposition,label,reason,recovery_id,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(campaign_id) DO UPDATE SET disposition=excluded.disposition,label=excluded.label,reason=excluded.reason,recovery_id=excluded.recovery_id,updated_at=excluded.updated_at",
                       (choice.campaign_id, choice.disposition, choice.label, choice.reason, recovery_id, now()))
        db.execute("INSERT INTO history_reviews(recovery_id,completed_rounds,next_review_round,result,applied_at) VALUES(?,?,?,?,?)", (recovery_id, count, count+interval, encode(result), now()))
        row = db.execute("SELECT campaign_id FROM recoveries WHERE id=?", (recovery_id,)).fetchone()
        store.event(row["campaign_id"], None, "history_review", "Orchestrator organized run history and logs",
                    {"recovery_id": recovery_id, "runs_reviewed": len(choices), "logs_removed": len(removed), "checkpoint_decisions": len(checkpoint_results), "checkpoint_bytes_released": sum(x["bytes_released"] for x in checkpoint_results), "next_review_round": count+interval}, db=db)
    atomic_write(workspace/"recoveries"/str(recovery_id)/"history-result.json", canonical(result))
    return result


def restore_log(service, cleanup_id):
    with source_lock(exclusive=True):
        row = service.store.one("SELECT * FROM log_cleanup WHERE id=?", (cleanup_id,))
        if not row:
            raise KeyError(cleanup_id)
        if row["status"] == "restored":
            return row
        target = log_path(service.settings.workspace, row["path"])
        saved = trash_path(service.settings.workspace, row["recovery_id"], row["path"])
        if target.is_file() and not saved.exists() and file_hash(target) == row["sha256"]:
            service.store.execute("UPDATE log_cleanup SET status='restored',updated_at=? WHERE id=?", (now(), cleanup_id))
            return service.store.one("SELECT * FROM log_cleanup WHERE id=?", (cleanup_id,))
        if target.exists() or not saved.is_file() or file_hash(saved) != row["sha256"]:
            raise ValueError("Restore requires an unchanged archived log and an empty destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        saved.rename(target)
        service.store.execute("UPDATE log_cleanup SET status='restored',updated_at=? WHERE id=?", (now(), cleanup_id))
        recovery = service.store.one("SELECT campaign_id FROM recoveries WHERE id=?", (row["recovery_id"],))
        service.store.event(recovery["campaign_id"], None, "log_restored", "Restored a cleaned raw log", {"cleanup_id": cleanup_id, "path": row["path"]})
        return service.store.one("SELECT * FROM log_cleanup WHERE id=?", (cleanup_id,))
