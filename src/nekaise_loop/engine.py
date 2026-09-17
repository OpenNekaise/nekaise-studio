"""Resumable stage runner. Artifacts are committed before their completion records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import stages
from .artifacts import Artifacts, digest, verify_checkpoint
from .config import CampaignConfig, ROOT, STAGES, STAGE_LABELS, resolve_student
from .teaching import Curriculum
from .processes import Cancelled, ProcessRunner
from .providers.local import LocalModel
from .providers.teacher import CliTeacher
from .storage import Store, encode, new_id, now
from .failures import TeacherUnavailable


class Context:
    def __init__(self, engine, campaign, row, stage_id, attempt, controls):
        self.engine, self.store, self.artifacts = engine, engine.store, engine.artifacts
        self.campaign, self.round = campaign, row
        self.config = CampaignConfig.model_validate(campaign["config"])
        self.stage_id, self.attempt = stage_id, attempt
        self.cancelled = controls
        self.directory = engine.settings.workspace / "runs" / row["id"] / f"{row['stage']}-{attempt}"
        self.directory.mkdir(parents=True, exist_ok=True)
        runner = ProcessRunner(self.store, stage_id, controls)
        self.teacher = engine.teacher_factory(self.config, engine.settings, self.store, campaign["id"], row["id"], runner, self.directory)
        runtime_config = self.config
        if not self.config.inherit_optimizer:
            # Diagnostic rounds retain the parent state; only a completed weight
            # update can consume the continuation's explicit optimizer reset.
            trained = self.store.one("SELECT id FROM rounds WHERE campaign_id=? AND number<? AND status='complete' AND checkpoint!=model_before LIMIT 1", (campaign["id"], row["number"]))
            runtime_config = self.config.model_copy(update={"inherit_optimizer": bool(trained)})
        if row["stage"] in {"freeze", "train"}:
            curriculum = Curriculum.model_validate(self.output("material_select")["curriculum"])
            runtime_config = runtime_config.model_copy(update={"token_mix": curriculum.token_mix, "train_epochs": curriculum.train_epochs})
        local = engine.model_factory(runtime_config, engine.settings, runner, self.directory)
        self.student, self.trainer = local, local

    def output(self, stage):
        row = self.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage=? AND status='complete' ORDER BY attempt DESC LIMIT 1", (self.round["id"], stage))
        if not row:
            raise RuntimeError(f"Missing completed dependency: {stage}")
        return self.artifacts.get(row["artifact"])

    def project(self, kind, rows):
        self.store.put_records(self.round["id"], kind, rows)

    def event(self, kind, message, data=None):
        self.store.event(self.campaign["id"], self.round["id"], kind, message, data)

    def metric(self, data):
        if not isinstance(data.get("loss"), (int, float)) or not 1 <= data["step"] <= data["total_steps"] or (self.config.train_steps and data["total_steps"] != self.config.train_steps):
            raise ValueError("Malformed training metric")
        self.store.execute("INSERT OR REPLACE INTO metrics(round_id,attempt,step,data,created_at) VALUES(?,?,?,?,?)", (self.round["id"], self.attempt, data["step"], encode(data), now()))

class Engine:
    def __init__(self, settings, teacher_factory=CliTeacher, model_factory=LocalModel, material_client_factory=None):
        self.settings, self.store, self.artifacts = settings, Store(settings.workspace), Artifacts(settings.workspace)
        self.teacher_factory, self.model_factory = teacher_factory, model_factory
        self.material_client_factory = material_client_factory

    def run(self, campaign_id, controls=lambda: False, pause=lambda: False):
        campaign = self.store.campaign(campaign_id)
        config = CampaignConfig.model_validate(campaign["config"])
        self.store.set_status(campaign_id, "running")
        try:
            restoration = (self.artifacts.get(campaign["context_artifact"]).get("restoration")
                           if campaign.get("context_artifact") else None)
            restoration_checked = False
            unfinished = self.store.one("SELECT MIN(number) AS number FROM rounds WHERE campaign_id=? AND status!='complete'", (campaign_id,))
            last = self.store.one("SELECT COALESCE(MAX(number),0) AS number FROM rounds WHERE campaign_id=?", (campaign_id,))
            number = unfinished["number"] or last["number"]+1
            while config.rounds == -1 or number <= config.rounds:
                if controls():
                    raise Cancelled("Stopped by operator")
                row = self.store.one("SELECT * FROM rounds WHERE campaign_id=? AND number=?", (campaign_id, number))
                if row and row["status"] == "complete":
                    number += 1
                    continue
                if not row:
                    prior = self.store.one("SELECT checkpoint FROM rounds WHERE campaign_id=? AND number=? AND status='complete'", (campaign_id, number-1))
                    if prior:
                        prior_stage = self.store.one("SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND r.number=? AND s.stage='train' AND s.status='complete' ORDER BY s.attempt DESC LIMIT 1", (campaign_id, number-1))
                        verify_checkpoint(self.artifacts.get(prior_stage["artifact"]))
                    parent = prior["checkpoint"] if prior else resolve_student(config.student_model)
                    initial_manifest = Path(parent) / "checkpoint.json"
                    if not prior and initial_manifest.is_file():
                        verify_checkpoint({"checkpoint": parent, "manifest": json.loads(initial_manifest.read_text())})
                    round_id = new_id("round")
                    self.store.execute("INSERT INTO rounds(id,campaign_id,number,status,model_before,created_at,updated_at) VALUES(?,?,?,'ready',?,?,?)", (round_id, campaign_id, number, parent, now(), now()))
                    row = self.store.one("SELECT * FROM rounds WHERE id=?", (round_id,))
                if restoration and not restoration_checked and row["model_before"] == restoration["reference"]["checkpoint"]:
                    from .restoration import verify_restoration
                    # Recheck before any work using Base, including stage retries
                    # and diagnostic-only rounds. Descendant weights stand alone.
                    verify_restoration(self.store, self.artifacts, restoration)
                    restoration_checked = True
                from .checkpoint_retention import storage_status
                storage = storage_status(self.settings.workspace, row['model_before'])
                if storage['pressure']:
                    self.store.recover(campaign_id, 'storage_pressure', f"Insufficient checkpoint headroom: {storage['free_bytes']} bytes free, {storage['required_bytes']} required. Orchestrator must review checkpoint retention and storage before training.")
                    return
                dependency_hashes = []
                prior_stage = self.store.one("SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND r.number=? AND s.stage='train' AND s.status='complete' ORDER BY s.attempt DESC LIMIT 1", (campaign_id, number-1))
                if prior_stage:
                    verify_checkpoint(self.artifacts.get(prior_stage["artifact"]))
                for stage in STAGES:
                    if controls():
                        raise Cancelled("Stopped by operator")
                    if pause():
                        self.store.set_status(campaign_id, "paused")
                        self.store.execute("UPDATE rounds SET status='paused' WHERE id=?", (row["id"],))
                        return
                    prompt_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "prompts").glob("*.txt"))}
                    prompt_hashes["docs/COAPT.md"] = hashlib.sha256((ROOT/"docs/COAPT.md").read_bytes()).hexdigest()
                    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((ROOT / "src/nekaise_loop").rglob("*.py")))).hexdigest()
                    fingerprint = digest({"config": config.model_dump(), "parent": row["model_before"], "parent_artifact": prior_stage["artifact"] if prior_stage else None, "stage": stage, "dependencies": dependency_hashes, "prompts": prompt_hashes, "code": code_hash})
                    existing = self.store.one("SELECT * FROM stage_runs WHERE round_id=? AND stage=? AND status='complete' ORDER BY attempt DESC LIMIT 1", (row["id"], stage))
                    if existing:
                        if existing["input_hash"] != fingerprint:
                            raise RuntimeError(f"Completed {stage} inputs/code changed. Preserve this run and create a new campaign.")
                        output = self.artifacts.get(existing["artifact"])
                        if stage == "train":
                            verify_checkpoint(output)
                        dependency_hashes.append(existing["artifact"])
                        continue
                    if stage == 'train':
                        storage = storage_status(self.settings.workspace, row['model_before'])
                        if storage['pressure']:
                            self.store.recover(campaign_id, 'storage_pressure', f"Insufficient checkpoint headroom before train: {storage['free_bytes']} bytes free, {storage['required_bytes']} required; review storage retention.")
                            return
                    attempt = self.store.one("SELECT COALESCE(MAX(attempt),0)+1 AS n FROM stage_runs WHERE round_id=? AND stage=?", (row["id"], stage))["n"]
                    with self.store.connect(immediate=True) as db:
                        db.execute("UPDATE rounds SET status='running',stage=?,error=NULL,updated_at=? WHERE id=?", (stage, now(), row["id"]))
                        stage_id = db.execute("INSERT INTO stage_runs(round_id,stage,attempt,status,input_hash,started_at) VALUES(?,?,?,'running',?,?)", (row["id"], stage, attempt, fingerprint, now())).lastrowid
                        self.store.event(campaign_id, row["id"], "stage_started", STAGE_LABELS[stage], {"stage": stage, "attempt": attempt}, db=db)
                    row["stage"] = stage
                    ctx = Context(self, campaign, row, stage_id, attempt, controls)
                    try:
                        result = getattr(stages, stage)(ctx)
                        if stage == "train":
                            verify_checkpoint(result)
                        artifact = self.artifacts.put(result)
                        with self.store.connect(immediate=True) as db:
                            db.execute("UPDATE stage_runs SET status='complete',artifact=?,finished_at=? WHERE id=?", (artifact, now(), stage_id))
                            if stage == "train":
                                db.execute("UPDATE rounds SET checkpoint=? WHERE id=?", (result["checkpoint"], row["id"]))
                            self.store.event(campaign_id, row["id"], "stage_complete", f"{STAGE_LABELS[stage]} complete", {"stage": stage, "artifact": artifact}, db=db)
                        dependency_hashes.append(artifact)
                    except BaseException as exc:
                        state = "interrupted" if isinstance(exc, Cancelled) else "waiting" if isinstance(exc, TeacherUnavailable) else "failed"
                        self.store.execute("UPDATE stage_runs SET status=?,error=?,finished_at=? WHERE id=?", (state, str(exc)[-3000:], now(), stage_id))
                        self.store.execute("UPDATE rounds SET status=?,error=?,updated_at=? WHERE id=?", (state, str(exc)[-3000:], now(), row["id"]))
                        raise
                self.store.execute("UPDATE rounds SET status='complete',updated_at=? WHERE id=?", (now(), row["id"]))
                self.store.event(campaign_id, row["id"], "round_complete", f"Round {number:02d} complete; curriculum updated")
                if controls():
                    raise Cancelled("Stopped by operator")
                if pause():
                    self.store.set_status(campaign_id, "paused")
                    return
                reflection = self.artifacts.get(dependency_hashes[-1])
                decision = reflection.get("action", "continue")
                if decision == "pause":
                    from .service import Service
                    Service(self.settings).action(campaign_id, "pause", spawn=False,
                                                  actor="teacher", reason=reflection.get("reason"))
                    return
                if decision == "complete":
                    break
                if config.auto_recover and config.manage_history:
                    from .history import review_due
                    if review_due(self.store):
                        self.store.recover(campaign_id, "history_review", "Scheduled orchestrator review of run history and training logs")
                        return
                number += 1
            self.store.set_status(campaign_id, "complete")
        except Cancelled:
            self.store.set_status(campaign_id, "stopped")
        except Exception as exc:
            if controls() or pause():
                self.store.set_status(campaign_id, "paused" if pause() else "stopped")
                return
            if isinstance(exc, TeacherUnavailable):
                from datetime import datetime, timedelta, timezone
                previous = self.store.one("SELECT COUNT(*) AS n FROM recoveries x JOIN stage_runs s ON s.id=x.stage_id WHERE x.campaign_id=? AND x.kind=? AND s.round_id=(SELECT round_id FROM stage_runs ORDER BY id DESC LIMIT 1) AND s.stage=(SELECT stage FROM stage_runs ORDER BY id DESC LIMIT 1)", (campaign_id, exc.kind))["n"]
                base_delay = 60 if exc.kind == "rate_limit" else config.teacher_retry_seconds
                delay = exc.retry_seconds or min(21600, base_delay * 2**min(previous, 10))
                retry_at = None if exc.kind == "budget" else (datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat(timespec="milliseconds")
                if config.auto_recover:
                    self.store.recover(campaign_id, exc.kind, exc, retry_at=retry_at)
                else:
                    self.store.set_status(campaign_id, "waiting", str(exc))
            elif config.auto_recover:
                self.store.recover(campaign_id, "failure", exc)
            else:
                self.store.set_status(campaign_id, "failed", str(exc)[-3000:])
            self.store.event(campaign_id, None, "waiting" if isinstance(exc, TeacherUnavailable) else "error", str(exc)[-1500:])
