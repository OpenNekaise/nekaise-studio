"""Application operations shared by HTTP and CLI. UI never executes stage code."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import signal
import time

from .artifacts import Artifacts
from .config import CampaignConfig, STAGES, STAGE_LABELS
from .storage import Store, encode, new_id, now
from .processes import process_start
from .ownership import source_fingerprint, locked


class Conflict(Exception):
    pass


class Service:
    def __init__(self, settings):
        self.settings, self.store, self.artifacts = settings, Store(settings.workspace), Artifacts(settings.workspace)

    def create(self, name, config: CampaignConfig, *, parent_id=None, context_artifact=None):
        campaign_id = new_id("campaign")
        if not parent_id and "general_material_policy" not in config.model_fields_set:
            config = config.model_copy(update={"general_material_policy": "required_v1"})
        if not parent_id and "material_authors" not in config.model_fields_set:
            from .author_config import load_pool
            config = config.model_copy(update={"material_authors": load_pool(self.settings.material_authors_path)})
        config = config.model_copy(update={"corpus_path": str((self.settings.root / config.corpus_path).resolve())})
        with self.store.connect(immediate=True) as db:
            db.execute("INSERT INTO campaigns(id,name,status,config,created_at,updated_at,parent_campaign_id,context_artifact,implementation_hash) VALUES(?,?,'ready',?,?,?,?,?,?)", (campaign_id, name, encode(config.model_dump()), now(), now(), parent_id, context_artifact, source_fingerprint()))
            self.store.event(campaign_id, None, "campaign", "Campaign created", {"name": name}, db=db)
        return self.store.campaign(campaign_id)

    def list_campaigns(self):
        result = self.store.query("SELECT c.*, COALESCE(h.disposition,'keep') AS retention, COALESCE(NULLIF(h.label,''),c.name) AS display_name, h.reason AS retention_reason, h.recovery_id AS retention_review, (SELECT COUNT(*) FROM rounds r WHERE r.campaign_id=c.id AND r.status='complete') AS completed_rounds FROM campaigns c LEFT JOIN run_retention h ON h.campaign_id=c.id ORDER BY c.created_at DESC LIMIT 100")
        for row in result:
            row["config"] = json.loads(row["config"])
        return result

    def action(self, campaign_id, kind, spawn=True, *, actor="operator", reason=None):
        if kind not in {"start", "pause", "resume", "stop", "review"}:
            raise ValueError("Unknown campaign action")
        if actor not in {"operator", "teacher", "supervisor", "orchestrator"}:
            raise ValueError("Unknown action actor")
        campaign = self.store.campaign(campaign_id)
        if actor != "operator" and self.store.operator_cancelled(campaign_id):
            raise Conflict("An explicit operator hold remains in effect")
        if kind == "resume" and campaign["status"] in {"paused", "failed", "stopped", "interrupted", "waiting", "complete"}:
            has_stages = self.store.one("SELECT s.id FROM stage_runs s JOIN rounds r ON s.round_id=r.id WHERE r.campaign_id=? LIMIT 1", (campaign_id,))
            if campaign["status"] == "complete" or (has_stages and campaign.get("implementation_hash") != source_fingerprint()):
                child = self.continue_campaign(campaign_id, reason="Resume after completion or implementation change", start=True, actor=actor)
                if spawn:
                    self.ensure_worker()
                return {"status": child["status"], "campaign_id": child["id"], "continued_from": campaign_id}
        with self.store.connect(immediate=True) as db:
            row = db.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if not row:
                raise KeyError(campaign_id)
            if actor != "operator" and self.store.operator_cancelled(campaign_id, db=db):
                raise Conflict("An explicit operator hold remains in effect")
            if kind in {"start", "resume", "review"}:
                valid = {"ready"} if kind == "start" else {"paused", "failed", "stopped", "interrupted", "waiting"}
                if kind == "review":
                    valid.add("recovering")
                    if not campaign["config"].get("auto_recover", True):
                        raise Conflict("Automatic orchestration is disabled for this campaign")
                if row["status"] not in valid:
                    raise Conflict(f"Cannot {kind} a {row['status']} campaign")
                active = db.execute("SELECT id FROM campaigns WHERE id!=? AND status IN ('running','queued','pausing','stopping','waiting','recovering') LIMIT 1", (campaign_id,)).fetchone()
                if active:
                    raise Conflict("Another campaign is already queued or running")
                db.execute("UPDATE actions SET handled_at=? WHERE campaign_id=? AND handled_at IS NULL AND kind IN ('pause','stop')", (now(), campaign_id))
                db.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE campaign_id=? AND status IN ('pending','waiting','decided')", (now(), campaign_id))
                if kind == "review":
                    # The active repair may finish its checks and report, but its
                    # older decision must not supersede this queued investigation.
                    db.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE campaign_id=? AND status='running'", (now(), campaign_id))
                if actor == "operator":
                    db.execute("UPDATE campaigns SET operator_hold=NULL WHERE id=?", (campaign_id,))
                if kind != "review" and actor == "operator":
                    db.execute("UPDATE campaigns SET teacher_budget_since=? WHERE id=?", (now(), campaign_id))
                status = "recovering" if kind == "review" else "queued"
            else:
                if row["status"] not in {"running", "queued", "pausing", "waiting", "recovering", "paused", "interrupted", "failed", "stopped"}:
                    raise Conflict(f"Cannot {kind} a {row['status']} campaign")
                status = "pausing" if kind == "pause" else "stopping"
                if actor == "operator":
                    db.execute("UPDATE campaigns SET operator_hold=? WHERE id=?", (kind, campaign_id))
                db.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE campaign_id=? AND status IN ('pending','waiting','decided')", (now(), campaign_id))
            db.execute("UPDATE campaigns SET status=?,updated_at=? WHERE id=?", (status, now(), campaign_id))
            action_id = db.execute("INSERT INTO actions(campaign_id,kind,created_at,actor,reason) VALUES(?,?,?,?,?)", (campaign_id, kind, now(), actor, reason)).lastrowid
            self.store.event(campaign_id, None, "action", f"{kind.capitalize()} requested", {"action": kind, "action_id": action_id, "actor": actor, "reason": reason}, db=db)
        if spawn:
            self.ensure_worker()
        return {"action_id": action_id, "status": status}

    def reconcile_execution(self):
        """Repair a missing operations handoff once the workspace is quiescent.

        This schedules investigation, never a training/recovery decision. Existing
        recovery timers and explicit holds remain authoritative across restarts.
        """
        with self.store.connect(immediate=True) as db:
            if db.execute("SELECT id FROM actions WHERE handled_at IS NULL LIMIT 1").fetchone():
                return None
            if db.execute("SELECT id FROM campaigns WHERE status IN ('running','queued','pausing','stopping') LIMIT 1").fetchone():
                return None
            if db.execute("SELECT id FROM recoveries WHERE status IN ('pending','running','waiting','decided') LIMIT 1").fetchone():
                return None
            row = db.execute("""SELECT c.* FROM campaigns c
                WHERE c.status IN ('paused','stopped','failed','interrupted','waiting','recovering')
                AND c.operator_hold IS NULL AND json_extract(c.config,'$.auto_recover')=1
                AND NOT EXISTS (SELECT 1 FROM campaigns child WHERE child.parent_campaign_id=c.id)
                ORDER BY c.updated_at DESC LIMIT 1""").fetchone()
            if not row or self.store.operator_cancelled(row["id"], db=db):
                return None
            action = db.execute("SELECT * FROM actions WHERE campaign_id=? ORDER BY id DESC LIMIT 1", (row["id"],)).fetchone()
            evidence = f"Execution is {row['status']} with no pending work or scheduled review."
            if row["error"]:
                evidence += f" Recorded interruption: {row['error']}"
            if action:
                evidence += f" Latest command: {action['kind']} by {action['actor']}. {action['reason'] or ''}"
            recovery_id = self.store.recover(row["id"], "status_review", evidence, db=db)
            self.store.event(row["id"], None, "execution_reconciled", evidence,
                             {"recovery_id": recovery_id, "previous_status": row["status"],
                              "action_id": action["id"] if action else None}, db=db)
            return recovery_id

    def ensure_worker(self):
        # The lightweight supervisor launches workers and wakes recovery agents.
        if self.settings.supervisor_service:
            subprocess.run(["systemctl", "--user", "--no-block", "start", self.settings.supervisor_service], check=True, timeout=10)
            return
        lock_path = self.settings.workspace / "supervisor.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            fcntl.flock(lock, fcntl.LOCK_UN)
        with (self.settings.workspace / "supervisor.log").open("a") as log:
            subprocess.Popen([sys.executable, "-m", "nekaise_loop.cli", "--workspace", str(self.settings.workspace), "supervisor"], cwd=self.settings.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)

    def continue_campaign(self, campaign_id, updates=None, reason="Continue learning", *, start=False, actor="operator", recovery_id=None):
        from .artifacts import verify_checkpoint
        from .restoration import base_reference, verify_restoration
        from .training import recipe_hash, training_code_hash
        updates = dict(updates or {})
        if "restore_base_from_round" in updates and (not isinstance(updates["restore_base_from_round"], str) or not updates["restore_base_from_round"]):
            raise ValueError("Base restoration requires a completed historical round ID")
        restore_round = updates.pop("restore_base_from_round", None)
        if restore_round is not None and (actor != "orchestrator" or recovery_id is None):
            raise Conflict("Base restoration requires an explicit orchestrator recovery decision")
        parent = self.store.campaign(campaign_id)
        if parent["status"] in {"running", "queued", "pausing", "stopping"}:
            raise Conflict("Pause or stop the campaign before creating a continuation")
        latest = self.store.one("SELECT s.artifact FROM stage_runs s JOIN rounds r ON s.round_id=r.id WHERE r.campaign_id=? AND s.stage='train' AND s.status='complete' ORDER BY r.number DESC,s.attempt DESC LIMIT 1", (campaign_id,))
        config = CampaignConfig.model_validate(parent["config"]).model_dump()
        if latest:
            trained = self.artifacts.get(latest["artifact"])
            verify_checkpoint(trained)
            config["student_model"] = trained["checkpoint"]
            # inherit_optimizer=false is a one-use reset, consumed by an actual
            # completed update. A prompt/control-plane continuation must not
            # replay that reset just because the immutable campaign config keeps
            # its original value. Diagnostics alone leave the reset pending.
            consumed_reset = self.store.one("""SELECT s.id FROM stage_runs s JOIN rounds r ON r.id=s.round_id
                WHERE r.campaign_id=? AND s.stage='train' AND s.status='complete'
                AND r.checkpoint!=r.model_before LIMIT 1""", (campaign_id,))
            if consumed_reset:
                config["inherit_optimizer"] = True
        complete = self.store.one("SELECT COUNT(*) AS n FROM rounds WHERE campaign_id=? AND status='complete'", (campaign_id,))["n"]
        if config["rounds"] != -1 and parent["status"] != "complete":
            config["rounds"] = max(1, config["rounds"]-complete)
        restoration = None
        if restore_round is not None:
            if updates.get("inherit_optimizer") is not False:
                raise ValueError("Base restoration requires inherit_optimizer=false")
            if "student_model" in updates:
                raise ValueError("Base restoration cannot accept an arbitrary student path")
            reference = base_reference(self.store, self.artifacts, restore_round)
            if reference["checkpoint"] == config["student_model"]:
                raise ValueError("Base restoration target is already the current checkpoint")
            restoration = {"reference": reference, "retained_checkpoint": config["student_model"],
                           "parent_campaign_id": campaign_id, "recovery_id": recovery_id,
                           "reason": reason, "inherit_optimizer": False}
            config["student_model"] = reference["checkpoint"]
        elif parent.get("context_artifact"):
            prior_restoration = self.artifacts.get(parent["context_artifact"]).get("restoration")
            if prior_restoration and config["student_model"] == prior_restoration["reference"]["checkpoint"]:
                verify_restoration(self.store, self.artifacts, prior_restoration)
                restoration = prior_restoration
        author_change = None
        if "material_authors" in updates or "remove_material_author_ids" in updates:
            from .author_config import AuthorPool, merge_pool
            from .artifacts import digest
            before = AuthorPool.model_validate(config["material_authors"])
            after = merge_pool(before, updates.get("material_authors", {}), updates.pop("remove_material_author_ids", []))
            updates["material_authors"] = after.model_dump()
            old = {a.id: a for a in before.authors}
            new = {a.id: a for a in after.authors}
            author_change = {"before_hash": digest(before.model_dump()), "after_hash": digest(after.model_dump()),
                "added": [i for i in new if i not in old], "removed": [i for i in old if i not in new],
                "updated": [i for i in new if i in old and new[i] != old[i]], "reason": reason}
        config.update(updates)
        config = CampaignConfig.model_validate(config)
        config = config.model_copy(update={"corpus_path": str((self.settings.root/config.corpus_path).resolve())})
        if latest:
            manifest = trained["manifest"]
            if manifest.get("recipe_hash") and (manifest["recipe_hash"] != recipe_hash(config.model_dump()) or manifest.get("training_code") != training_code_hash()):
                config = config.model_copy(update={"inherit_optimizer": False})
        from .teacher_tools import latest_strategy
        context = {"parent_campaign_id": campaign_id, "history_access": "all_workspace_campaigns", "gaps": latest_strategy(self.settings.workspace, campaign_id).get("gaps", [])}
        if author_change:
            context["material_author_update"] = author_change
        if restoration:
            context["restoration"] = restoration
        context_key, child_id = self.artifacts.put(context), new_id("campaign")
        with self.store.connect(immediate=True) as db:
            if recovery_id is not None:
                recovery = db.execute("SELECT status FROM recoveries WHERE id=? AND campaign_id=?", (recovery_id, campaign_id)).fetchone()
                if not recovery or recovery["status"] != "decided":
                    raise Conflict("Recovery decision was superseded while preparing its continuation")
            current = db.execute("SELECT status FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if current["status"] != parent["status"]:
                raise Conflict("Campaign state changed while preparing its continuation")
            if actor != "operator" and self.store.operator_cancelled(campaign_id, db=db):
                raise Conflict("An explicit operator hold remains in effect")
            if db.execute("SELECT id FROM campaigns WHERE id!=? AND status IN ('running','queued','pausing','stopping','waiting','recovering')", (campaign_id,)).fetchone():
                raise Conflict("Another campaign is already active")
            budget_since = now() if start and actor == "operator" else parent.get("teacher_budget_since") or parent["created_at"]
            db.execute("INSERT INTO campaigns(id,name,status,config,created_at,updated_at,parent_campaign_id,context_artifact,implementation_hash,teacher_budget_since) VALUES(?,?,?,?,?,?,?,?,?,?)", (child_id, parent["name"][:75] + " · continued", "queued" if start else "ready", encode(config.model_dump()), now(), now(), campaign_id, context_key, source_fingerprint(), budget_since))
            self.store.event(child_id, None, "campaign", "Continuation created", {"parent_campaign_id": campaign_id, "inherit_optimizer": config.inherit_optimizer}, db=db)
            if restore_round is not None:
                self.store.event(child_id, None, "checkpoint_restoration", "Original Base selected; prior checkpoint and teaching lineage preserved", restoration, db=db)
            if start:
                db.execute("INSERT INTO actions(campaign_id,kind,created_at,actor,reason) VALUES(?,'start',?,?,?)", (child_id, now(), actor, reason))
            recovering = db.execute("SELECT id FROM recoveries WHERE campaign_id=? AND status IN ('pending','running','waiting','decided')", (campaign_id,)).fetchall()
            db.execute("UPDATE recoveries SET status='resolved',continuation_id=?,updated_at=? WHERE campaign_id=? AND status IN ('pending','running','waiting','decided')", (child_id, now(), campaign_id))
            for recovery in recovering:
                self.store.event(campaign_id, None, "recovery_applied", "Continuation created; start queued" if start else "Continuation created", {"recovery_id": recovery["id"], "continuation_id": child_id, "start_queued": start}, db=db)
            if parent["status"] in {"waiting", "recovering"}:
                db.execute("UPDATE campaigns SET status='interrupted',updated_at=? WHERE id=?", (now(), campaign_id))
            self.store.event(campaign_id, None, "continuation", reason, {"campaign_id": child_id}, db=db)
        return self.store.campaign(child_id)

    def round_detail(self, round_id):
        row = self.store.one("SELECT * FROM rounds WHERE id=?", (round_id,))
        if not row:
            raise KeyError(round_id)
        from .checkpoint_retention import receipt
        row['checkpoint_retention'] = receipt(Path(row['checkpoint'])) if row.get('checkpoint') else None
        row["stages"] = self.store.query("SELECT id,stage,attempt,status,input_hash,artifact,started_at,finished_at,error FROM stage_runs WHERE round_id=? ORDER BY id", (round_id,))
        row["lessons"] = self.store.records(round_id, "lesson")
        material_page = self.materials(round_id)
        row["materials"] = material_page["rows"]
        row["material_count"] = material_page["total"]
        row["material_next_offset"] = material_page["next_offset"]
        row["evaluations"] = self.store.records(round_id, "evaluation")
        row["gaps"] = self.store.records(round_id, "gap")
        row["metrics"] = self.store.metrics(round_id)
        from .learning_work import safe_round_work
        row["learning_work"] = safe_round_work(self.store, self.artifacts, round_id)
        from .experiments import safe_detail
        row["experiment"] = safe_detail(self.store, self.artifacts, round_id)
        row["score"] = self._score(row["evaluations"])
        row["curriculum"] = None
        row["teaching_strategy"] = None
        for stage, key in (("select", "curriculum"), ("material_select", "curriculum"), ("adapt", "teaching_strategy")):
            artifact = self.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage=? AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id, stage))
            if artifact:
                output = self.artifacts.get(artifact["artifact"])
                row[key] = output.get("curriculum") if stage in {"select", "material_select"} else output if "student_notes" in output else None
        return row

    def materials(self, round_id, offset=0, limit=20):
        if not self.store.one("SELECT id FROM rounds WHERE id=?", (round_id,)):
            raise KeyError(round_id)
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Invalid material page")
        rows = self.store.query("SELECT data FROM records WHERE round_id=? AND kind='material' ORDER BY position,id LIMIT ? OFFSET ?", (round_id, limit, offset))
        total = self.store.one("SELECT COUNT(*) AS n FROM records WHERE round_id=? AND kind='material'", (round_id,))["n"]
        return {"rows": [json.loads(r["data"]) for r in rows], "total": total,
                "next_offset": offset+len(rows) if offset+len(rows) < total else None}

    @staticmethod
    def _score(items):
        grades = [r["grade"]["score"] for r in items if r.get("grade")]
        return sum(grades)/len(grades) if grades else None

    def snapshot(self, campaign_id, round_id=None):
        campaign = self.store.campaign(campaign_id)
        rounds = self.store.query("SELECT * FROM rounds WHERE campaign_id=? ORDER BY number DESC LIMIT 100", (campaign_id,))
        for row in rounds:
            row["score"] = self._score(self.store.records(row["id"], "evaluation"))
        selected = round_id or (rounds[0]["id"] if rounds else None)
        if selected and selected not in {r["id"] for r in rounds}:
            older = self.store.one("SELECT * FROM rounds WHERE id=? AND campaign_id=?", (selected, campaign_id))
            if not older:
                raise KeyError(selected)
            rounds.append(older)
        detail = self.round_detail(selected) if selected else None
        events = self.store.query("SELECT * FROM events WHERE campaign_id=? ORDER BY id DESC LIMIT 100", (campaign_id,))
        for event in events:
            event["data"] = json.loads(event["data"])
        usage = self.store.one("SELECT COUNT(*) AS calls, COALESCE(SUM(json_extract(usage,'$.cost_usd')),0) AS cost_usd, COUNT(json_extract(usage,'$.cost_usd'))>0 AS cost_reported FROM teacher_calls WHERE campaign_id=?", (campaign_id,))
        usage["cost_reported"] = bool(usage["cost_reported"])
        recovery = self.store.one("SELECT id,kind,status,error,retry_at,attempts,decision,continuation_id FROM recoveries WHERE campaign_id=? ORDER BY id DESC LIMIT 1", (campaign_id,))
        if detail:
            frozen = self.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='freeze' AND status='complete' ORDER BY attempt DESC LIMIT 1", (detail["id"],))
            detail["token_ledger"] = self.artifacts.get(frozen["artifact"]).get("ledger") if frozen else None
        return {"campaign": campaign, "rounds": rounds, "round": detail, "events": events, "recovery": recovery, "teacher_usage": usage, "stages": [{"id": s, "label": STAGE_LABELS[s]} for s in STAGES], "timestamp": now()}

    def readiness(self):
        settings = self.settings
        model_cache = Path.home()/".cache/huggingface/hub/models--openbmb--MiniCPM5-1B-Base"
        return {"model_python": {"available": Path(settings.model_python).is_file(), "path": settings.model_python}, "claude": {"available": bool(shutil.which(settings.claude))}, "codex": {"available": bool(shutil.which(settings.codex))}, "student_cache": {"available": model_cache.exists(), "model": "openbmb/MiniCPM5-1B-Base"}, "corpus": {"available": (settings.root.parent/"nekaise-corpus/corpus").is_dir()}, "worker": {"available": self.worker_alive()}, "storage": {"available": True, "path": str(settings.workspace)}, "training": "CoAPT Mid-training; full-parameter training on corrected prose and QA text", "extensions": {"post_training": "planned", "agentic_sft": "planned", "opd": "planned"}}

    def worker_alive(self):
        with (self.settings.workspace/"worker.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True

    def shutdown_worker(self):
        for campaign in self.store.query("SELECT id FROM campaigns WHERE status IN ('running','queued','pausing','waiting','recovering')"):
            self.action(campaign["id"], "stop", spawn=False)
        if self.settings.supervisor_service:
            subprocess.run(["systemctl", "--user", "stop", self.settings.supervisor_service], check=True, timeout=20)
        supervisor_lock = self.settings.workspace/"supervisor.lock"
        if locked(supervisor_lock):
            identity = json.loads(supervisor_lock.read_text())
            if process_start(identity["pid"]) == identity["start"]:
                os.kill(identity["pid"], signal.SIGTERM)
                deadline = time.monotonic()+5
                while locked(supervisor_lock) and time.monotonic()<deadline:
                    time.sleep(0.1)
        deadline = time.monotonic()+15
        while locked(self.settings.workspace/"recovery.lock") and time.monotonic()<deadline:
            time.sleep(0.1)
        if locked(self.settings.workspace/"recovery.lock"):
            raise Conflict("Recovery is still stopping; inspect orchestrator.log")
        if not self.worker_alive():
            return {"status": "worker stopped", "supervisor": "stopped"}
        identity = json.loads((self.settings.workspace/"worker.lock").read_text())
        if process_start(identity["pid"]) != identity["start"]:
            raise Conflict("Worker identity changed; retry")
        os.kill(identity["pid"], signal.SIGTERM)
        deadline = time.monotonic()+15
        while self.worker_alive() and time.monotonic()<deadline:
            time.sleep(0.1)
        if self.worker_alive():
            raise Conflict("Worker has not stopped yet; inspect worker.log")
        return {"status": "worker stopped"}
