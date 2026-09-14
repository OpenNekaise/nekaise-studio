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
        config = config.model_copy(update={"corpus_path": str((self.settings.root / config.corpus_path).resolve())})
        with self.store.connect(immediate=True) as db:
            db.execute("INSERT INTO campaigns(id,name,status,config,created_at,updated_at,parent_campaign_id,context_artifact,implementation_hash) VALUES(?,?,'ready',?,?,?,?,?,?)", (campaign_id, name, encode(config.model_dump()), now(), now(), parent_id, context_artifact, source_fingerprint()))
            self.store.event(campaign_id, None, "campaign", "Campaign created", {"name": name}, db=db)
        return self.store.campaign(campaign_id)

    def list_campaigns(self):
        result = self.store.query("SELECT c.*, (SELECT COUNT(*) FROM rounds r WHERE r.campaign_id=c.id AND r.status='complete') AS completed_rounds FROM campaigns c ORDER BY created_at DESC LIMIT 100")
        for row in result:
            row["config"] = json.loads(row["config"])
        return result

    def action(self, campaign_id, kind, spawn=True):
        if kind not in {"start", "pause", "resume", "stop"}:
            raise ValueError("Unknown campaign action")
        campaign = self.store.campaign(campaign_id)
        if kind == "resume" and campaign["status"] in {"paused", "failed", "stopped", "interrupted", "waiting", "complete"}:
            has_stages = self.store.one("SELECT s.id FROM stage_runs s JOIN rounds r ON s.round_id=r.id WHERE r.campaign_id=? LIMIT 1", (campaign_id,))
            if campaign["status"] == "complete" or (has_stages and campaign.get("implementation_hash") != source_fingerprint()):
                child = self.continue_campaign(campaign_id, reason="Resume after completion or implementation change", start=True)
                if spawn:
                    self.ensure_worker()
                return {"status": child["status"], "campaign_id": child["id"], "continued_from": campaign_id}
        with self.store.connect(immediate=True) as db:
            row = db.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if not row:
                raise KeyError(campaign_id)
            if kind in {"start", "resume"}:
                valid = {"ready"} if kind == "start" else {"paused", "failed", "stopped", "interrupted", "waiting"}
                if row["status"] not in valid:
                    raise Conflict(f"Cannot {kind} a {row['status']} campaign")
                active = db.execute("SELECT id FROM campaigns WHERE id!=? AND status IN ('running','queued','pausing','stopping','waiting','recovering') LIMIT 1", (campaign_id,)).fetchone()
                if active:
                    raise Conflict("Another campaign is already queued or running")
                db.execute("UPDATE actions SET handled_at=? WHERE campaign_id=? AND handled_at IS NULL AND kind IN ('pause','stop')", (now(), campaign_id))
                db.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE campaign_id=? AND status IN ('pending','waiting','decided')", (now(), campaign_id))
                db.execute("UPDATE campaigns SET teacher_budget_since=? WHERE id=?", (now(), campaign_id))
                status = "queued"
            else:
                if row["status"] not in {"running", "queued", "pausing", "waiting", "recovering"}:
                    raise Conflict(f"Cannot {kind} a {row['status']} campaign")
                status = "pausing" if kind == "pause" else "stopping"
                db.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE campaign_id=? AND status IN ('pending','waiting','decided')", (now(), campaign_id))
            db.execute("UPDATE campaigns SET status=?,updated_at=? WHERE id=?", (status, now(), campaign_id))
            action_id = db.execute("INSERT INTO actions(campaign_id,kind,created_at) VALUES(?,?,?)", (campaign_id, kind, now())).lastrowid
            self.store.event(campaign_id, None, "action", f"{kind.capitalize()} requested", {"action": kind, "action_id": action_id}, db=db)
        if spawn:
            self.ensure_worker()
        return {"action_id": action_id, "status": status}

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

    def continue_campaign(self, campaign_id, updates=None, reason="Continue learning", *, start=False):
        from .artifacts import verify_checkpoint
        from .training import recipe_hash, training_code_hash
        parent = self.store.campaign(campaign_id)
        if parent["status"] in {"running", "queued", "pausing", "stopping"}:
            raise Conflict("Pause or stop the campaign before creating a continuation")
        latest = self.store.one("SELECT s.artifact FROM stage_runs s JOIN rounds r ON s.round_id=r.id WHERE r.campaign_id=? AND s.stage='train' AND s.status='complete' ORDER BY r.number DESC,s.attempt DESC LIMIT 1", (campaign_id,))
        config = CampaignConfig.model_validate(parent["config"]).model_dump()
        if latest:
            trained = self.artifacts.get(latest["artifact"])
            verify_checkpoint(trained)
            config["student_model"] = trained["checkpoint"]
        complete = self.store.one("SELECT COUNT(*) AS n FROM rounds WHERE campaign_id=? AND status='complete'", (campaign_id,))["n"]
        if config["rounds"] != -1 and parent["status"] != "complete":
            config["rounds"] = max(1, config["rounds"]-complete)
        config.update(updates or {})
        config = CampaignConfig.model_validate(config)
        config = config.model_copy(update={"corpus_path": str((self.settings.root/config.corpus_path).resolve())})
        if latest:
            manifest = trained["manifest"]
            if manifest.get("recipe_hash") and (manifest["recipe_hash"] != recipe_hash(config.model_dump()) or manifest.get("training_code") != training_code_hash()):
                config = config.model_copy(update={"inherit_optimizer": False})
        from .teacher_tools import latest_strategy
        context = {"parent_campaign_id": campaign_id, "history_access": "all_workspace_campaigns", "gaps": latest_strategy(self.settings.workspace, campaign_id).get("gaps", [])}
        context_key, child_id = self.artifacts.put(context), new_id("campaign")
        with self.store.connect(immediate=True) as db:
            current = db.execute("SELECT status FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if current["status"] != parent["status"]:
                raise Conflict("Campaign state changed while preparing its continuation")
            if db.execute("SELECT id FROM campaigns WHERE id!=? AND status IN ('running','queued','pausing','stopping','waiting','recovering')", (campaign_id,)).fetchone():
                raise Conflict("Another campaign is already active")
            db.execute("INSERT INTO campaigns(id,name,status,config,created_at,updated_at,parent_campaign_id,context_artifact,implementation_hash,teacher_budget_since) VALUES(?,?,?,?,?,?,?,?,?,?)", (child_id, parent["name"][:75] + " · continued", "queued" if start else "ready", encode(config.model_dump()), now(), now(), campaign_id, context_key, source_fingerprint(), now() if start else None))
            self.store.event(child_id, None, "campaign", "Continuation created", {"parent_campaign_id": campaign_id, "inherit_optimizer": config.inherit_optimizer}, db=db)
            if start:
                db.execute("INSERT INTO actions(campaign_id,kind,created_at) VALUES(?,'start',?)", (child_id, now()))
            db.execute("UPDATE recoveries SET status='resolved',continuation_id=?,updated_at=? WHERE campaign_id=? AND status IN ('pending','running','waiting','decided')", (child_id, now(), campaign_id))
            if parent["status"] in {"waiting", "recovering"}:
                db.execute("UPDATE campaigns SET status='interrupted',updated_at=? WHERE id=?", (now(), campaign_id))
            self.store.event(campaign_id, None, "continuation", reason, {"campaign_id": child_id}, db=db)
        return self.store.campaign(child_id)

    def round_detail(self, round_id):
        row = self.store.one("SELECT * FROM rounds WHERE id=?", (round_id,))
        if not row:
            raise KeyError(round_id)
        row["stages"] = self.store.query("SELECT id,stage,attempt,status,input_hash,artifact,started_at,finished_at,error FROM stage_runs WHERE round_id=? ORDER BY id", (round_id,))
        row["lessons"] = self.store.records(round_id, "lesson")
        row["evaluations"] = self.store.records(round_id, "evaluation")
        row["gaps"] = self.store.records(round_id, "gap")
        row["metrics"] = self.store.metrics(round_id)
        row["score"] = self._score(row["evaluations"])
        row["curriculum"] = None
        row["teaching_strategy"] = None
        for stage, key in (("select", "curriculum"), ("adapt", "teaching_strategy")):
            artifact = self.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage=? AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id, stage))
            if artifact:
                output = self.artifacts.get(artifact["artifact"])
                row[key] = output.get("curriculum") if stage == "select" else output if "student_notes" in output else None
        return row

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
        return {"model_python": {"available": Path(settings.model_python).is_file(), "path": settings.model_python}, "claude": {"available": bool(shutil.which(settings.claude))}, "codex": {"available": bool(shutil.which(settings.codex))}, "student_cache": {"available": model_cache.exists(), "model": "openbmb/MiniCPM5-1B-Base"}, "corpus": {"available": (settings.root.parent/"nekaise-corpus/corpus").is_dir()}, "worker": {"available": self.worker_alive()}, "storage": {"available": True, "path": str(settings.workspace)}, "training": "Full-parameter CPT; QA examples are serialized as text", "extensions": {"agentic_sft": "planned", "opd": "planned"}}

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
