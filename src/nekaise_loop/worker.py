"""Independent coordinator process with a durable command queue and one workspace lock."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import time

from .processes import process_start, stop_owned
from .storage import now
from .ownership import source_lock
from .config import CampaignConfig


def run_worker(settings):
    try:
        with source_lock():
            _run_worker(settings)
    except BlockingIOError:
        return


def _run_worker(settings):
    from .engine import Engine
    with (settings.workspace/"worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "start": process_start(os.getpid())}))
        lock.flush()
        engine = Engine(settings)
        store = engine.store
        closing = [False]
        signal.signal(signal.SIGTERM, lambda *_: closing.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: closing.__setitem__(0, True))
        from .material_jobs import recover_author_processes
        recover_author_processes(store)
        for stage in store.query("SELECT * FROM stage_runs WHERE status='running'"):
            if stage["process_pid"]:
                stop_owned(stage["process_pid"], stage["process_start"])
            store.execute("UPDATE stage_runs SET status='interrupted',error='Worker stopped before completing this stage',finished_at=?,process_pid=NULL,process_start=NULL WHERE id=?", (now(), stage["id"]))
        store.execute("UPDATE rounds SET status='interrupted' WHERE status='running'")
        # A queued start survives a restart. Explicit holds survive a consumed
        # command too, including a crash before the status update was persisted.
        for campaign in store.query("SELECT * FROM campaigns WHERE status IN ('running','pausing','stopping')"):
            pending = store.one("SELECT id FROM actions WHERE campaign_id=? AND handled_at IS NULL", (campaign["id"],))
            if pending:
                continue
            if campaign["operator_hold"]:
                store.set_status(campaign["id"], "paused" if campaign["operator_hold"] == "pause" else "stopped")
                continue
            store.set_status(campaign["id"], "interrupted", "Worker exited before completing the campaign")
            if CampaignConfig.model_validate_json(campaign["config"]).auto_recover:
                store.recover(campaign["id"], "worker_exit", "Worker exited before completing the campaign")
        for campaign in store.query("SELECT id FROM campaigns WHERE status='queued'"):
            if not store.one("SELECT id FROM actions WHERE campaign_id=? AND handled_at IS NULL", (campaign["id"],)):
                store.execute("INSERT INTO actions(campaign_id,kind,created_at,actor) VALUES(?,'resume',?,'supervisor')", (campaign["id"], now()))
        while not closing[0]:
            action = store.one("SELECT * FROM actions WHERE handled_at IS NULL ORDER BY id LIMIT 1")
            if not action:
                return
            store.execute("UPDATE actions SET handled_at=? WHERE id=?", (now(), action["id"]))
            campaign_id = action["campaign_id"]
            if action["kind"] == "review":
                store.recover(campaign_id, "status_review", action["reason"] or "Operator requested an orchestrator review")
                continue
            if action["kind"] not in {"start", "resume"}:
                store.set_status(campaign_id, "paused" if action["kind"] == "pause" else "stopped")
                continue
            if store.operator_cancelled(campaign_id):
                hold = store.campaign(campaign_id)["operator_hold"]
                if hold:
                    store.set_status(campaign_id, "paused" if hold == "pause" else "stopped")
                continue
            # An old queued start/resume cannot cross a newer review handoff,
            # including queues persisted by earlier controller implementations.
            if (store.one("SELECT id FROM actions WHERE campaign_id=? AND kind='review' AND handled_at IS NULL", (campaign_id,))
                    or store.one("SELECT id FROM recoveries WHERE campaign_id=? AND status IN ('pending','running','waiting','decided')", (campaign_id,))):
                continue
            stop_requested, pause_requested = [False], [False]
            def controls():
                for pending in store.query("SELECT * FROM actions WHERE handled_at IS NULL AND campaign_id=? ORDER BY id", (campaign_id,)):
                    if pending["kind"] in {"pause", "stop"}:
                        pause_requested[0] = pending["kind"] == "pause"
                        stop_requested[0] = pending["kind"] == "stop"
                        store.execute("UPDATE actions SET handled_at=? WHERE id=?", (now(), pending["id"]))
                return closing[0] or stop_requested[0]
            engine.run(campaign_id, controls=controls, pause=lambda: pause_requested[0])
            if closing[0] and not stop_requested[0]:
                store.set_status(campaign_id, "interrupted")
            if store.campaign(campaign_id)["status"] in {"waiting", "recovering"}:
                return
