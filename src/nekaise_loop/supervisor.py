"""Small persistent scheduler; agent reasoning happens only on recovery incidents."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback

from .config import CampaignConfig
from .ownership import locked, source_fingerprint
from .processes import process_start
from .recovery import defer, later, operator_cancelled
from .service import Service, Conflict
from .storage import now


def tick(service):
    settings, store = service.settings, service.store
    if service.worker_alive() or locked(settings.workspace/"recovery.lock"):
        return None
    if store.one("SELECT id FROM actions WHERE handled_at IS NULL LIMIT 1") or store.one("SELECT id FROM campaigns WHERE status IN ('running','queued','pausing','stopping') LIMIT 1"):
        return ["worker"]
    row = store.one("SELECT * FROM recoveries WHERE status IN ('pending','waiting','running','decided') ORDER BY id LIMIT 1")
    if not row:
        return None
    campaign = store.campaign(row["campaign_id"])
    if operator_cancelled(store, campaign["id"]):
        store.execute("UPDATE recoveries SET status='cancelled',updated_at=? WHERE id=?", (now(), row["id"]))
        return None
    config = CampaignConfig.model_validate(campaign["config"])
    if not config.auto_recover:
        return None
    if row["status"] == "decided":
        return ["apply-recovery", str(row["id"])]
    if row["kind"] in {"quota", "rate_limit", "budget"}:
        if row["status"] == "pending":
            store.execute("UPDATE recoveries SET status='waiting',decision=?,updated_at=? WHERE id=?", (json.dumps({"action": "wait", "actor": "supervisor", "reason": row["error"]}), now(), row["id"]))
            return None
        if row["retry_at"] and row["retry_at"] <= now():
            try:
                service.action(campaign["id"], "resume", spawn=False)
            except Conflict:
                return None
            return ["worker"]
        return None
    if row["status"] == "waiting" and (not row["retry_at"] or row["retry_at"] > now()):
        return None
    return ["recover", str(row["id"])]


def run_supervisor(settings):
    with (settings.workspace/"supervisor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "start": process_start(os.getpid())}))
        lock.flush()
        closing = [False]
        signal.signal(signal.SIGTERM, lambda *_: closing.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: closing.__setitem__(0, True))
        service, child, fingerprint = Service(settings), None, source_fingerprint()
        while not closing[0]:
            if child is not None and child.poll() is None:
                time.sleep(0.5)
                continue
            child = None
            if source_fingerprint() != fingerprint:
                # Every repair is followed by a fresh interpreter, including this
                # scheduler. Durable records and command queues survive the exec.
                os.execv(sys.executable, [sys.executable, "-m", "nekaise_loop.cli", "--workspace", str(settings.workspace), "supervisor"])
            try:
                command = tick(service)
                if command:
                    with (settings.workspace/("worker.log" if command[0] == "worker" else "orchestrator.log")).open("a") as log:
                        child = subprocess.Popen([sys.executable, "-m", "nekaise_loop.cli", "--workspace", str(settings.workspace), *command], cwd=settings.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            except Exception:
                traceback.print_exc()
                time.sleep(5)
            time.sleep(0.5)
