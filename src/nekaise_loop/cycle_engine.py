"""Bounded CPU/Author producer and sequential, checkpoint-committing GPU consumer."""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timedelta, timezone

from . import stages, cycle_stages
from .artifacts import digest, verify_checkpoint
from .config import CampaignConfig, STAGES, STAGE_LABELS
from .cycle_store import open_cycle, cycle_for, bind_block, tokenizer_identity
from .engine import Context
from .failures import TeacherUnavailable
from .processes import Cancelled
from .storage import now

PREPARE = STAGES[:STAGES.index("train")]
ORDER = ("cycle_research", "cycle_plan", *STAGES)
LABELS = {**STAGE_LABELS, "cycle_research": "Research upcoming curriculum units",
          "cycle_plan": "Teacher block package and assessment"}
ROUTES = {"cycle_research": cycle_stages.research, "cycle_plan": cycle_stages.plan,
          **{s: getattr(cycle_stages, s) for s in ("select", "draft", "revise", "evaluate", "grade", "adapt")}}


def execute(engine, campaign, round_id, stage, cancelled):
    store, artifacts = engine.store, engine.artifacts
    if cancelled():
        raise Cancelled("Execution cancelled")
    row = store.one("SELECT * FROM rounds WHERE id=?", (round_id,))
    cycle = cycle_for(store, round_id)
    dependencies = store.query("SELECT stage,artifact FROM stage_runs WHERE round_id=? AND status='complete' ORDER BY id", (round_id,))
    inputs = {r["stage"]: r["artifact"] for r in dependencies if ORDER.index(r["stage"]) < ORDER.index(stage)}
    fingerprint = digest({"source": cycle["source_hash"], "contract": cycle["contract"],
        "round_id": round_id, "stage": stage, "dependencies": inputs,
        "plan": cycle["plan_artifact"] if stage in STAGES else None,
        "parent_binding": cycle["parent_binding_artifact"] if stage in STAGES[8:] else None})
    existing = store.one("SELECT * FROM stage_runs WHERE round_id=? AND stage=? AND status='complete' ORDER BY id DESC LIMIT 1", (round_id, stage))
    if existing:
        if existing["input_hash"] != fingerprint:
            raise ValueError(f"Completed {stage} inputs changed; preserve this block and create a continuation")
        result = artifacts.get(existing["artifact"])
        if stage == "train":
            verify_checkpoint(result)
        return result
    attempt = store.one("SELECT COALESCE(MAX(attempt),0)+1 AS n FROM stage_runs WHERE round_id=? AND stage=?", (round_id, stage))["n"]
    with store.connect(immediate=True) as db:
        db.execute("UPDATE rounds SET status=?,stage=?,error=NULL,updated_at=? WHERE id=?",
                   ("preparing" if stage in PREPARE else "running", stage, now(), round_id))
        stage_id = db.execute("INSERT INTO stage_runs(round_id,stage,attempt,status,input_hash,started_at) VALUES(?,?,?,'running',?,?)",
                              (round_id, stage, attempt, fingerprint, now())).lastrowid
        store.event(campaign["id"], round_id, "stage_started", LABELS[stage], {"stage": stage, "attempt": attempt}, db=db)
    row["stage"] = stage
    try:
        ctx = Context(engine, campaign, row, stage_id, attempt, cancelled)
        ctx.tokenization_checkpoint = cycle["anchor_checkpoint"]
        if stage == "freeze" and tokenizer_identity(ctx.tokenization_checkpoint) != cycle["contract"]["tokenizer"]:
            raise ValueError("Preparation tokenizer changed")
        result = (ROUTES[stage] if stage in ROUTES else getattr(stages, stage))(ctx)
        if stage == "freeze":
            result["preparation_basis"] = {"tokenizer_checkpoint": ctx.tokenization_checkpoint,
                "tokenizer": cycle["contract"]["tokenizer"], "weights_used": False,
                "training_parent": "Separate write-once binding before training"}
        if stage == "train":
            verify_checkpoint(result)
        key = artifacts.put(result)
        with store.connect(immediate=True) as db:
            db.execute("UPDATE stage_runs SET status='complete',artifact=?,finished_at=? WHERE id=?", (key, now(), stage_id))
            if stage == "train":
                from .curriculum_progress import commit_progress
                commit_progress(db, round_id, result, key)
                db.execute("UPDATE rounds SET checkpoint=? WHERE id=?", (result["checkpoint"], round_id))
            store.event(campaign["id"], round_id, "stage_complete", LABELS[stage]+" complete", {"stage": stage, "artifact": key}, db=db)
        return result
    except BaseException as exc:
        state = "interrupted" if isinstance(exc, Cancelled) else "waiting" if isinstance(exc, TeacherUnavailable) else "failed"
        store.execute("UPDATE stage_runs SET status=?,error=?,finished_at=? WHERE id=?", (state, str(exc)[-3000:], now(), stage_id))
        store.execute("UPDATE rounds SET status=?,error=?,updated_at=? WHERE id=?", (state, str(exc)[-3000:], now(), round_id))
        exc.cycle_stage_id = stage_id
        raise


class PauseCycle(Cancelled):
    pass


def run_cycle(engine, campaign, cycle, controls, pause):
    store = engine.store
    config = CampaignConfig.model_validate(campaign["config"])
    stop = threading.Event()

    def check():
        if controls():
            stop.set()
            raise Cancelled("Stopped by operator")
        if pause():
            stop.set()
            raise PauseCycle()

    def main_cancelled():
        # Only the consumer polls the worker's command queue.
        check()
        return stop.is_set()

    first = store.one("SELECT round_id FROM teaching_blocks WHERE cycle_id=? AND position=0", (cycle["id"],))["round_id"]
    execute(engine, campaign, first, "cycle_research", main_cancelled)
    execute(engine, campaign, first, "cycle_plan", main_cancelled)
    blocks = store.query("SELECT r.* FROM rounds r JOIN teaching_blocks b ON b.round_id=r.id WHERE b.cycle_id=? ORDER BY b.position", (cycle["id"],))
    pending = [b for b in blocks if b["status"] != "complete"]
    ready, failure = queue.Queue(), []
    capacity = threading.Semaphore(config.teaching_cycle.prefetch_blocks)

    def produce():
        try:
            for block in pending:
                while not capacity.acquire(timeout=.1):
                    if stop.is_set():
                        return
                if stop.is_set():
                    return
                for stage in PREPARE:
                    execute(engine, campaign, block["id"], stage, stop.is_set)
                # Do not erase a saved train or failed assessment's status on retry.
                if not store.one("SELECT id FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete'", (block["id"],)):
                    store.execute("UPDATE rounds SET status='prepared',updated_at=? WHERE id=?", (now(), block["id"]))
                ready.put(block["id"])
        except BaseException as exc:
            failure.append(exc)
        finally:
            ready.put(None)

    producer = threading.Thread(target=produce, name="curriculum-preparation")
    producer.start()
    reflection = None
    try:
        for block in pending:
            while True:
                check()
                try:
                    round_id = ready.get(timeout=.1)
                    break
                except queue.Empty:
                    continue
            if round_id is None:
                raise failure[0] if failure else RuntimeError("Preparation ended without the planned block")
            if round_id != block["id"]:
                raise ValueError("Preparation queue is out of order")
            if not store.one("SELECT id FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete'", (round_id,)):
                row = store.one("SELECT * FROM rounds WHERE id=?", (round_id,))
                row["stage"] = "train"
                ctx = Context(engine, campaign, row, None, 0, main_cancelled)
                bind_block(ctx, cycle_for(store, round_id), ctx.output("freeze"))
            capacity.release()  # At most two preparing/prepared future blocks.
            from .checkpoint_retention import storage_status
            parent = store.one("SELECT model_before FROM rounds WHERE id=?", (round_id,))["model_before"]
            storage = storage_status(engine.settings.workspace, parent)
            if storage["pressure"]:
                raise RuntimeError(f"Insufficient checkpoint headroom: {storage['free_bytes']} bytes free, {storage['required_bytes']} required; orchestrator must inspect storage")
            for stage in STAGES[8:]:
                reflection = execute(engine, campaign, round_id, stage, main_cancelled)
            store.execute("UPDATE rounds SET status='complete',updated_at=? WHERE id=?", (now(), round_id))
            store.event(campaign["id"], round_id, "round_complete", f"Block {block['number']} saved; " + ("cycle review pending" if reflection.get("review_pending") else "cycle reviewed"))
            check()
            # Preserve a saved current block before handing a future failure to recovery.
            if failure:
                raise failure[0]
        if reflection is None:
            last = store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='adapt' AND status='complete' ORDER BY id DESC LIMIT 1", (blocks[-1]["id"],))
            reflection = engine.artifacts.get(last["artifact"])
        if reflection.get("review_pending"):
            raise ValueError("Cycle cannot complete without an observed Teacher review")
        key = engine.artifacts.put(reflection)
        store.execute("UPDATE teaching_cycles SET status='complete',review_artifact=?,updated_at=? WHERE id=?", (key, now(), cycle["id"]))
        return reflection
    finally:
        stop.set()
        producer.join()


def run(engine, campaign_id, controls, pause):
    store = engine.store
    campaign = store.campaign(campaign_id)
    config = CampaignConfig.model_validate(campaign["config"])
    store.set_status(campaign_id, "running")
    try:
        while True:
            if controls():
                raise Cancelled("Stopped by operator")
            if pause():
                raise PauseCycle()
            cycle = open_cycle(engine, campaign)
            if cycle is None:
                break
            reflection = run_cycle(engine, campaign, cycle, controls, pause)
            if reflection["action"] == "pause":
                from .service import Service
                Service(engine.settings).action(campaign_id, "pause", spawn=False, actor="teacher", reason=reflection["reason"])
                return
            if reflection["action"] == "complete":
                break
            if config.auto_recover and config.manage_history:
                from .history import review_due
                if review_due(store):
                    store.recover(campaign_id, "history_review", "Scheduled orchestrator review at the completed cycle boundary")
                    return
        store.set_status(campaign_id, "complete")
    except (Cancelled, PauseCycle) as exc:
        store.set_status(campaign_id, "paused" if isinstance(exc, PauseCycle) else "stopped")
    except Exception as exc:
        if controls() or pause():
            store.set_status(campaign_id, "paused" if pause() else "stopped")
            return
        unavailable = isinstance(exc, TeacherUnavailable)
        if config.auto_recover:
            retry_at = None
            if unavailable and exc.kind not in {"budget", "material_budget"}:
                delay = exc.retry_seconds or (60 if exc.kind == "rate_limit" else config.teacher_retry_seconds)
                retry_at = (datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat(timespec="milliseconds")
            store.recover(campaign_id, exc.kind if unavailable else "failure", exc,
                          retry_at=retry_at, failed_stage_id=getattr(exc, "cycle_stage_id", None))
        else:
            store.set_status(campaign_id, "waiting" if unavailable else "failed", str(exc)[-3000:])
        store.event(campaign_id, None, "waiting" if unavailable else "error", str(exc)[-1500:])
