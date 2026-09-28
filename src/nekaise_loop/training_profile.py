"""Standalone profiling worker. Owns ML children, never adds campaign exposure."""
import fcntl
import json
import os
from pathlib import Path
import signal

from .artifacts import canonical, digest, atomic_write, verify_checkpoint
from .config import ROOT
from .ownership import source_lock, source_fingerprint
from .processes import ProcessRunner, process_start
from .service import Service
from .storage import Store, now, new_id, encode


def profile_cases(*, equivalence_only=False, headroom=False):
    if equivalence_only:
        return [{"equivalence": True, "headroom": headroom,
                 **({"microbatch_size": 4, "activation_checkpointing": False} if headroom else {})}]
    if headroom:
        return [{"serial": False, "microbatch_size": size, "activation_checkpointing": False,
                 "headroom": True, "baseline_repeat": index == 3} for index, size in enumerate((4, 6, 8, 4))]
    return ([{"serial": True, "microbatch_size": 1, "activation_checkpointing": True}] +
            [{"serial": False, "microbatch_size": size, "activation_checkpointing": ac}
             for size in (1, 2, 4) for ac in (True, False)])


def run_profile(settings, round_id, *, mixed=False, warmup=8, steps=32, equivalence_only=False, headroom=False):
    """Exclusive worker ownership plus shared source lock, like the live worker."""
    if not 1 <= warmup <= 100 or not 1 <= steps <= 1000:
        raise ValueError("Profiling update count is outside its bounded range")
    with source_lock(), (settings.workspace / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        service = Service(settings)
        if service.store.one("SELECT id FROM campaigns WHERE status IN ('running','queued','pausing','stopping','waiting','recovering')"):
            raise ValueError("Pause or stop automatic execution before profiling")
        lock.seek(0); lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "start": process_start(os.getpid()), "kind": "training_profile"})); lock.flush()
        trained_stage = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id,))
        frozen_stage = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='freeze' AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id,))
        if not trained_stage or not frozen_stage:
            raise ValueError("Profiling requires a verified completed training round")
        trained = service.artifacts.get(trained_stage["artifact"])
        verify_checkpoint(trained)
        frozen = service.artifacts.get(frozen_stage["artifact"])
        if frozen["dataset_hash"] != trained["manifest"]["dataset_hash"]:
            raise ValueError("Frozen dataset and completed checkpoint disagree")
        config = {**trained["manifest"]["config"], "training_execution": "batched_v1", "inherit_optimizer": True}
        samples = [r for r in frozen["samples"] if mixed or r["stream"] == "corpus"]
        if not samples:
            raise ValueError("No selected real training samples")
        directory = settings.workspace / "profiles" / new_id("training")
        directory.mkdir(parents=True)
        db = Store(directory)  # separate diagnostics journal, never teaching history
        campaign, rid = "diagnostic", "profile"
        db.execute("INSERT INTO campaigns(id,name,status,config,created_at,updated_at) VALUES(?,?,'running',?,?,?)",
                   (campaign, "Throughput diagnostic; no official training exposure", encode(config), now(), now()))
        db.execute("INSERT INTO rounds(id,campaign_id,number,status,model_before,created_at,updated_at) VALUES(?,?,1,'running',?,?,?)",
                   (rid, campaign, trained["checkpoint"], now(), now()))
        evidence = {"round_id": round_id, "checkpoint_artifact": trained_stage["artifact"], "freeze_artifact": frozen_stage["artifact"],
                    "source_hash": source_fingerprint(), "sample_set": "mixed" if mixed else "raw_corpus",
                    "distinct_available_targets": sum(len(r["input_ids"])-1 for r in samples),
                    "coverage_advanced": False, "results": []}
        atomic_write(directory / "evidence.json", canonical(evidence))
        print(json.dumps({"profile_directory": str(directory)}), flush=True)
        cancelled = [False]
        signal.signal(signal.SIGTERM, lambda *_: cancelled.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: cancelled.__setitem__(0, True))
        def stopping():
            return cancelled[0] or bool(service.store.one("SELECT id FROM actions WHERE handled_at IS NULL"))
        cases = profile_cases(equivalence_only=equivalence_only, headroom=headroom)
        for index, variant in enumerate(cases):
            if stopping():
                raise RuntimeError("Profiling cancelled; pending operator commands take precedence")
            case = {**variant, "warmup_updates": warmup, "measured_updates": steps}
            payload = {"checkpoint": trained["checkpoint"], "dataset": {"samples": samples},
                "config": {**config, "training_microbatch_size": variant.get("microbatch_size", 1),
                           "training_activation_checkpointing": variant.get("activation_checkpointing", True)}, "profile_case": case}
            case_dir = directory / str(index); case_dir.mkdir()
            atomic_write(case_dir / "input.json", canonical(payload))
            db.execute("INSERT INTO stage_runs(round_id,stage,attempt,status,input_hash,started_at) VALUES(?, ?,1,'running',?,?)",
                       (rid, str(index), digest(payload), now()))
            stage = db.one("SELECT id FROM stage_runs WHERE stage=?", (str(index),))["id"]
            runner, received = ProcessRunner(db, stage, stopping), []
            def observe(message):
                if message["type"] == "result":
                    received.append(message["data"])
            try:
                runner.run([settings.model_python, "-u", str(ROOT / "src/nekaise_loop/workers/training_profile.py"), str(case_dir / "input.json")],
                    cwd=case_dir, log=case_dir / "process.log", timeout=1200, on_message=observe,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
                if len(received) != 1:
                    raise ValueError("Profiling worker did not return one result")
                result = received[0]
                db.execute("UPDATE stage_runs SET status='complete',finished_at=? WHERE id=?", (now(), stage))
            except Exception as exc:
                result = {"case": case, "error": str(exc)}
                db.execute("UPDATE stage_runs SET status='failed',finished_at=?,error=? WHERE id=?", (now(), str(exc), stage))
            evidence["results"].append(result)
            atomic_write(directory / "evidence.json", canonical(evidence))
            print(json.dumps({k: v for k, v in result.items() if k != "step_seconds"}), flush=True)
        # Saved checkpoint bytes must still verify after every diagnostic case.
        verify_checkpoint(trained)
        evidence["checkpoint_verified_after"] = True
        atomic_write(directory / "evidence.json", canonical(evidence))
        failed = any("error" in r or r.get("passed") is False for r in evidence["results"])
        db.execute("UPDATE campaigns SET status=?,updated_at=? WHERE id=?", ("failed" if failed else "complete", now(), campaign))
        db.execute("UPDATE rounds SET status=?,updated_at=? WHERE id=?", ("failed" if failed else "complete", now(), rid))
        if failed:
            raise RuntimeError(f"Profiling has failed cases; inspect {directory / 'evidence.json'}")
        return str(directory)
