"""Explicit Base restoration provenance; no model loading or checkpoint mutation."""
from .artifacts import verified_snapshot


def base_reference(store, artifacts, round_id):
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("Base restoration requires a completed historical round ID")
    row = store.one("SELECT checkpoint,model_before FROM rounds WHERE id=? AND status='complete'", (round_id,))
    stage = store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id,))
    if not row or not stage or not row["checkpoint"]:
        raise ValueError("Base restoration requires a completed round and training artifact")
    trained = artifacts.get(stage["artifact"])
    if (trained["checkpoint"] != row["checkpoint"] or trained.get("trained") is False
            or trained["manifest"].get("parent") != row["model_before"]):
        raise ValueError("Base restoration does not match immutable training parent provenance")
    return {"round_id": round_id, "artifact": stage["artifact"],
            "checkpoint": row["model_before"], "checkpoint_kind": "input",
            "identity": verified_snapshot(row["model_before"])}


def verify_restoration(store, artifacts, restoration):
    reference = restoration["reference"]
    if base_reference(store, artifacts, reference["round_id"]) != reference:
        raise ValueError("Base restoration provenance changed after branch selection")
