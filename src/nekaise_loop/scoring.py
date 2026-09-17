"""Resolve immutable historical training pairs without loading model libraries."""
from pathlib import Path

from .artifacts import digest, verify_checkpoint, verified_snapshot


def historical_training_pair(ctx, round_id):
    row = ctx.store.one("SELECT * FROM rounds WHERE id=? AND status='complete'", (round_id,))
    if not row:
        raise ValueError("Scoring requires a completed historical round")
    keys, outputs = {}, {}
    for stage in ("draft", "freeze", "train"):
        saved = ctx.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage=? AND status='complete' ORDER BY attempt DESC LIMIT 1", (round_id, stage))
        if not saved:
            raise ValueError("Scoring requires immutable draft, freeze and train artifacts")
        keys[stage] = saved["artifact"]
        outputs[stage] = ctx.artifacts.get(saved["artifact"])
    trained, frozen = outputs["train"], outputs["freeze"]
    if (trained.get("trained") is False or trained["checkpoint"] != row["checkpoint"]
            or trained["manifest"].get("parent") != row["model_before"]
            or trained["manifest"].get("dataset_hash") != frozen["dataset_hash"]):
        raise ValueError("Scoring pair does not match immutable training provenance")
    # Freeze adds observations after hashing the prepared dataset.
    prepared = {k: frozen[k] for k in ("rows", "samples", "ledger")}
    if digest(prepared) != frozen["dataset_hash"]:
        raise ValueError("Scoring dataset identity differs from the consumed dataset")
    workspace_runs = ctx.engine.settings.workspace.resolve() / "runs"
    if not Path(trained["checkpoint"]).resolve().is_relative_to(workspace_runs):
        raise ValueError("Scoring output checkpoint must belong to this workspace")
    verify_checkpoint(trained, require_optimizer=False)
    parent = ctx.store.one("""SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id
        WHERE r.status='complete' AND r.checkpoint=? AND s.stage='train' AND s.status='complete'
        ORDER BY s.id DESC LIMIT 1""", (row["model_before"],))
    if parent:
        before = ctx.artifacts.get(parent["artifact"])
        if before["checkpoint"] != row["model_before"] or not Path(before["checkpoint"]).resolve().is_relative_to(workspace_runs):
            raise ValueError("Scoring parent checkpoint provenance differs")
        verify_checkpoint(before, require_optimizer=False)
        parent_identity = {"artifact": parent["artifact"], "files": before["manifest"]["files"]}
    else:
        parent_identity = verified_snapshot(row["model_before"])
    lessons = {lesson["id"]: lesson for lesson in outputs["draft"]["lessons"]}
    samples = []
    for index, sample in enumerate(frozen["samples"]):
        lesson = lessons.get(sample["row_id"]) if sample["stream"] == "teacher" else None
        prompt = lesson.get("prompt_token_ids") if lesson else None
        # Only classify an exact, untruncated prompt prefix. Chunked, merged or
        # legacy samples are still scored, but their loss split stays unknown.
        aligned = bool(prompt and not lesson.get("prompt_truncated")
                       and sample["input_ids"][:len(prompt)] == prompt)
        samples.append({**sample, "sample_index": index,
                        "prompt_tokens": len(prompt) if aligned else None})
    return {"round_id": round_id, "artifacts": keys, "dataset_hash": frozen["dataset_hash"],
            "before": row["model_before"], "after": row["checkpoint"],
            "parent_identity": parent_identity, "output_files": trained["manifest"]["files"],
            "samples": samples, "training": {k: trained["manifest"].get(k) for k in
                ("steps", "tokens", "global_step", "global_tokens", "optimizer_origin", "mean_loss", "config", "versions")}}
