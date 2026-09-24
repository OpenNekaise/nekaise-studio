"""Read-only accounting for teacher and orchestrator decisions, never a gate."""
from datetime import datetime
import json
from pathlib import Path
from .work_accounting import generation_work, preparation_work
from .material_accounting import author_yield, job_work
from .efficiency import round_efficiency

def prepared_coverage(frozen):
    """Split only exact known prefixes; report preparation, not consumption."""
    prefixes = {}
    for row in frozen.get("rows", []):
        ids = row.get("serialization", {}).get("prompt_token_ids")
        if ids:
            stream = "teacher" if row.get("stream") in {"cpt", "sft"} else row.get("stream")
            prefixes[stream, row["id"]] = ids
    observations = frozen.get("prompt_training_prefixes", {}).get("lessons", [])
    aligned = {c["sample_index"]: o["prompt_tokens"] for o in observations
               if not o.get("prompt_truncated") for c in o["sample_comparisons"]
               if c["prompt_is_sample_prefix"]}
    counts = {"prompt": 0, "continuation": 0, "unassigned": 0}
    for index, sample in enumerate(frozen.get("samples", [])):
        tokens = sample["input_ids"]
        prefix = prefixes.get((sample.get("stream"), sample["row_id"]))
        length = len(prefix) if prefix and tokens[:len(prefix)] == prefix else aligned.get(index)
        if length is None:
            counts["unassigned"] += len(tokens) - 1
        else:
            counts["prompt"] += max(0, length - 1)
            counts["continuation"] += len(tokens) - length
    return {"basis": "prepared_causal_targets_per_pass_not_consumed_exposure", **counts}


def round_work(store, artifacts, round_id):
    row = store.one("SELECT id,campaign_id,number,status FROM rounds WHERE id=?", (round_id,))
    if not row:
        return None
    stages = store.query("SELECT stage,attempt,status,started_at,finished_at,artifact FROM stage_runs WHERE round_id=? ORDER BY id", (round_id,))
    wall, missing, outputs, generations = {}, [], {}, []
    for stage in stages:
        if stage["started_at"] and stage["finished_at"]:
            seconds = (datetime.fromisoformat(stage["finished_at"]) - datetime.fromisoformat(stage["started_at"])).total_seconds()
            wall[stage["stage"]] = wall.get(stage["stage"], 0) + max(0, seconds)
        else:
            missing.append({"stage": stage["stage"], "attempt": stage["attempt"], "status": stage["status"]})
        if stage["status"] == "complete" and stage["stage"] in {"freeze", "train", "select", "material_select"}:
            outputs[stage["stage"]] = artifacts.get(stage["artifact"])
        if stage["status"] == "complete" and stage["stage"] in {"draft", "answer"}:
            generations.append((stage["stage"], stage["attempt"], artifacts.get(stage["artifact"])))
    attempts = {}
    previous, update_sizes = {}, []
    for metric in store.query("SELECT attempt,step,data FROM metrics WHERE round_id=? ORDER BY attempt,step", (round_id,)):
        value = json.loads(metric["data"])
        prior_step, prior_tokens = previous.get(metric["attempt"], (0, 0))
        if isinstance(value.get("tokens"), (int, float)):
            if metric["step"] == prior_step + 1 and value["tokens"] >= prior_tokens:
                update_sizes.append(value["tokens"] - prior_tokens)
            previous[metric["attempt"]] = (metric["step"], value["tokens"])
        total = attempts.setdefault(metric["attempt"], {"updates": 0, "tokens": 0, "elapsed_seconds": 0})
        total["updates"] += 1
        for key in ("tokens", "elapsed_seconds"):
            if isinstance(value.get(key), (int, float)):
                total[key] = max(total[key], value[key])
    trained, frozen = outputs.get("train", {}), outputs.get("freeze", {})
    manifest = trained.get("manifest", {})
    checkpoint = Path(trained["checkpoint"]) if trained else None
    present_bytes = 0 if checkpoint else None
    for name in manifest.get("files", {}):
        if Path(name).name != name:
            continue
        try:
            present_bytes += (checkpoint / name).stat().st_size
        except FileNotFoundError:
            pass  # Retention can remove historical bytes during this read.
    diagnostic = trained.get("trained") is False
    selected = outputs.get("material_select", outputs.get("select", {})).get("curriculum", {})
    campaign = store.one("SELECT config FROM campaigns WHERE id=?", (row["campaign_id"],))
    config = json.loads(campaign["config"])
    capacity = config["tokens_per_update"]
    preparation = (frozen.get("preparation_work") or preparation_work(frozen,
        {**config, "train_epochs": selected.get("train_epochs", config["train_epochs"])})) if frozen else None
    return {**row, "expansion_policy": config.get("expansion_policy", "legacy_optional"), "stage_seconds": wall, "finished_stage_seconds": sum(wall.values()),
            "timing_complete": not missing,
            "teacher_stage_seconds": sum(wall.get(s, 0) for s in ("select", "revise", "evaluate", "grade", "adapt")) + (wall.get("material_select", 0) if outputs.get("material_select", {}).get("selection") is not None and outputs["material_select"].get("review_policy") != "trusted_author_v1" else 0),
            "unfinished_stage_timings": missing,
            "measured_work_all_attempts": {k: sum(a[k] for a in attempts.values()) for k in ("updates", "tokens", "elapsed_seconds")},
            "retained_training": {"trained": not diagnostic if trained else None,
                "updates": 0 if diagnostic else manifest.get("steps"),
                "tokens": 0 if diagnostic else manifest.get("tokens"),
                "optimizer_origin": manifest.get("optimizer_origin") if not diagnostic else None},
            "material_portfolio": {"preparation": frozen.get("material_portfolio"), "completed_training": trained.get("material_portfolio")},
            "prepared_targets_per_pass": frozen.get("ledger", {}).get("total_tokens"),
            "prepared_target_coverage": prepared_coverage(frozen), "requested_passes": selected.get("train_epochs"),
            "teacher_work_plan": selected.get("work_plan"), "preparation": preparation,
            "update_work": {"target_capacity": capacity, "updates_with_known_size": len(update_sizes),
                "short_updates": sum(n < capacity for n in update_sizes),
                "mean_targets": sum(update_sizes)/len(update_sizes) if update_sizes else None,
                "mean_fill": sum(update_sizes)/(len(update_sizes)*capacity) if update_sizes else None},
            "generation_work": generation_work(generations),
            "material_author_work": job_work(store, round_id), "material_sources": frozen.get("material_sources"),
            "author_yield": author_yield(store, artifacts, round_id, frozen),
            "material_expansion": frozen.get("material_expansion"),
            "teacher_efficiency": {**round_efficiency(store, artifacts.root.parent, round_id, sum(a["tokens"] for a in attempts.values())),
                                   "final": row["status"] == "complete"},
            "checkpoint_bytes_present": present_bytes,
            "limitations": "Measured work includes retry attempts, not inherited global counters. Inner optimizer elapsed excludes model loading, inference and saving. Stage sums include finished attempts only, exclude between-round reviews and are not GPU utilization. Prepared coverage is not consumed exposure; continuation includes termination tokens. Present checkpoint bytes may reflect retention. Counts and scores do not establish learning."}


def safe_round_work(store, artifacts, round_id):
    try:
        return round_work(store, artifacts, round_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Accounting is descriptive. An unavailable observation must not block
        # the teacher or the operations agent that can investigate its cause.
        return {"id": round_id, "error": f"Learning-work evidence unavailable: {exc}"[:600]}


def latest_work(store, artifacts, campaign_id):
    visited = set()
    while campaign_id and campaign_id not in visited:
        visited.add(campaign_id)
        row = store.one("SELECT id FROM rounds WHERE campaign_id=? AND status='complete' ORDER BY number DESC LIMIT 1", (campaign_id,))
        if row:
            return safe_round_work(store, artifacts, row["id"])
        campaign = store.one("SELECT parent_campaign_id FROM campaigns WHERE id=?", (campaign_id,))
        campaign_id = campaign["parent_campaign_id"] if campaign else None
    return None
