"""Same-question online comparisons, without teaching or acceptance decisions."""
from pathlib import Path
import random

from .artifacts import digest, verify_checkpoint, verified_snapshot
from .teaching import Grade


def evaluation_pair(ctx):
    """Freeze both identities after training and recheck them before inference."""
    stage = ctx.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train' AND status='complete' ORDER BY attempt DESC LIMIT 1", (ctx.round["id"],))
    if not stage:
        raise ValueError("Online comparison requires a completed training stage")
    trained = ctx.artifacts.get(stage["artifact"])
    before, after = ctx.round["model_before"], trained["checkpoint"]
    recorded = ctx.store.one("SELECT checkpoint FROM rounds WHERE id=?", (ctx.round["id"],))
    if not recorded or recorded["checkpoint"] != after:
        raise ValueError("Online comparison output differs from its recorded checkpoint")
    unchanged = trained.get("trained") is False
    if (unchanged and before != after) or (not unchanged and
            (trained["manifest"].get("parent") != before or before == after)):
        raise ValueError("Online comparison differs from immutable training parent provenance")
    if not unchanged and not Path(after).resolve().is_relative_to(ctx.engine.settings.workspace.resolve() / "runs"):
        raise ValueError("Online comparison output must belong to this workspace")
    verify_checkpoint(trained, require_optimizer=False)
    if unchanged:
        identity = {"kind": "unchanged_checkpoint", "files": trained["manifest"]["files"]}
    elif Path(before).resolve().is_relative_to(ctx.engine.settings.workspace.resolve() / "runs"):
        # Continuations retain completed training even if a later assessment or
        # reflection failed. Resolve the weight-producing stage, independently
        # of the enclosing round's status, and still verify its immutable files.
        parent = ctx.store.one("""SELECT s.artifact FROM stage_runs s JOIN rounds r ON r.id=s.round_id
            WHERE r.checkpoint=? AND s.stage='train' AND s.status='complete'
            AND r.model_before!=r.checkpoint
            ORDER BY s.id DESC LIMIT 1""", (before,))
        if not parent:
            raise ValueError("Online comparison requires a completed parent training artifact")
        saved = ctx.artifacts.get(parent["artifact"])
        if saved["checkpoint"] != before:
            raise ValueError("Online comparison parent checkpoint provenance differs")
        verify_checkpoint(saved, require_optimizer=False)
        identity = {"kind": "workspace_checkpoint", "artifact": parent["artifact"], "files": saved["manifest"]["files"]}
    else:
        identity = verified_snapshot(before)
    return {"before": before, "after": after, "train_artifact": stage["artifact"],
            "before_identity": identity, "after_files": trained["manifest"]["files"],
            "weights_changed": not unchanged}


def comparable_generations(after, before):
    """Compare recorded execution conditions, never the quality of the answers."""
    differences = []
    keys = ["prompt_token_ids", "generation_settings", "runtime"]
    if "generation_execution" in after or "generation_execution" in before:
        keys.append("generation_execution")
    for key in keys:
        a, b = after.get(key), before.get(key)
        if not a or not b:
            differences.append(f"{key}_unavailable")
            continue
        if key == "generation_settings":
            # Transformers metadata does not change decoding behavior.
            a = {k: v for k, v in a.items() if not k.startswith("_") and k != "transformers_version"}
            b = {k: v for k, v in b.items() if not k.startswith("_") and k != "transformers_version"}
            if a.get("do_sample") is not False or b.get("do_sample") is not False:
                differences.append("decoding_is_not_greedy")
        if a != b:
            differences.append(f"{key}_differ")
    return {"comparable": not differences, "differences": differences,
            "basis": "same_recorded_prompt_tokens_decoding_runtime_and_available_batch_context"}


def grade_requests(items, round_id):
    """Hide checkpoint labels and shuffle paired answers in one grading call.

    This removes explicit before/after labels, not all possible clues from the
    teacher's memory. References and criteria are identical on both sides.
    """
    requests, mapping = [], {}
    for item in items:
        pair = item.get("comparison")
        sides = ("after", "before") if pair and pair["weights_changed"] else ("after",)
        for side in sides:
            key = digest({"round": round_id, "item": item["id"], "side": side})[:24]
            mapping[key] = (item["id"], side)
            requests.append({"id": key, **{k: item[k] for k in
                ("question", "student_prompt", "reference", "rubric", "concept", "dimensions",
                 "evidence", "sources", "document_id", "source_title", "source_url") if k in item},
                "student": item["student"] if side == "after" else pair["generation"]["text"]})
    random.Random(round_id).shuffle(requests)
    return requests, mapping


def apply_grades(items, responses, mapping):
    grades = [Grade.model_validate(g).model_dump() for g in responses]
    if len(grades) != len(mapping) or {g["id"] for g in grades} != set(mapping):
        raise ValueError("Missing evaluation grades")
    rows = {r["id"]: r for r in items}
    for grade in grades:
        item_id, side = mapping[grade["id"]]
        row = rows[item_id]
        expected = {d["name"] for d in row.get("dimensions", [])}
        names = [d["name"] for d in grade["dimensions"]]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Grades must match the dimensions frozen before answers")
        grade["id"] = item_id
        if side == "before":
            row["comparison"]["grade"] = grade
        else:
            row["grade"] = grade
    for row in items:
        pair = row.get("comparison")
        if not pair:
            continue
        pair["grading"] = "checkpoint_labels_hidden_in_one_shuffled_call" if pair["weights_changed"] else "same_recorded_answer_and_grade_reused_no_update"
        if not pair["weights_changed"]:
            pair["grade"] = dict(row["grade"])
        pair["score_delta"] = (row["grade"]["score"] - pair["grade"]["score"]
                               if pair["weights_changed"] and pair["conditions"]["comparable"] else None)
    return items
