"""Teacher-declared material scope and exact target accounting, not content grading."""
from collections import Counter
from typing import Literal

from .training import update_batches

MaterialScope = Literal["general_chat", "general_prose", "domain", "unspecified"]
SCOPES = ("general_chat", "general_prose", "domain", "unspecified")


def _origin(row):
    if row["stream"] in {"corpus", "replay"}:
        return row["stream"]
    return "author" if row.get("material_origin") else "teacher"


def portfolio(frozen, config, planned_shares=None):
    """Count prepared targets and the deterministic requested optimizer traversal."""
    rows = {row["id"]: row for row in frozen.get("rows", [])}
    if len(rows) != len(frozen.get("rows", [])):
        raise ValueError("Material portfolio requires unique frozen row IDs")
    for row in rows.values():
        scope = row.get("material_scope", "unspecified")
        if scope not in SCOPES:
            raise ValueError("Unknown declared material scope")
        if scope == "general_chat" and (row.get("training_tokenization") != "chat_response"
                                         or not row.get("training_response", "").strip()):
            raise ValueError("General chat requires native chat_response with a nonempty answer")

    def count(samples):
        scopes, origins = Counter(), Counter()
        for sample in samples:
            row = rows[sample["row_id"]]
            scope = row.get("material_scope", "unspecified")
            n = len(sample["input_ids"]) - 1
            scopes[scope] += n
            origins[(scope, _origin(row))] += n
        return scopes, origins

    prepared, prepared_origins = count(frozen.get("samples", []))
    expected, expected_origins, steps = Counter(), Counter(), 0
    if config["train_epochs"] > 0 and frozen.get("samples"):
        for batch in update_batches(frozen.get("samples", []), config["tokens_per_update"],
                                    config["train_epochs"], config["seed"], config["train_steps"]):
            scopes, origins = count(batch)
            expected.update(scopes)
            expected_origins.update(origins)
            steps += 1
    total = sum(prepared.values())
    if total != frozen.get("ledger", {}).get("total_tokens", total):
        raise ValueError("Material portfolio differs from frozen token ledger")
    result = {"policy": config.get("general_material_policy", "legacy_optional"),
              "basis": "teacher_declared_scope; causal_target_occurrences_including_prompt_and_eos",
              "planned_shares": planned_shares,
              "prepared_targets": total,
              "by_scope": {scope: {"prepared_targets": prepared[scope],
                  "prepared_share": prepared[scope] / total if total else None,
                  "requested_exposure": expected[scope],
                  "prepared_by_origin": {origin: prepared_origins[(scope, origin)] for origin in ("teacher", "author", "corpus", "replay")},
                  "requested_by_origin": {origin: expected_origins[(scope, origin)] for origin in ("teacher", "author", "corpus", "replay")}}
                  for scope in SCOPES},
              "requested_updates": steps, "requested_exposure": sum(expected.values()),
              "limitations": "Scope is a Teacher declaration, not semantic verification. Unknown historical material stays unspecified. Repetition is exposure, not distinct coverage; requested exposure is not completed training."}
    if config.get("general_material_policy", "legacy_optional") == "required_v1" and config["train_epochs"] > 0:
        if not expected["general_chat"]:
            raise ValueError("Required general material has zero general-chat targets in the requested optimizer traversal")
        for origin in ("teacher", "author"):
            if not sum(expected_origins[(scope, origin)] for scope in ("general_chat", "general_prose")):
                raise ValueError(f"Required general material has zero new {origin} general targets in the requested optimizer traversal")
    return result


def completed_portfolio(receipt, manifest):
    """Only completed, matching trainer counters substantiate consumed exposure."""
    if manifest.get("tokens") is None or manifest.get("steps") is None:
        return {"status": "unverified", "reason": "Completed trainer counters unavailable"}
    if (manifest["tokens"] != receipt["requested_exposure"]
            or manifest["steps"] != receipt["requested_updates"]):
        return {"status": "mismatch", "reason": "Completed trainer counters differ from the material portfolio traversal",
                "observed": {"tokens": manifest["tokens"], "updates": manifest["steps"]},
                "expected": {"tokens": receipt["requested_exposure"], "updates": receipt["requested_updates"]}}
    return {"status": "verified", "basis": "reconstructed_frozen_traversal_matched_to_completed_trainer_counters",
            "tokens": manifest["tokens"], "updates": manifest["steps"],
            "by_scope": {scope: part["requested_exposure"] for scope, part in receipt["by_scope"].items()}}
