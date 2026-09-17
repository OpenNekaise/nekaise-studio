"""Descriptive preparation and inference accounting, never a teaching decision."""
import math


def preparation_work(frozen, config):
    ledger = frozen.get("ledger", {})
    targets = ledger.get("total_tokens", 0)
    capacity = config["tokens_per_update"]
    passes = config["train_epochs"]
    per_pass = math.ceil(targets / capacity) if targets else 0
    steps = (config["train_steps"] or passes * per_pass) if passes and targets else 0
    cycles, remainder = divmod(steps, per_pass) if per_pass else (0, 0)
    exposure = cycles * targets + remainder * capacity
    return {"basis": "prepared_causal_targets_and_requested_execution_not_actual_training",
        "anchor_stream": ledger.get("anchor_stream"), "selected_rows": len(frozen.get("rows", [])),
        "prepared_samples": len(frozen.get("samples", [])), "targets_per_pass": targets,
        "streams": {name: {"available_targets": part["available_tokens"],
            "prepared_targets": part["prepared_tokens"],
            "unused_targets": max(0, part["available_tokens"] - part["prepared_tokens"]),
            "repeated_targets": part.get("repeated_tokens", 0)} for name, part in ledger.get("streams", {}).items()},
        "requested_passes": passes, "explicit_step_override": config["train_steps"],
        "expected_updates": steps, "expected_target_exposure": exposure,
        "update_target_capacity": capacity,
        "expected_mean_update_fill": exposure / (steps * capacity) if steps else None,
        "expected_short_updates": cycles if targets % capacity else 0}


def generation_work(outputs):
    """Count each shared batch timer once, including separate reference runs."""
    batches, rows, tokens, unavailable = {}, 0, 0, 0
    audit = {"tokens_checked": 0, "mismatched_tokens": 0, "nonfinite_answers": 0,
             "empty_answers": 0, "max_new_tokens_answers": 0}
    for stage, attempt, output in outputs:
        for row in output.get("lessons", output.get("items", [])):
            observations = [("current", row)]
            pair = row.get("comparison")
            if pair and pair.get("weights_changed", True):
                observations.append(("reference", pair["generation"]))
            for side, observation in observations:
                generation = observation if "generation_batch" in observation else observation.get("generation", {})
                batch = generation.get("generation_batch")
                if not batch:
                    unavailable += 1
                    continue
                rows += 1
                tokens += generation["tokens"]
                check = generation.get("generation_audit", {})
                audit["tokens_checked"] += check.get("tokens_checked", 0)
                audit["mismatched_tokens"] += len(check.get("mismatches", []))
                audit["nonfinite_answers"] += check.get("finite_logits") is False
                text = observation.get("text", observation.get("student"))
                audit["empty_answers"] += isinstance(text, str) and not text.strip()
                audit["max_new_tokens_answers"] += generation.get("stop_reason") == "max_new_tokens"
                key = (stage, attempt, side, batch["index"])
                if key in batches and batches[key] != batch:
                    raise ValueError("Inconsistent shared generation batch evidence")
                batches[key] = batch
    return {"basis": "completed_stage_attempts_only_shared_batch_time_counted_once",
        "answers": rows, "generated_tokens": tokens, "batches": len(batches),
        "max_batch_size": max((b["size"] for b in batches.values()), default=0),
        "generation_seconds": sum(b["generation_seconds"] for b in batches.values()),
        "audit": audit,
        "answers_without_batch_evidence": unavailable}
