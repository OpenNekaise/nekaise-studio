"""Small default Teacher requests with exact, unrestricted archive retrieval."""
from .artifacts import digest


def compact_inputs(inputs):
    def ref(value, pointer):
        return {"$evidence": {"op": "request_data", "pointer": pointer,
                              "canonical_sha256": digest(value)}}

    config = inputs["config_hints"]
    strategy = inputs["latest_strategy"]
    operations = inputs["operations"]
    task = inputs["task"]
    view = {
        "current": inputs["current"],
        "config_hints": {k: config[k] for k in (
            "student_format", "student_identity", "teacher_model", "expansion_policy",
            "general_material_policy", "material_review_policy", "teaching_cycle",
            "curriculum_loop", "max_seq_len", "max_new_tokens", "workload_guidance")},
        "complete_config": ref(config, "/config_hints"),
        "latest_strategy": {k: strategy[k] for k in (
            "student_notes", "next_round_instructions", "action", "reason", "gaps") if k in strategy},
        "complete_strategy": ref(strategy, "/latest_strategy"),
        "operations": {"operator_hold": operations.get("operator_hold"),
                       "latest_action": operations.get("latest_action"),
                       "complete": ref(operations, "/operations")},
        "task": dict(task),
    }
    review = operations.get("latest_applied_review")
    if review:
        view["operations"]["latest_applied_review"] = {
            "id": review["id"], "status": review["status"],
            "action": review["decision"].get("action"), "reason": review["decision"].get("reason"),
            "complete": {"op": "report", "recovery_id": review["id"]}}
    if "research" in task:
        compact = []
        for i, unit in enumerate(task["research"]):
            sources = []
            for j, source in enumerate(unit["sources"]):
                sources.append({**{k: source[k] for k in ("id", "title", "url", "purpose", "source_sha256", "span_start", "span_length") if k in source},
                    "preview": source["text"][:1600],
                    "complete_text": ref(source["text"], f"/task/research/{i}/sources/{j}/text")})
            compact.append({"unit_id": unit["unit_id"], "sources": sources,
                            "fetch_failures": unit.get("fetch_failures", [])})
        view["task"]["research"] = compact
    return view
