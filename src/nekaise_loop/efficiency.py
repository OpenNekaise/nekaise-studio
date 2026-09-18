"""Measured student exposure per primary-teacher token; no learning-score gate."""
from .teacher_usage import call_usage

BASIS = "Actual optimizer target tokens across passes and attempts / primary teacher input plus output tokens. Cache and reasoning count once. Material Author and orchestrator tokens are excluded; author usage is reported separately. More exposure is not proof of learning."


def usage_summary():
    return {"input": 0, "output": 0, "cached": 0, "reported_calls": 0,
            "missing_calls": 0, "pending_calls": 0}


def add_call(summary, call, usage):
    if call["status"] == "running":
        summary["pending_calls"] += 1
    elif usage is None:
        summary["missing_calls"] += 1
    if usage is not None:
        summary["reported_calls"] += 1
        for key in ("input", "output", "cached"):
            summary[key] += usage[key]


def efficiency(trained, usage):
    total = usage["input"] + usage["output"]
    status = ("pending" if usage["pending_calls"] else "missing" if usage["missing_calls"]
              else "complete" if total > 0 else "unreported")
    reported = trained / total if total > 0 else None
    return {"trained_tokens": trained, "teacher_tokens": total if usage["reported_calls"] else None,
            "ratio": reported if status == "complete" else None, "reported_ratio": reported,
            "status": status, "teacher_usage": dict(usage), "basis": BASIS}


def round_efficiency(store, workspace, round_id, trained):
    summary = usage_summary()
    for call in store.query("SELECT * FROM teacher_calls WHERE round_id=? ORDER BY id", (round_id,)):
        value, _ = call_usage(store, workspace, call)
        add_call(summary, call, value)
    return efficiency(trained, summary)


def efficiency_history(rounds, usage_by_round, trained_by_round):
    points = []
    for row in rounds:
        if row["status"] != "complete":
            continue
        value = efficiency(trained_by_round.get(row["id"], 0), usage_by_round.get(row["id"], usage_summary()))
        points.append({"round_id": row["id"], "campaign_id": row["campaign_id"], "number": row["number"],
                       "iteration": len(points)+1, "at": row["updated_at"],
                       **{k: value[k] for k in ("ratio", "status", "trained_tokens", "teacher_tokens")}})
    return points
