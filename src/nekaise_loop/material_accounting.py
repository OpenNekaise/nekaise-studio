"""Read-only material API usage accounting, independent of provider execution."""
import json

def job_work(store, round_id):
    if not store.one("SELECT name FROM sqlite_master WHERE type='table' AND name='material_jobs'"):
        return None
    jobs = store.query("SELECT status,COUNT(*) AS n FROM material_jobs WHERE round_id=? GROUP BY status", (round_id,))
    if not jobs:
        return None
    calls = store.query("SELECT c.status,c.reserved_tokens,c.usage FROM material_calls c JOIN material_jobs j ON j.id=c.job_id WHERE j.round_id=?", (round_id,))
    usage = [json.loads(c["usage"]) for c in calls]
    return {"jobs": {r["status"]: r["n"] for r in jobs}, "calls": len(calls),
            "reserved_output_tokens_all_attempts": sum(c["reserved_tokens"] for c in calls),
            "reported_input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "reported_output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "calls_without_usage": sum("prompt_tokens" not in u or "completion_tokens" not in u for u in usage),
            "limitations": "Provider tokens and reservations include retry work and are not student training targets. Missing usage may include a remote charge. Reasoning tokens are part of completion usage, not added twice."}
