"""Read-only material API usage accounting, independent of provider execution."""
import json
from collections import Counter
from .artifacts import digest


def author_yield(store, artifacts, round_id, frozen):
    """Observed sizing evidence for teacher planning, never a required dose.

    Successful job yield and all-attempt cost have different denominators. Only
    jobs with reported completion usage contribute to the output/candidate ratio.
    Prepared targets may contain repeated occurrences and are not optimizer work.
    """
    if not store.one("SELECT name FROM sqlite_master WHERE type='table' AND name='material_jobs'"):
        return None
    groups, jobs = {}, []
    reservations = {call["id"]: call["reserved_tokens"] for call in store.query(
        "SELECT c.id,c.reserved_tokens FROM material_calls c JOIN material_jobs j ON j.id=c.job_id WHERE j.round_id=?", (round_id,))}
    for job in store.query("SELECT id,plan_id,author_id,artifact FROM material_jobs WHERE round_id=? AND status='complete' ORDER BY id", (round_id,)):
        result = artifacts.get(job["artifact"])
        group = groups.setdefault(job["author_id"], {"completed_jobs": 0, "empty_completed_jobs": 0, "produced_candidates": 0,
            "jobs_with_output_usage": 0, "jobs_without_output_usage": 0,
            "reported_output_tokens": 0, "candidates_with_output_usage": 0,
            "max_reported_output_tokens_per_job": None})
        count = len(result["rows"])
        group["completed_jobs"] += 1
        group["empty_completed_jobs"] += count == 0
        group["produced_candidates"] += count
        output = (result.get("usage") or {}).get("completion_tokens")
        if isinstance(output, int) and not isinstance(output, bool) and output >= 0:
            group["jobs_with_output_usage"] += 1
            group["reported_output_tokens"] += output
            group["candidates_with_output_usage"] += count
            group["max_reported_output_tokens_per_job"] = max(group["max_reported_output_tokens_per_job"] or 0, output)
        else:
            group["jobs_without_output_usage"] += 1
            output = None
        jobs.append({"job_id": job["id"], "plan_id": job["plan_id"], "author_id": job["author_id"],
            "result_artifact": job["artifact"], "call_id": result.get("call_id"),
            "expected_items": result.get("expected_items"), "produced_candidates": count,
            "reserved_output_tokens": reservations.get(result.get("call_id")),
            "reported_output_tokens": output,
            "reported_output_tokens_per_candidate": output/count if output is not None and count else None})
    rows = {r["id"]: r for r in frozen.get("rows", []) if r.get("material_origin") and r.get("stream") in {"cpt", "sft"}}
    targets = Counter()
    for sample in frozen.get("samples", []):
        row = rows.get(sample["row_id"])
        if row and sample["stream"] == "teacher":
            targets[row["material_origin"]["author_id"]] += len(sample["input_ids"])-1
    for author, group in groups.items():
        selected = [r for r in rows.values() if r["material_origin"]["author_id"] == author]
        group.update(selected_candidates=len(selected),
            selected_exact_content_variants=len({digest({k: r.get(k) for k in ("text", "training_prompt", "training_response", "training_tokenization")}) for r in selected}),
            prepared_targets_per_pass=targets[author],
            prepared_targets_per_selected_candidate=targets[author]/len(selected) if selected else None,
            reported_output_tokens_per_candidate=group["reported_output_tokens"]/group["candidates_with_output_usage"] if group["candidates_with_output_usage"] else None)
        if not group["jobs_with_output_usage"]:
            group["reported_output_tokens"] = None
    return {"round_id": round_id, "by_author": groups, "by_job": jobs,
            "basis": "Completed-job output/candidate sizing includes empty completed job overhead; by_job reservations belong only to the successful call, not all attempts. Job costs cover the whole response, including prompts, metadata and reasoning; an author average does not bound a different job. Prepared selected target occurrences are per pass. Failed/retried call cost is in material_author_work. Missing usage is not zero; exact content variants are not semantic coverage. Provider tokens are not student-tokenizer targets; preparation is not actual consumption."}

def job_work(store, round_id):
    if not store.one("SELECT name FROM sqlite_master WHERE type='table' AND name='material_jobs'"):
        return None
    jobs = store.query("SELECT status,COUNT(*) AS n FROM material_jobs WHERE round_id=? GROUP BY status", (round_id,))
    if not jobs:
        return None
    calls = store.query("SELECT j.author_id,c.status,c.reserved_tokens,c.usage FROM material_calls c JOIN material_jobs j ON j.id=c.job_id WHERE j.round_id=?", (round_id,))
    usage = [json.loads(c["usage"]) for c in calls]
    by_author = {}
    for call, reported in zip(calls, usage):
        group = by_author.setdefault(call["author_id"], {"calls": 0, "reserved_output_tokens": 0,
            "reported_input_tokens": 0, "reported_output_tokens": 0, "calls_without_usage": 0})
        group["calls"] += 1
        group["reserved_output_tokens"] += call["reserved_tokens"]
        group["reported_input_tokens"] += reported.get("prompt_tokens", 0)
        group["reported_output_tokens"] += reported.get("completion_tokens", 0)
        group["calls_without_usage"] += "prompt_tokens" not in reported or "completion_tokens" not in reported
    return {"jobs": {r["status"]: r["n"] for r in jobs}, "calls": len(calls),
            "reserved_output_tokens_all_attempts": sum(c["reserved_tokens"] for c in calls),
            "reported_input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "reported_output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "calls_without_usage": sum("prompt_tokens" not in u or "completion_tokens" not in u for u in usage),
            "by_author": by_author,
            "limitations": "Provider tokens and reservations include retry work and are not student training targets. Missing usage may include a remote charge. Reasoning tokens are part of completion usage, not added twice."}
