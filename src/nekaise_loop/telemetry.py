"""Measured usage across one continuation lineage; no model calls or data writes."""
from datetime import datetime, timezone
import json

from .storage import now
from .teacher_usage import normalize_usage, call_usage as recorded_call_usage
from .efficiency import usage_summary, add_call, efficiency, efficiency_history


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def call_usage(service, call):
    return recorded_call_usage(service.store, service.settings.workspace, call)


def sampled(points, limit=240):
    if len(points) <= limit:
        return points
    # Cumulative endpoints retain exact totals even for long training campaigns.
    return [points[round(i * (len(points) - 1) / (limit - 1))] for i in range(limit)]


def training_telemetry(service, campaign_id):
    store, lineage, seen = service.store, [], set()
    cid = campaign_id
    while cid and cid not in seen:
        seen.add(cid)
        campaign = store.campaign(cid)
        lineage.append(campaign)
        cid = campaign.get("parent_campaign_id")
    lineage.reverse()
    observed_at = now()
    observed = timestamp(observed_at)
    calls, measurements, events, rounds = [], [], [], []
    elapsed, started, clock_running = 0, None, False
    completed_rounds = 0
    for index, campaign in enumerate(lineage):
        cid = campaign["id"]
        end = lineage[index + 1]["created_at"] if index + 1 < len(lineage) else observed_at
        campaign_events = store.query("SELECT kind,data,created_at FROM events WHERE campaign_id=? AND created_at<=? ORDER BY id", (cid, end))
        active = None
        for event in campaign_events:
            event["data"] = json.loads(event["data"])
            data, kind, when = event["data"], event["kind"], timestamp(event["created_at"])
            opening = (kind == "campaign" and data.get("status") in {"running", "recovering", "waiting"}) or kind == "recovery" and "kind" in data
            closing = kind == "continuation" or kind == "campaign" and data.get("status") in {"paused", "stopped", "complete", "failed", "interrupted"}
            if opening and active is None:
                active = when
                started = min(started, when) if started is not None else when
            if closing and active is not None:
                elapsed += max(0, when - active)
                active = None
        if active is not None:
            live = campaign["status"] in {"running", "queued", "waiting", "recovering", "pausing", "stopping"}
            finish = timestamp(end) if live else min(timestamp(end), timestamp(campaign["updated_at"]))
            elapsed += max(0, finish - active)
            clock_running = index == len(lineage) - 1 and live
        events.extend(campaign_events)
        calls.extend(store.query("SELECT * FROM teacher_calls WHERE campaign_id=? AND created_at<=? ORDER BY id", (cid, end)))
        measurements.extend(store.query("SELECT m.* FROM metrics m JOIN rounds r ON r.id=m.round_id WHERE r.campaign_id=? AND m.created_at<=? ORDER BY m.id", (cid, end)))
        rounds.extend(store.query("SELECT id,campaign_id,number,status,updated_at FROM rounds WHERE campaign_id=? AND status='complete' AND updated_at<=? ORDER BY number", (cid, end)))
        completed_rounds += store.one("SELECT COUNT(*) AS n FROM rounds WHERE campaign_id=? AND status='complete' AND updated_at<=?", (cid, end))["n"]

    completed_calls = {e["data"]["call_id"]: e["created_at"] for e in events if e["kind"] == "teacher" and "call_id" in e["data"]}
    totals = {"input": 0, "output": 0, "cached": 0}
    teacher_points = []
    usage_by_round, all_usage = {}, usage_summary()
    for call in calls:
        usage, log_time = call_usage(service, call)
        add_call(usage_by_round.setdefault(call["round_id"], usage_summary()), call, usage)
        add_call(all_usage, call, usage)
        if usage is None:
            continue
        for key in totals:
            totals[key] += usage[key]
        teacher_points.append({"at": completed_calls.get(call["id"]) or log_time or call["created_at"], "delta": usage["input"] + usage["output"]})
    teacher_points.sort(key=lambda p: p["at"])
    total = 0
    for point in teacher_points:
        total += point.pop("delta")
        point["tokens"] = total

    training_points, attempt_tokens, streams = [], {}, {}
    trained = 0
    for row in sorted(measurements, key=lambda r: (r["created_at"], r["id"])):
        metric = json.loads(row["data"])
        tokens = metric.get("tokens")
        if not isinstance(tokens, (int, float)) or tokens < 0:
            continue
        key = (row["round_id"], row["attempt"])
        previous = attempt_tokens.get(key, 0)
        trained += max(0, tokens - previous)
        attempt_tokens[key] = max(previous, tokens)
        if metric.get("stream_tokens"):
            streams[key] = metric["stream_tokens"]
        training_points.append({"at": row["created_at"], "tokens": trained})

    trained_by_round = {}
    for (rid, _), tokens in attempt_tokens.items():
        trained_by_round[rid] = trained_by_round.get(rid, 0) + tokens
    ratios = efficiency_history(rounds, usage_by_round, trained_by_round)
    efficiency_result = {**efficiency(trained, all_usage), "series": sampled(ratios),
                         "completed_observations": len(ratios),
                         "missing_observations": sum(p["ratio"] is None for p in ratios)}

    origin = datetime.fromtimestamp(started, timezone.utc).isoformat() if started is not None else None
    def series(points, known=True):
        if not known or not origin:
            return []
        return sampled([{"at": origin, "tokens": 0}, *points, {"at": observed_at, "tokens": points[-1]["tokens"] if points else 0}])
    known = all_usage["reported_calls"] > 0 or len(calls) == 0
    return {"campaign_id": campaign_id, "root_campaign_id": lineage[0]["id"], "campaign_count": len(lineage),
            "observed_at": observed_at, "started_at": origin, "elapsed_seconds": elapsed,
            "clock_running": clock_running, "completed_rounds": completed_rounds, "efficiency": efficiency_result,
            "teacher": {**totals, "total": total if known else None, "reported_calls": all_usage["reported_calls"],
                        "missing_calls": all_usage["missing_calls"], "pending_calls": all_usage["pending_calls"], "series": series(teacher_points, known)},
            "training": {"total": trained, "updates": len(training_points),
                         "streams": {name: sum(s.get(name, 0) for s in streams.values()) for name in ("teacher", "corpus", "replay")},
                         "series": series(training_points)}}
