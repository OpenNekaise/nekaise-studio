"""Measured usage across one continuation lineage; no model calls or data writes."""
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

from .storage import now


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def normalize_usage(usage):
    if not isinstance(usage, dict):
        return None
    valid = lambda value: type(value) is int and value >= 0
    if valid(usage.get("input_tokens")) and valid(usage.get("output_tokens")) and valid(usage.get("cached_input_tokens", 0)):
        return {"input": usage["input_tokens"], "output": usage["output_tokens"],
                "cached": usage.get("cached_input_tokens", 0)}
    models = usage.get("models") or usage.get("modelUsage")
    if isinstance(models, dict) and models and all(isinstance(m, dict) and valid(m.get("inputTokens")) and valid(m.get("outputTokens")) and valid(m.get("cacheReadInputTokens", 0)) and valid(m.get("cacheCreationInputTokens", 0)) for m in models.values()):
        # Claude reports uncached input separately from cache reads/writes.
        return {"input": sum(m["inputTokens"] + m.get("cacheReadInputTokens", 0) + m.get("cacheCreationInputTokens", 0) for m in models.values()),
                "output": sum(m["outputTokens"] for m in models.values()),
                "cached": sum(m.get("cacheReadInputTokens", 0) for m in models.values())}
    return None


@lru_cache(maxsize=512)
def log_usage(path, size, modified):
    """Read only provider usage envelopes; ignore embedded text/tool output.

    Stat identity invalidates active logs. Completed logs are parsed once per API
    process, and existing immutable provider evidence is not rewritten.
    """
    total = None
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            try:
                envelope = json.loads(line)
            except ValueError:
                continue
            if not isinstance(envelope, dict):
                continue
            if envelope.get("type") == "turn.completed":
                value = normalize_usage(envelope.get("usage") or {})
                if value:
                    total = {k: (total or {}).get(k, 0) + v for k, v in value.items()}
            elif envelope.get("type") == "result" and envelope.get("modelUsage"):
                total = normalize_usage(envelope)
    return total


def call_usage(service, call):
    value = normalize_usage(json.loads(call["usage"]))
    if value is not None:
        return value, None
    workspace = service.settings.workspace
    candidates = list((workspace / "runs" / call["round_id"]).glob(f"*/teacher-{call['id']}/provider.log"))
    if not candidates:
        rows = service.store.query("SELECT recovery_id,path FROM log_cleanup WHERE path LIKE ? AND status='removed'", (f"runs/{call['round_id']}/%/teacher-{call['id']}/provider.log",))
        candidates = [workspace / "trash" / "history" / str(r["recovery_id"]) / r["path"] for r in rows]
    for path in candidates:
        if not path.resolve().is_relative_to(workspace.resolve()):
            continue
        try:
            stat = path.stat()
            value = log_usage(str(path), stat.st_size, stat.st_mtime_ns)
            if value is not None:
                return value, datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        except OSError:
            continue
    return None, None


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
    calls, measurements, events = [], [], []
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
        completed_rounds += store.one("SELECT COUNT(*) AS n FROM rounds WHERE campaign_id=? AND status='complete' AND updated_at<=?", (cid, end))["n"]

    completed_calls = {e["data"]["call_id"]: e["created_at"] for e in events if e["kind"] == "teacher" and "call_id" in e["data"]}
    totals = {"input": 0, "output": 0, "cached": 0}
    teacher_points, reported, missing, pending = [], 0, 0, 0
    for call in calls:
        usage, log_time = call_usage(service, call)
        if usage is None:
            if call["status"] == "running":
                pending += 1
            else:
                missing += 1
            continue
        reported += 1
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

    origin = datetime.fromtimestamp(started, timezone.utc).isoformat() if started is not None else None
    def series(points, known=True):
        if not known or not origin:
            return []
        return sampled([{"at": origin, "tokens": 0}, *points, {"at": observed_at, "tokens": points[-1]["tokens"] if points else 0}])
    known = reported > 0 or len(calls) == 0
    return {"campaign_id": campaign_id, "root_campaign_id": lineage[0]["id"], "campaign_count": len(lineage),
            "observed_at": observed_at, "started_at": origin, "elapsed_seconds": elapsed,
            "clock_running": clock_running, "completed_rounds": completed_rounds,
            "teacher": {**totals, "total": total if known else None, "reported_calls": reported,
                        "missing_calls": missing, "pending_calls": pending, "series": series(teacher_points, known)},
            "training": {"total": trained, "updates": len(training_points),
                         "streams": {name: sum(s.get(name, 0) for s in streams.values()) for name in ("teacher", "corpus", "replay")},
                         "series": series(training_points)}}
