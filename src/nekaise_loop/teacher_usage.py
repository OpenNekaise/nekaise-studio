"""Read-only primary-teacher token evidence shared by telemetry and teaching feedback."""
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

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


def call_usage(store, workspace, call):
    value = normalize_usage(json.loads(call["usage"]))
    if value is not None:
        return value, None
    candidates = list((workspace / "runs" / call["round_id"]).glob(f"*/teacher-{call['id']}/provider.log"))
    if not candidates:
        rows = store.query("SELECT recovery_id,path FROM log_cleanup WHERE path LIKE ? AND status='removed'", (f"runs/{call['round_id']}/%/teacher-{call['id']}/provider.log",))
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

