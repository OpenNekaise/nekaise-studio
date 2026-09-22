"""Worker-owned Codex material generation with explicit CLI budget limitations."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading

from ..failures import quota_kind, retry_seconds
from ..processes import ProcessRunner


def strict_schema(node):
    """Copy the candidate schema into the Responses strict-output subset."""
    if isinstance(node, list):
        return [strict_schema(value) for value in node]
    if not isinstance(node, dict):
        return node
    result = {key: strict_schema(value) for key, value in node.items() if key != "default"}
    if result.get("type") == "object" and "properties" in result:
        result.update(required=list(result["properties"]), additionalProperties=False)
    return result


def reported_usage(event):
    usage = event.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    # Codex input already includes cached input; output already includes reasoning.
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens")):
        return {}
    result = {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
              "total_tokens": usage["input_tokens"] + usage["output_tokens"]}
    cached = usage.get("cached_input_tokens")
    if type(cached) is int and 0 <= cached <= usage["input_tokens"]:
        result["prompt_cache_hit_tokens"] = cached
    return result


class CodexCodeAuthor:
    def __init__(self, client):
        pass

    async def generate(self, author, request, *, env_file, max_bytes, execution=None):
        from .material import AuthorHTTPError, AuthorResult, AuthorWaiting

        if execution is None:
            raise RuntimeError("Codex material calls require a worker process owner")
        directory = Path(execution["directory"]).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.loads(request["messages"][1]["content"])
        schema = directory / "schema.json"
        schema.write_text(json.dumps(strict_schema(payload["output_schema"])))
        reserved = request["max_tokens"]
        command = [execution["codex"], "exec", "--strict-config", "--ignore-user-config",
                   "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "-C", str(directory),
                   "-s", "read-only", "-m", author.model,
                   "-c", "model_reasoning_effort=" + json.dumps(author.options.get("effort", "low")),
                   "-c", "project_doc_max_bytes=0", "-c", 'web_search="disabled"',
                   "-c", "suppress_unstable_features_warning=true",
                   "-c", "features.skip_host_skill_discovery=true"]
        for feature in ("shell_tool", "apps", "plugins", "hooks", "multi_agent", "browser_use",
                        "computer_use", "image_generation", "sleep_tool", "goals", "skill_search",
                        "unbounded_connection_retries"):
            command.extend(["-c", f"features.{feature}=false"])
        command.extend(["--output-schema", str(schema), "--json", "--color", "never", "-"])
        prompt = (request["messages"][0]["content"] + "\nGenerate only the requested final JSON. "
                  "Do not use tools, inspect files, or perform additional agent turns.\n" +
                  json.dumps(payload, ensure_ascii=False))
        cancelled = threading.Event()
        events, completed, content, turns = [], None, None, 0
        runner = ProcessRunner(execution["store"], execution["call_id"], cancelled.is_set, table="material_calls")

        def evidence():
            return {"events": events, "execution": {"transport": "codex_code",
                "requested_model": author.model, "model_provenance": "pinned CLI request; JSONL does not attest served model",
                "executable": str(Path(execution["codex"]).resolve()) if "/" in execution["codex"] else execution["codex"],
                "returncode": getattr(runner, "returncode", None), "observed_turns": turns,
                "reserved_output_tokens": reserved, "provider_output_token_cap": None,
                "budget_enforcement": "worker timeout/bytes; reported output acceptance ceiling",
                "internal_requests_observable": False}}

        def fail(message):
            error = AuthorHTTPError(f"Material author {author.id}: {message}", evidence())
            error.usage = reported_usage(completed) if completed else {}
            return error

        def observe(line):
            nonlocal completed, content, turns
            try:
                event = json.loads(line)
            except ValueError:
                return
            if not isinstance(event, dict):
                return
            events.append(event)
            kind = event.get("type")
            if kind == "turn.started":
                turns += 1
                if turns > 1:
                    raise fail("unexpected additional Codex turn; cancelled with remaining usage unknown")
            if kind in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item") or {}
                if item.get("type") not in {"agent_message", "reasoning", "error"}:
                    raise fail("Codex attempted a tool or unsupported item; cancelled with usage unknown")
                if kind == "item.completed" and item.get("type") == "agent_message":
                    content = item.get("text")
            if kind == "turn.completed":
                if completed is not None:
                    raise fail("duplicate Codex completion")
                completed = event

        task = asyncio.create_task(asyncio.to_thread(runner.run, command, cwd=directory,
            log=directory/"provider.log", timeout=author.timeout_seconds, stdin=prompt,
            check=False, max_output_bytes=max_bytes, on_output=observe))
        try:
            output = await asyncio.shield(task)
        except BaseException:
            cancelled.set()
            await asyncio.gather(task, return_exceptions=True)
            raise
        usage = reported_usage(completed) if completed else {}
        errors = any(e.get("type") in {"turn.failed", "error"} or
                     (e.get("type") in {"item.started", "item.updated", "item.completed"}
                      and (e.get("item") or {}).get("type") == "error") for e in events)
        if runner.returncode or completed is None or turns != 1 or errors:
            kind = quota_kind(output)
            if kind:
                error = AuthorWaiting(f"Material author {author.id}: Codex {kind}; {output[-1000:]}",
                                      kind, retry_seconds(output), evidence())
                error.usage = usage
                raise error
            raise fail("Codex did not finish one clean turn")
        if usage.get("completion_tokens", 0) > reserved:
            raise fail("Codex reported output exceeded its reservation")
        return AuthorResult(content if isinstance(content, str) else None,
                            isinstance(content, str), author.model, usage, evidence())
