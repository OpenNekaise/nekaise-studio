"""Tool-free Claude Code material production through worker-owned processes."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import threading

from ..failures import quota_kind, retry_seconds
from ..processes import ProcessRunner
from ..teacher_usage import normalize_usage


class ClaudeCodeAuthor:
    def __init__(self, client):
        pass  # The shared HTTP client belongs to other author transports.

    async def generate(self, author, request, *, env_file, max_bytes, execution=None):
        from .material import AuthorResult, AuthorHTTPError, AuthorWaiting

        if execution is None:
            raise RuntimeError("Claude Code material calls require a worker process owner")
        store, call_id = execution["store"], execution["call_id"]
        directory = Path(execution["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.loads(request["messages"][1]["content"])
        reserved = request["max_tokens"]
        if reserved < 256:
            raise ValueError("Claude Code jobs need at least 256 reserved output tokens")
        # CLI max-turns does not bound internal truncation recovery. Observe the
        # API stream, stop on truncation/another request, and reserve a second
        # message for a continuation already accepted before local cancellation.
        per_message = reserved // 2
        command = [execution["claude"], "-p", "--model", author.model,
            "--effort", author.options.get("effort", "medium"), "--max-turns", "1",
            "--tools", "", "--safe-mode", "--setting-sources", "",
            "--strict-mcp-config", "--no-session-persistence", "--output-format", "stream-json",
            "--verbose", "--include-partial-messages",
            "--system-prompt", request["messages"][0]["content"],
            "--json-schema", json.dumps(payload["output_schema"])]
        env = dict(os.environ)
        for key in ("ANTHROPIC_MODEL", "CLAUDE_CODE_RETRY_WATCHDOG", "CLAUDE_CODE_RESUME_INTERRUPTED_TURN",
                    "CLAUDE_CODE_RESUME_PROMPT", "CLAUDE_CODE_FALLBACK_MODEL"):
            env.pop(key, None)
        env.update(CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(per_message), CLAUDE_CODE_MAX_RETRIES="0",
                   MAX_STRUCTURED_OUTPUT_RETRIES="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        if not author.options.get("thinking", False):
            env["MAX_THINKING_TOKENS"] = "0"
        else:
            env.pop("MAX_THINKING_TOKENS", None)
        cancelled = threading.Event()
        envelope, starts, rejected_output = None, 0, None
        assistant_events, stream_message = [], None
        class TruncatedOutput(AuthorHTTPError):
            pass
        def observe(line):
            nonlocal envelope, starts, rejected_output, stream_message
            try:
                event = json.loads(line)
            except ValueError:
                return  # Preserve CLI diagnostics in the owned process log.
            if not isinstance(event, dict):
                return
            if event.get("type") == "result":
                envelope = event
            if event.get("type") == "assistant" and event.get("parent_tool_use_id") is None:
                assistant_events.append(event)
                message = event.get("message")
                blocks = message.get("content") if isinstance(message, dict) else None
                for block in blocks if isinstance(blocks, list) else []:
                    if (isinstance(block, dict) and block.get("type") == "tool_use"
                            and block.get("name") == "StructuredOutput" and isinstance(block.get("input"), dict)):
                        rejected_output = block["input"]
            part = event.get("event", {}) if event.get("type") == "stream_event" else {}
            if part.get("type") == "message_start":
                starts += 1
                stream_message = part.get("message")
                if starts > 1:
                    raise AuthorHTTPError("Claude Code attempted another model request; cancelled with usage unknown",
                                          {"stream_event": event, "reserved_output_tokens": reserved})
            if part.get("type") == "message_delta" and part.get("delta", {}).get("stop_reason") == "max_tokens":
                raise TruncatedOutput("Claude Code output was truncated; cancelled before internal continuation; usage unknown",
                                      {"stream_event": event, "reserved_output_tokens": reserved,
                                       "assistant_events": assistant_events, "stream_message": stream_message})
        runner = ProcessRunner(store, call_id, cancelled.is_set, table="material_calls")
        task = asyncio.create_task(asyncio.to_thread(runner.run, command, cwd=directory,
            log=directory/"provider.log", timeout=author.timeout_seconds,
            stdin=json.dumps(payload, ensure_ascii=False), env=env, check=False,
            max_output_bytes=max_bytes, on_output=observe))
        try:
            output = await asyncio.shield(task)
        except TruncatedOutput as exc:
            cancelled.set()
            await asyncio.gather(task, return_exceptions=True)
            # The owned runner has stopped before we expose any material. Only
            # explicit assistant text from this one pinned-model message may
            # reach salvage; tool input, thinking and unbounded deltas may not.
            if (execution.get("material_response_policy") != "salvage_v1" or starts != 1
                    or not isinstance(stream_message, dict)
                    or stream_message.get("model") != author.model
                    or stream_message.get("role") != "assistant" or not stream_message.get("id")):
                raise
            texts = []
            for event in assistant_events:
                message = event.get("message") or {}
                if (message.get("model") != author.model or message.get("role") != "assistant"
                        or message.get("id") != stream_message["id"]):
                    raise
                for block in message.get("content", []):
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        texts.append(block["text"])
            if not any(text.strip() for text in texts):
                raise
            raw = {**exc.evidence, "execution": {"transport": "claude_code",
                "requested_model": author.model, "observed_model_requests": starts,
                "max_model_requests": 1, "reserved_output_tokens": reserved,
                "max_output_tokens_per_message": per_message,
                "cancelled_before_internal_continuation": True, "usage_unknown": True}}
            # Partial stream usage is not final billing: retain the entire
            # reservation and unknown usage, including possible cancellation races.
            return AuthorResult("".join(texts), False, author.model, {}, raw)
        except BaseException:
            # Cancelling an asyncio waiter does not stop its thread or child.
            # Signal the owned runner and join it before releasing the worker.
            cancelled.set()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if envelope is None:
            raise AuthorHTTPError(f"Material author {author.id}: invalid Claude Code envelope",
                                  {"returncode": runner.returncode, "raw_output": output}) from None
        usage = normalize_usage(envelope)
        normalized = ({"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                       "total_tokens": usage["input"] + usage["output"],
                       "prompt_cache_hit_tokens": usage["cached"]} if usage is not None else {})
        raw = {"envelope": envelope, "execution": {"transport": "claude_code",
            "returncode": runner.returncode, "requested_model": author.model,
            "max_agent_turns": 1, "max_structured_attempts": 1,
            "observed_model_requests": starts, "max_model_requests": 1,
            "reserved_output_tokens": reserved, "max_output_tokens_per_message": per_message}}
        if runner.returncode or envelope.get("is_error") or envelope.get("subtype") != "success":
            error = str(envelope.get("result") or envelope.get("errors") or envelope.get("subtype"))
            kind = quota_kind(error)
            if kind:
                waiting = AuthorWaiting(f"Material author {author.id}: Claude Code {kind}; {error[-1000:]}",
                                        kind, retry_seconds(error), raw)
                waiting.usage = normalized
                raise waiting
            # Preserve rejected tool input solely for local structural diagnostics.
            # It is never a successful completion, even if our schema accepts it.
            content = None
            if envelope.get("subtype") == "error_max_structured_output_retries" and rejected_output is not None:
                raw["rejected_structured_output"] = rejected_output
                content = json.dumps(rejected_output, ensure_ascii=False)
            # A CLI failure is evidence for recovery, not a final author message.
            return AuthorResult(content, False, author.model, normalized, raw, 'nonfinal')
        models = envelope.get("modelUsage") or {}
        if models and set(models) != {author.model}:
            failure = AuthorHTTPError(f"Material author {author.id}: Claude Code used an unexpected model", raw)
            failure.usage = normalized
            raise failure
        if starts != 1:
            failure = AuthorHTTPError("Claude Code did not expose the required model-request stream", raw)
            failure.usage = normalized
            raise failure
        if normalized.get("completion_tokens", 0) > reserved:
            failure = AuthorHTTPError(f"Material author {author.id}: Claude Code exceeded its recorded output reservation", raw)
            failure.usage = normalized
            raise failure
        content = envelope.get("structured_output")
        return AuthorResult(json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else None,
                            isinstance(content, dict), author.model, normalized, raw)
