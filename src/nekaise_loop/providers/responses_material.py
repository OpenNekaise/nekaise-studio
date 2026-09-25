"""Bounded Responses SSE transport for HTTP Material Authors, without retries."""
from __future__ import annotations

import asyncio
import hashlib
import json

import httpx

from ..author_config import credential


def wire_request(request):
    """Map the journaled common Author request without changing its reservation."""
    if not isinstance(request.get("messages"), list) or any(
            not isinstance(message, dict) or not isinstance(message.get("content"), str)
            for message in request["messages"]):
        raise ValueError("Responses material requests require text-only messages")
    body = {k: v for k, v in request.items()
            if k not in {"messages", "max_tokens", "stream", "response_format"}}
    body.update(input=request["messages"], max_output_tokens=request["max_tokens"],
                stream=True, store=False)
    if "response_format" in request:
        if request["response_format"] != {"type": "json_object"}:
            raise ValueError("Responses material requests support the common json_object format only")
        body["text"] = {"format": request["response_format"]}
    return body


def reported_usage(response):
    """Reasoning is already included in output_tokens; absent usage stays unknown."""
    raw = response.get("usage")
    if not isinstance(raw, dict):
        return {}
    usage = {}
    for source, target in (("input_tokens", "prompt_tokens"),
                           ("output_tokens", "completion_tokens"),
                           ("total_tokens", "total_tokens")):
        if type(raw.get(source)) is int and raw[source] >= 0:
            usage[target] = raw[source]
    details = raw.get("input_tokens_details")
    if isinstance(details, dict) and type(details.get("cached_tokens")) is int and details["cached_tokens"] >= 0:
        usage["prompt_cache_hit_tokens"] = details["cached_tokens"]
    return usage


def final_content(response):
    """Only one completed assistant message is candidate material, never reasoning."""
    output = response.get("output")
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        return None
    if any(item.get("type") not in {"message", "reasoning"} for item in output):
        return None
    messages = [item for item in output if item.get("type") == "message"]
    if len(messages) != 1:
        return None
    message = messages[0]
    parts = message.get("content")
    if (message.get("role") != "assistant" or message.get("status") != "completed"
            or not isinstance(parts, list) or not parts
            or any(not isinstance(part, dict) or part.get("type") != "output_text"
                   or not isinstance(part.get("text"), str) for part in parts)):
        return None
    content = "".join(part["text"] for part in parts)
    return content if content.strip() else None


class SSEFrames:
    """Incremental LF/CRLF framing; UTF-8 is decoded only after a whole frame."""
    def __init__(self):
        self.buffer = b""
        self.data = []

    def feed(self, chunk):
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.removesuffix(b"\r")
            if not line:
                if self.data:
                    payload = b"\n".join(self.data).decode("utf-8", errors="strict")
                    self.data = []
                    yield payload
            elif line.startswith(b"data:"):
                self.data.append(line[5:].removeprefix(b" "))


class OpenAIResponsesAuthor:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def generate(self, author, request, *, env_file, max_bytes, execution=None):
        # Local import avoids the transport registry's import cycle.
        from .material import AuthorHTTPError, AuthorResult, AuthorWaiting, _reject_nonfinite

        key = credential(author.api_key_env, env_file)
        if author.api_key_env and not key:
            raise RuntimeError(f"Material author {author.id}: credential {author.api_key_env} is not configured")
        headers = {"Accept": "text/event-stream"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = wire_request(request)
        evidence = {"transport": "openai_responses", "adapter_version": 1,
                    "request_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                    "event_count": 0, "event_types": [], "received_bytes": 0}

        def checked(value):
            if key and key in json.dumps(value, ensure_ascii=False):
                raise AuthorHTTPError(f"Material author {author.id}: response echoed credential; response withheld",
                                      {"transport_error": "credential_echo_withheld"})
            return value

        def failure(kind):
            return AuthorHTTPError(f"Material author {author.id}: Responses {kind}; remote charge may be unknown",
                                   checked({**evidence, "transport_error": kind}))

        try:
            async with asyncio.timeout(author.timeout_seconds):
                # Keep the scheduler's idle timeout as well as the total deadline.
                async with self.client.stream("POST", author.base_url + "/responses", json=body, headers=headers) as response:
                    evidence["status"] = response.status_code
                    if response.status_code != 200:
                        raw = bytearray()
                        async for chunk in response.aiter_bytes():
                            raw.extend(chunk)
                            evidence["received_bytes"] = len(raw)
                            if len(raw) > max_bytes:
                                raise failure("response_exceeded_byte_budget")
                        text = checked(raw.decode("utf-8", errors="replace"))
                        try:
                            envelope = checked(json.loads(text, parse_constant=_reject_nonfinite))
                        except ValueError:
                            # Invalid JSON can contain a partially escaped secret.
                            # Preserve a fingerprint, never that undecodable body.
                            envelope = {"transport_error": "invalid_json_error_body", "bytes": len(raw),
                                        "sha256": hashlib.sha256(raw).hexdigest()}
                        error = checked({**evidence, "response": envelope})
                        if response.status_code in {402, 429}:
                            delay = response.headers.get("retry-after", "")
                            numeric_delay = delay.isascii() and delay.isdigit() and len(delay) < 10
                            seconds = max(30, min(86400, int(delay))) if numeric_delay else None
                            raise AuthorWaiting(f"Material author {author.id}: HTTP {response.status_code}; waiting for provider availability",
                                                "quota" if response.status_code == 402 else "rate_limit", seconds, error)
                        raise AuthorHTTPError(f"Material author {author.id}: HTTP {response.status_code}", error)
                    if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                        raise failure("expected_event_stream")
                    frames = SSEFrames()
                    response_id = None
                    async for chunk in response.aiter_bytes():
                        evidence["received_bytes"] += len(chunk)
                        if evidence["received_bytes"] > max_bytes:
                            raise failure("response_exceeded_byte_budget")
                        for payload in frames.feed(chunk):
                            checked(payload)
                            if payload.strip() == "[DONE]":
                                raise failure("missing_terminal_response")
                            event = checked(json.loads(payload, parse_constant=_reject_nonfinite))
                            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                                raise failure("invalid_event")
                            kind = event["type"]
                            evidence["event_count"] += 1
                            if kind not in evidence["event_types"]:
                                evidence["event_types"].append(kind)
                            envelope = event.get("response")
                            if isinstance(envelope, dict) and isinstance(envelope.get("id"), str):
                                if response_id is not None and response_id != envelope["id"]:
                                    raise failure("conflicting_response_ids")
                                response_id = envelope["id"]
                            if kind == "error":
                                raise AuthorHTTPError(f"Material author {author.id}: Responses error event", checked({**evidence, "event": event}))
                            if kind in {"response.completed", "response.done", "response.incomplete", "response.failed"}:
                                if not isinstance(envelope, dict):
                                    raise failure("missing_terminal_envelope")
                                content = final_content(envelope)
                                checked(content)
                                if content is not None:
                                    try:
                                        # Candidate JSON can itself encode escaped secrets.
                                        decoded_content = json.loads(content, parse_constant=_reject_nonfinite)
                                    except ValueError:
                                        pass  # Candidate validation owns JSON correctness.
                                    else:
                                        checked(decoded_content)
                                complete = (kind in {"response.completed", "response.done"}
                                            and envelope.get("status") == "completed" and not envelope.get("error")
                                            and not envelope.get("incomplete_details") and content is not None)
                                model = envelope.get("model")
                                raw = checked({**evidence, "terminal_event": kind, "response": envelope})
                                usage = reported_usage(envelope)
                                if usage.get("completion_tokens", 0) > request["max_tokens"]:
                                    error = AuthorHTTPError(f"Material author {author.id}: Responses reported output exceeded its reservation", raw)
                                    error.usage = usage
                                    raise error
                                # The first terminal ends this one request; do not wait for
                                # EOF or accept trailing generations on the same connection.
                                return AuthorResult(content, complete, model if isinstance(model, str) else None,
                                                    usage, raw)
                    raise failure("missing_terminal_response")
        except (httpx.HTTPError, TimeoutError) as exc:
            raise failure(type(exc).__name__) from None
        except (ValueError, UnicodeError):
            # No partial/raw frame: invalid or escaped credential echoes must not
            # escape through diagnostics. Counts and request hash locate the call.
            raise failure("invalid_event_encoding_or_json") from None
