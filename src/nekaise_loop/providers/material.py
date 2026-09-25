"""Material-author transport. Only the worker calls this network boundary."""
from __future__ import annotations

import asyncio
import json
from typing import Protocol
from dataclasses import dataclass

import httpx

from ..author_config import AuthorSpec, credential
from ..failures import TeacherUnavailable


def _reject_nonfinite(value):
    raise ValueError("Nonfinite JSON value")


@dataclass(frozen=True)
class AuthorResult:
    """Provider-neutral content/completion evidence with the original envelope."""
    content: str | None
    complete: bool
    model: str | None
    usage: dict
    raw: object


class MaterialAuthor(Protocol):
    async def generate(self, author: AuthorSpec, request: dict, *, env_file, max_bytes: int, execution=None) -> AuthorResult: ...


class AuthorHTTPError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


class AuthorWaiting(TeacherUnavailable):
    def __init__(self, message, kind, retry_seconds, evidence):
        super().__init__(message, kind, retry_seconds)
        self.evidence = evidence


class OpenAIChatAuthor:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def generate(self, author, request, *, env_file, max_bytes, execution=None):
        key = credential(author.api_key_env, env_file)
        if author.api_key_env and not key:
            raise RuntimeError(f"Material author {author.id}: credential {author.api_key_env} is not configured")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            async with asyncio.timeout(author.timeout_seconds):
                async with self.client.stream("POST", author.base_url + "/chat/completions",
                                              json=request, headers=headers) as response:
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > max_bytes:
                            raise RuntimeError(f"Material author {author.id}: response exceeded byte budget")
                    text = body.decode("utf-8", errors="replace")
                    if key and key in text:
                        text = text.replace(key, "[REDACTED]")
                        raise AuthorHTTPError(f"Material author {author.id}: response echoed credential; response withheld", {"status": response.status_code, "body": text})
                    try:
                        envelope = json.loads(text, parse_constant=_reject_nonfinite)
                    except ValueError:
                        # Preserve malformed non-secret evidence for recovery.
                        envelope = {"transport_error": "invalid_json_envelope", "raw_body": text}
                    # Credentials must never reach saved outputs, even if echoed.
                    if key and key in json.dumps(envelope, ensure_ascii=False):
                        raise RuntimeError(f"Material author {author.id}: response echoed credential; response withheld")
                    if response.status_code in {402, 429}:
                        delay = response.headers.get("retry-after", "")
                        seconds = max(30, min(86400, int(delay))) if delay.isdigit() else None
                        raise AuthorWaiting(f"Material author {author.id}: HTTP {response.status_code}; waiting for {'balance' if response.status_code == 402 else 'rate availability'}",
                                            "quota" if response.status_code == 402 else "rate_limit", seconds, {"status": response.status_code, "response": envelope})
                    if response.status_code != 200:
                        raise AuthorHTTPError(f"Material author {author.id}: HTTP {response.status_code}", {"status": response.status_code, "response": envelope})
                    choices = envelope.get("choices", []) if isinstance(envelope, dict) else []
                    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
                    message = choice.get("message") or {}
                    content = message.get("content") if isinstance(message, dict) else None
                    model = envelope.get("model") if isinstance(envelope, dict) else None
                    usage = envelope.get("usage") if isinstance(envelope, dict) else None
                    return AuthorResult(content if isinstance(content, str) else None, choice.get("finish_reason") == "stop",
                                        model if isinstance(model, str) else None, usage if isinstance(usage, dict) else {}, envelope)
        except (httpx.HTTPError, TimeoutError) as exc:
            raise RuntimeError(f"Material author {author.id}: {type(exc).__name__}; remote charge may be unknown") from None


from .claude_material import ClaudeCodeAuthor
from .codex_material import CodexCodeAuthor
from .responses_material import OpenAIResponsesAuthor

AUTHOR_TRANSPORTS = {"openai_chat": OpenAIChatAuthor, "openai_responses": OpenAIResponsesAuthor,
                     "claude_code": ClaudeCodeAuthor, "codex_code": CodexCodeAuthor}
