"""Non-secret material-author configuration, frozen with each campaign."""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AuthorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    label: str = Field(min_length=1, max_length=160)
    transport: str = "openai_chat"
    location: str = Field(default="remote", pattern=r"^(remote|local)$")
    base_url: str = ""
    model: str = Field(min_length=1, max_length=160)
    api_key_env: str = Field(default="", pattern=r"^([A-Z][A-Z0-9_]*)?$")
    concurrency: int = Field(default=4, ge=1, le=256)
    resource_pool: str = Field(default="", max_length=100)
    max_output_tokens: int = Field(default=8192, ge=128, le=131072)
    timeout_seconds: int = Field(default=600, ge=1, le=3600)
    json_mode: bool = True
    options: dict = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def endpoint(cls, value):
        if not value:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Author endpoint must be an HTTP(S) URL without credentials, query or fragment")
        return value.rstrip("/")

    @field_validator("options")
    @classmethod
    def model_options(cls, value):
        reserved = {"model", "messages", "stream", "max_tokens", "response_format", "api_key", "authorization", "headers", "base_url"}
        if reserved.intersection(k.lower() for k in value):
            raise ValueError("Author options cannot override transport, credentials, messages or execution limits")
        json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def transport_options(self):
        if self.transport == "claude_code":
            if self.base_url or self.api_key_env:
                raise ValueError("Claude Code uses its existing CLI authentication, not an author endpoint or key")
            if not self.model.startswith("claude-"):
                raise ValueError("Claude Code authors require a pinned full model ID")
            if set(self.options) - {"effort", "thinking"}:
                raise ValueError("Claude Code author options allow only effort and thinking")
            if self.options.get("effort", "medium") not in {"low", "medium", "high"}:
                raise ValueError("Claude Code author effort must be low, medium or high")
            if type(self.options.get("thinking", False)) is not bool:
                raise ValueError("Claude Code author thinking must be boolean")
            if self.max_output_tokens < 256:
                raise ValueError("Claude Code needs at least 256 reserved output tokens")
        elif not self.base_url:
            raise ValueError("HTTP material authors require an endpoint")
        return self


class AuthorPool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    authors: list[AuthorSpec] = Field(default_factory=list)
    resource_limits: dict[str, int] = Field(default_factory=dict)
    concurrency: int = Field(default=4, ge=1, le=256)
    max_calls_per_round: int = Field(default=16, ge=1, le=100000)
    max_output_tokens_per_round: int = Field(default=32768, ge=128, le=100000000)
    max_input_chars_per_call: int = Field(default=120000, ge=1000, le=1000000)
    max_response_bytes: int = Field(default=2000000, ge=1000, le=20000000)

    @model_validator(mode="after")
    def pools(self):
        if len({a.id for a in self.authors}) != len(self.authors):
            raise ValueError("Material author IDs must be unique")
        if any(not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 256 for n in self.resource_limits.values()):
            raise ValueError("Resource pool limits must be integers in 1..256")
        if any(a.resource_pool and a.resource_pool not in self.resource_limits for a in self.authors):
            raise ValueError("Every shared resource pool needs an explicit concurrency limit")
        return self


def credential(name: str, env_file: Path) -> str:
    """Read only the requested credential. Never export it into teacher subprocesses.

    Supports simple KEY=value, quoted values and comments, without shell expansion.
    The process environment takes precedence, including an explicitly empty value.
    """
    if not name:
        return ""
    if name in os.environ:
        return os.environ[name]
    if not env_file.is_file():
        return ""
    for line in env_file.read_text().splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$", line)
        if match and match[1] == name:
            try:
                words = shlex.split(match[2], comments=True, posix=True)
            except ValueError:
                raise ValueError(f"Invalid quoting for credential variable {name}") from None
            if len(words) > 1:
                raise ValueError(f"Credential variable {name} must be one value")
            return words[0] if words else ""
    return ""


def load_pool(path: Path) -> AuthorPool:
    return AuthorPool.model_validate_json(path.read_text()) if path.is_file() else AuthorPool()


def catalog(pool: AuthorPool, env_file: Path):
    return [{"id": a.id, "label": a.label, "model": a.model, "location": a.location,
             "transport": a.transport,
             "max_output_tokens": a.max_output_tokens,
             "max_response_output_tokens": a.max_output_tokens // 2 if a.transport == "claude_code" else a.max_output_tokens,
             "output_budget_basis": "Claude Code allows one streamed model response using half the reservation; half covers a possible in-flight continuation on cancellation. Visible payload shares its response with reasoning." if a.transport == "claude_code" else "Provider output includes reasoning and final content",
             "credentials_configured": not a.api_key_env or bool(credential(a.api_key_env, env_file))}
            for a in pool.authors]
