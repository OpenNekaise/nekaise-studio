"""llm — text generation backends for data building. FIXED plumbing.

One function, `generate(...)`, talks to whatever model produces training data:

  - "ollama:<model>"     local model via the host Ollama server (default, free, offline)
  - "anthropic:<model>"  frontier teacher via the Anthropic API (needs ANTHROPIC_API_KEY)
  - "openai:<model>"     frontier teacher via the OpenAI API (needs OPENAI_API_KEY)

build_data.py picks the backend (that choice is a knob it varies); *how* we call it
is fixed here. stdlib-only (urllib), so no extra dependency. Backend ids keep the
model verbatim after the first ":" (Ollama tags contain colons, e.g. "qwen3.6:27b").
"""
from __future__ import annotations

import json
import os
import urllib.request

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _messages(system: str | None, user: str | None, messages: list[dict] | None) -> list[dict]:
    if messages is not None:
        return messages
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user or ""})
    return msgs


def generate(
    backend: str,
    *,
    system: str | None = None,
    user: str | None = None,
    messages: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 600.0,
) -> str:
    """Generate a completion. Provide (system, user) or a full `messages` list."""
    provider, _, model = backend.partition(":")
    msgs = _messages(system, user, messages)

    if provider == "ollama":
        out = _post(
            f"{OLLAMA_BASE_URL}/api/chat",
            {"model": model, "messages": msgs, "stream": False,
             "options": {"temperature": temperature, "num_predict": max_tokens}},
            {}, timeout,
        )
        return out["message"]["content"]

    if provider == "anthropic":
        # Opus 4.8 rejects temperature/top_p/top_k (400) — omit them; steer via the prompt.
        # Cache the (large, reused) system context so repeated distillation calls pay ~0.1x
        # on the shared prefix instead of full price.
        key = os.environ["ANTHROPIC_API_KEY"]
        sys_txt = next((m["content"] for m in msgs if m["role"] == "system"), None)
        chat = [m for m in msgs if m["role"] != "system"]
        body: dict = {"model": model, "max_tokens": max_tokens, "messages": chat}
        if sys_txt:
            body["system"] = [{"type": "text", "text": sys_txt,
                               "cache_control": {"type": "ephemeral"}}]
        out = _post(
            "https://api.anthropic.com/v1/messages", body,
            {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout,
        )
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")

    if provider == "openai":
        key = os.environ["OPENAI_API_KEY"]
        out = _post(
            "https://api.openai.com/v1/chat/completions",
            {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens},
            {"Authorization": f"Bearer {key}"}, timeout,
        )
        return out["choices"][0]["message"]["content"]

    raise ValueError(f"unknown backend provider: {provider!r} (use ollama:/anthropic:/openai:)")
