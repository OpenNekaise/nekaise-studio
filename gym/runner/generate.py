"""Generation backends for the runner. Two, and only two (R4):

    VllmEngine(checkpoint)   local weights — vLLM offline engine; LoRA adapter dirs are
                             resolved to (base, LoRARequest) automatically.
    OpenAIServer(base_url, model)   any OpenAI-compatible endpoint: a local `vllm serve`,
                             Ollama's /v1, or a frontier API — one code path for all.

Both expose complete(prompts) and chat(messages_list) -> list[str].
transformers `.generate()` is banned on these paths (tests grep for it).
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


class VllmEngine:
    """Offline vLLM over a local checkpoint dir or HF id. GPU-only, lazy import."""

    def __init__(self, checkpoint: str, *, max_model_len: int = 4096, seed: int = 3407):
        from vllm import LLM
        self.lora_request = None
        ckpt = Path(checkpoint)
        if (ckpt / "adapter_config.json").exists():
            from vllm.lora.request import LoRARequest
            base = json.loads((ckpt / "adapter_config.json").read_text())[
                "base_model_name_or_path"]
            self.llm = LLM(model=base, max_model_len=max_model_len, seed=seed,
                           enable_lora=True, max_lora_rank=64)
            self.lora_request = LoRARequest("adapter", 1, str(ckpt))
        else:
            self.llm = LLM(model=str(checkpoint), max_model_len=max_model_len, seed=seed)

    def _params(self, max_tokens: int, temperature: float, stop):
        from vllm import SamplingParams
        return SamplingParams(max_tokens=max_tokens, temperature=temperature,
                              stop=list(stop) if stop else None, seed=3407)

    def complete(self, prompts: list[str], *, max_tokens: int = 128,
                 temperature: float = 0.0, stop=None) -> list[str]:
        outs = self.llm.generate(prompts, self._params(max_tokens, temperature, stop),
                                 lora_request=self.lora_request)
        return [o.outputs[0].text for o in outs]

    def chat(self, messages_list: list[list[dict]], *, max_tokens: int = 128,
             temperature: float = 0.0, stop=None) -> list[str]:
        tok = self.llm.get_tokenizer()
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_list]
        return self.complete(prompts, max_tokens=max_tokens, temperature=temperature,
                             stop=stop)


class OpenAIServer:
    """OpenAI-compatible HTTP endpoint (vllm serve / Ollama /v1 / frontier API)."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def complete(self, prompts: list[str], *, max_tokens: int = 128,
                 temperature: float = 0.0, stop=None) -> list[str]:
        out = []
        for p in prompts:
            body = {"model": self.model, "prompt": p, "max_tokens": max_tokens,
                    "temperature": temperature, **({"stop": list(stop)} if stop else {})}
            r = self._post("/v1/completions", body)
            out.append(r["choices"][0]["text"])
        return out

    def chat(self, messages_list: list[list[dict]], *, max_tokens: int = 128,
             temperature: float = 0.0, stop=None) -> list[str]:
        out = []
        for msgs in messages_list:
            body = {"model": self.model, "messages": msgs, "max_tokens": max_tokens,
                    "temperature": temperature, **({"stop": list(stop)} if stop else {})}
            r = self._post("/v1/chat/completions", body)
            out.append(r["choices"][0]["message"]["content"])
        return out


def generator_for(target: str):
    """Resolve a CLI target spec:

        ckpt:<path-or-hf-id>              → VllmEngine
        server:<model>@<base_url>         → OpenAIServer
    """
    kind, _, rest = target.partition(":")
    if kind == "ckpt":
        return VllmEngine(rest)
    if kind == "server":
        model, _, base = rest.partition("@")
        return OpenAIServer(base or "http://localhost:8000", model)
    raise ValueError(f"unknown target '{target}' (use ckpt:<path> or server:<model>@<url>)")
