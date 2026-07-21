"""gym.runner — batch generation + evaluation orchestration (the *runner*, not "harness").

R4 wiring: NO `transformers.generate()` anywhere on eval/rollout/datagen paths.
- local checkpoints → vLLM offline engine (generate.VllmEngine)
- served / API models → OpenAI-compatible HTTP (generate.OpenAIServer) — a local
  `vllm serve` and a frontier API are the SAME code path.
"""
from gym.runner.evaluate import evaluate  # noqa: F401
from gym.runner.generate import OpenAIServer, VllmEngine, generator_for  # noqa: F401
