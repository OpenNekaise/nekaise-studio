"""CLI for the runner — the gym README's quick start, for real.

    python -m gym.runner --base-url http://localhost:8000/v1 --model MiniCPM5-1B \
        --task corpus_probes --split dev [--limit 200]
    python -m gym.runner --checkpoint experiments/<exp>/outputs/<stage> --task bench

Any OpenAI-compatible endpoint (vllm serve / Ollama /v1 / frontier API) or a local
checkpoint via the vLLM offline engine. Scores come from gym verifiers — the same
functions training rewards import. No studio required.
"""
from __future__ import annotations

import argparse
import json
import sys

from gym import tasks
from gym.runner import OpenAIServer, VllmEngine, evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gym.runner")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--base-url", help="OpenAI-compatible endpoint (with or without /v1)")
    who.add_argument("--checkpoint", help="local checkpoint dir or HF id (vLLM, GPU)")
    ap.add_argument("--model", help="model name at the endpoint (required with --base-url)")
    ap.add_argument("--task", default="corpus_probes")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)

    if args.base_url:
        if not args.model:
            ap.error("--model is required with --base-url")
        base = args.base_url.rstrip("/")
        base = base[:-3] if base.endswith("/v1") else base
        gen = OpenAIServer(base, args.model, api_key=args.api_key)
        target = f"{args.model}@{base}"
    else:
        gen = VllmEngine(args.checkpoint)
        target = args.checkpoint

    rows = tasks.load(args.task, split=args.split, n=args.limit)
    print(f"[gym] {target} on {len(rows)} {args.task}/{args.split} tasks", file=sys.stderr)
    report = evaluate(rows, gen, max_tokens=args.max_tokens, temperature=args.temperature)
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    else:
        print(f"SCORE {args.task}/{args.split} n={report['n']} "
              f"acc={report['acc']} mean={report['score']} by_track={report['by_track']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
