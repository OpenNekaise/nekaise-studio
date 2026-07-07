#!/usr/bin/env python3
"""Run nekaise-bench — the INDEPENDENT building-energy QA benchmark — against an Ollama model.

Third probe of the harness, next to `building_judge` (the gap) and `domain_quiz` (the
ceiling): its questions were authored and hardened OUTSIDE this repo's corpus, so a model
that merely memorized corpus text (e.g. via CPT) inflates corpus-derived quizzes but NOT
this one. Advisory signal — the keep/revert decision stays with the pack METRIC.

    python serve/to_ollama.py --exp <exp> --name nekaise-candidate    # export checkpoint first
    python tools/eval_bench.py --model nekaise-candidate [--limit 50]

Needs a local clone of https://github.com/OpenNekaise/nekaise-bench ; its location comes
from NEKAISE_BENCH_DIR, defaulting to ../nekaise-bench next to this repo. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("NEKAISE_BENCH_DIR", REPO.parent / "nekaise-bench"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name to evaluate")
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (smoke test)")
    args = ap.parse_args()

    runner = BENCH / "eval_ollama.py"
    if not runner.exists():
        print(f"nekaise-bench not found at {BENCH} — clone "
              f"https://github.com/OpenNekaise/nekaise-bench there, or set NEKAISE_BENCH_DIR.",
              file=sys.stderr)
        return 2

    cmd = [sys.executable, str(runner), "--models", args.model]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    proc = subprocess.run(cmd, cwd=BENCH, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    result = None
    for line in proc.stdout.splitlines():
        print(line)
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if proc.returncode != 0 or result is None:
        print("bench run failed", file=sys.stderr)
        return proc.returncode or 1
    # The studio-style metric line the loop logs alongside METRIC (advisory, never keep/revert).
    print(f"BENCH[{args.model}] nekaise_bench={result['overall']:.4f} n={result['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
