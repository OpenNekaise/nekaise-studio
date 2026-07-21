#!/usr/bin/env python3
"""Score a model on nekaise-bench — the EXTERNAL milestone referee (never a loop metric).

Two modes, unchanged since the decoupling reform:

  # Milestone / advisory: local weights via the gym runner (vLLM offline engine).
  python tools/eval_bench.py --checkpoint experiments/<exp>/outputs/<stage> --split dev
  python tools/eval_bench.py --checkpoint unsloth/Qwen3.5-0.8B --split dev     # baseline

  # Deployment parity (milestones): export to Ollama, run the official harness as-is.
  python serve/to_ollama.py --exp <exp> --name nekaise-candidate
  python tools/eval_bench.py --model nekaise-candidate

Grading + split + version all come from gym.tasks.bench / gym.verifiers.bench_qa (which
import the bench's own harness — the single grading truth). No transformers.generate on
this path (R4; the pre-refactor version is preserved in attic/eval_bench_hf.py). The
`test` split is FROZEN: requires --milestone, every use audited. Results record the bench
dataset version; scores are only comparable within one version. Checkpoint-mode and
Ollama-mode numbers are not interchangeable.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lib"))

from results import log_result  # noqa: E402

from gym.runner import VllmEngine, evaluate  # noqa: E402
from gym.tasks import bench as bench_tasks  # noqa: E402
from gym.verifiers import bench_qa  # noqa: E402


def eval_checkpoint(args) -> dict:
    rows = bench_tasks.load(split=args.split, n=args.limit)
    t0 = time.time()

    def progress(done, total, records):
        if done % (args.batch_size * 2) < args.batch_size or done == total:
            acc = sum(r["correct"] for r in records) / done
            print(f"  [{done}/{total}] running acc={acc:.3f}", flush=True)

    report = evaluate(rows, VllmEngine(args.checkpoint), max_tokens=args.max_new_tokens,
                      batch_size=args.batch_size, progress=progress)

    out_dir = REPO / "workspace" / "bench_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", str(args.checkpoint))
    (out_dir / f"{safe}.{args.split}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in report["records"]) + "\n")
    return {"model": str(args.checkpoint), "split": args.split, "n": report["n"],
            "overall": report["acc"], "by_track": report["by_track"],
            "mode": "checkpoint", "batch_size": args.batch_size,
            "minutes": round((time.time() - t0) / 60, 1)}


def eval_ollama(args) -> dict | None:
    cmd = [sys.executable, str(bench_qa.bench_dir() / "eval_ollama.py"),
           "--models", args.model]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    proc = subprocess.run(cmd, cwd=bench_qa.bench_dir(), capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    result = None
    for line in proc.stdout.splitlines():
        print(line)
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if proc.returncode != 0 or result is None:
        print("bench run failed", file=sys.stderr)
        return None
    result.update(split="all", mode="ollama")
    return result


def ledger_dir(args) -> Path | None:
    if args.exp:
        return REPO / "experiments" / args.exp
    if args.checkpoint:
        try:
            rel = Path(args.checkpoint).resolve().relative_to(REPO / "experiments")
            return REPO / "experiments" / rel.parts[0]
        except ValueError:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="Ollama model name (official harness, deployment parity)")
    who.add_argument("--checkpoint", help="local checkpoint dir or HF id (gym runner, GPU)")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--milestone", action="store_true")
    ap.add_argument("--exp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    if args.split == "test" and args.checkpoint:
        if not args.milestone:
            sys.exit("the test split is FROZEN for milestone checks — pass --milestone "
                     "(and log the verdict), or use --split dev")
        audit = REPO / "workspace" / "bench_eval" / "test_split_audit.log"
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {args.checkpoint}\n")

    result = eval_ollama(args) if args.model else eval_checkpoint(args)
    if result is None:
        return 1
    result["bench_version"] = bench_tasks.version()
    exp = ledger_dir(args)
    if exp and exp.exists():
        log_result(exp, kind="bench_eval", **result)
    print(f"BENCH[{result['model']}@{result['split']}|{result['bench_version']}] "
          f"nekaise_bench={result['overall']:.4f} "
          f"n={result['n']} by_track={result.get('by_track')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
