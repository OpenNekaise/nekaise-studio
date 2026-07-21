#!/usr/bin/env python3
"""eval_probes.py — score a model on corpus_probes (the pure-CPT loop metric) via the
gym runner. Base-model COMPLETION mode; the verdict per probe is gym.verifiers.
numeric_cloze — the same function training rewards import (R1).

    python tools/eval_probes.py --target ckpt:experiments/agentic-cpt/outputs/cpt --exp agentic-cpt
    python tools/eval_probes.py --target server:nekaise-1b@http://localhost:8000 --split dev
    python tools/eval_probes.py --checkpoint unsloth/Qwen3.5-0.8B        # ckpt: shorthand

Local checkpoints run on the vLLM offline engine; served/API models on the OpenAI-
compatible path — one runner, no transformers.generate (R4; the old HF implementation is
preserved unreferenced in attic/eval_probes_hf.py).

The frozen split (~20%) is MILESTONE-ONLY: requires --milestone; every use is audited.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lib"))

from results import log_result  # noqa: E402

from gym import tasks as gym_tasks  # noqa: E402
from gym.runner import evaluate, generator_for  # noqa: E402

OUT_DIR = REPO / "workspace" / "probe_eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--target", help="ckpt:<path|hf-id> or server:<model>@<base_url>")
    who.add_argument("--checkpoint", help="shorthand for ckpt:<value>")
    ap.add_argument("--split", default="dev", choices=["dev", "frozen", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--exp", default=None, help="experiment name for the results ledger")
    ap.add_argument("--milestone", action="store_true")
    args = ap.parse_args()
    target = args.target or f"ckpt:{args.checkpoint}"

    if args.split == "frozen":
        if not args.milestone:
            sys.exit("--split frozen is MILESTONE-ONLY: pass --milestone (and mean it). "
                     "The loop's signal is --split dev.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with (OUT_DIR / "frozen_split_audit.log").open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} target={target}\n")

    rows = gym_tasks.load("corpus_probes", split=args.split, n=args.limit)
    print(f"[probes] {target} on {len(rows)} {args.split} probes")

    def progress(done, total, records):
        if done % (args.batch_size * 10) < args.batch_size or done == total:
            acc = sum(r["correct"] for r in records) / done
            print(f"  [{done}/{total}] running acc={acc:.3f}", flush=True)

    report = evaluate(rows, generator_for(target), max_tokens=args.max_tokens,
                      batch_size=args.batch_size, progress=progress)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", target)
    (OUT_DIR / f"{safe}.{args.split}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in report["records"]) + "\n")

    absorption = report["by_track"].get("absorption")
    transfer = report["by_track"].get("transfer")
    print(f"PROBES[{target}] split={args.split} n={report['n']} "
          f"absorption={absorption} transfer={transfer}")
    if args.exp:
        log_result(REPO / "experiments" / args.exp, kind="eval", checkpoint=target,
                   split=args.split, metric=f"corpus_probes_{args.split}",
                   value=absorption, n=report["n"], transfer=transfer)
    if absorption is not None:
        print(f"METRIC corpus_probes_{args.split}={absorption:.4f}")


if __name__ == "__main__":
    main()
