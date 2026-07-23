#!/usr/bin/env python3
"""calibrate.py — measure per-task pass rates over a model ladder → difficulty bands (R6).

    python gym/tools/calibrate.py --task corpus_probes --split dev \
        --ladder student=ckpt:<resolved-checkpoint-path> \
        --ladder mid=server:qwen3.6:8b@http://localhost:11434 \
        --samples 4 --temperature 1.0

Each ladder model attempts every task `--samples` times at the given temperature; the
STUDENT's pass rate (ladder entry named 'student') drives the bands; other rungs are
recorded for range diagnostics (a task no rung solves is miscalibrated or broken, not
"hard"). Writes the sidecar via gym.tasks.sampler.write_calibration:

    gym/tasks/<set>/splits/calibration.json + {frontier,...,graduated}.json

Re-run periodically — the student's range drifts as it improves. Eval splits stay frozen
regardless; bands only shape the TRAINING pool (data recipe, not banned dynamic sampling).
Gaps in the ladder (bands with no tasks) are the signal to synthesize simplified variants
of frontier tasks (API teacher) — that authoring step lives in build_data recipes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym import tasks  # noqa: E402
from gym.runner import generator_for  # noqa: E402
from gym.tasks import sampler  # noqa: E402


def pass_rates(rows, gen, samples: int, temperature: float, max_tokens: int) -> dict[str, float]:
    hits = {t.id: 0 for t in rows}
    for s in range(samples):
        completes = [t for t in rows if t.mode == "complete"]
        chats = [t for t in rows if t.mode != "complete"]
        if completes:
            outs = gen.complete([t.prompt for t in completes],
                                max_tokens=max_tokens, temperature=temperature)
            for t, o in zip(completes, outs):
                hits[t.id] += tasks.is_correct(t, o)
        if chats:
            msgs = [([{"role": "system", "content": t.system}] if t.system else [])
                    + [{"role": "user", "content": t.prompt}] for t in chats]
            outs = gen.chat(msgs, max_tokens=max_tokens, temperature=temperature)
            for t, o in zip(chats, outs):
                hits[t.id] += tasks.is_correct(t, o)
        print(f"  sample {s + 1}/{samples} done", flush=True)
    return {tid: h / samples for tid, h in hits.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ladder", action="append", required=True,
                    metavar="NAME=TARGET", help="e.g. student=ckpt:<path>; rung named "
                    "'student' drives the bands")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    rows = tasks.load(args.task, split=args.split, n=args.limit)
    ladder = dict(kv.split("=", 1) for kv in args.ladder)
    if "student" not in ladder:
        sys.exit("ladder needs a rung named 'student' (its pass rate drives the bands)")

    all_rates: dict[str, dict[str, float]] = {}
    for name, target in ladder.items():
        print(f"[calibrate] {name} = {target} on {len(rows)} {args.task}/{args.split}")
        all_rates[name] = pass_rates(rows, generator_for(target), args.samples,
                                     args.temperature, args.max_tokens)

    out = sampler.write_calibration(
        args.task, all_rates["student"],
        meta={"split": args.split, "samples": args.samples,
              "temperature": args.temperature, "ladder": ladder,
              "rung_means": {n: round(sum(r.values()) / len(r), 4)
                             for n, r in all_rates.items()},
              "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"CALIBRATION written -> {out}")


if __name__ == "__main__":
    main()
