#!/usr/bin/env python3
"""variance_check — reproducibility smoke test: same config, 2–3 seeds → noise band (R5).

    python -m studio.tools.variance_check --stage cpt --config configs/cpt.yaml --seeds 3

Runs the stage entry point per seed, parses its METRIC line, and writes the per-metric
noise band to experiments/<exp>/noise_band.json. The adjudication rule (explog.decide)
refuses proper keep/revert verdicts until this file exists: deciding against noise fills
LOG.md with false conclusions. Re-run after any change that could shift variance (new
task set, new corpus slice, new base).

First duty after the refactor: run this once on the mainline config and tag affected
crystallized skills `unverified` (crystallize_gate --mark-unverified).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STAGES = {"cpt", "sft", "rlvr", "opd"}


def band(values: list[float]) -> dict:
    """Noise band from repeated runs: sample std (n≥2). Half-range reported alongside —
    with 2–3 seeds the std underestimates tails; the CONSERVATIVE band is max(std, hr)."""
    if len(values) < 2:
        raise ValueError("need ≥2 seed runs for a band")
    std = statistics.stdev(values)
    hr = (max(values) - min(values)) / 2
    return {"values": [round(v, 6) for v in values], "std": round(std, 6),
            "half_range": round(hr, 6), "band": round(max(std, hr), 6)}


def run_seed(stage: str, config: str, seed: int) -> tuple[str, float] | None:
    proc = subprocess.run(
        [sys.executable, "-m", f"studio.stages.{stage}", "--config", config,
         "--seed", str(seed)], capture_output=True, text=True, cwd=REPO)
    sys.stdout.write(proc.stdout[-1500:])
    m = re.search(r"METRIC (\S+)=([0-9.]+)", proc.stdout)
    return (m.group(1), float(m.group(2))) if m else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=3407)
    args = ap.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_dir = REPO / "experiments" / cfg["run"]["experiment"]

    values: dict[str, list[float]] = {}
    for i in range(args.seeds):
        seed = args.base_seed + i * 1013
        print(f"[variance] seed {seed} ({i + 1}/{args.seeds})")
        got = run_seed(args.stage, args.config, seed)
        if got is None:
            sys.exit(f"seed {seed}: no METRIC line — fix the run before calibrating noise")
        metric, value = got
        values.setdefault(metric, []).append(value)

    out = {m: {**band(vs), "seeds": args.seeds, "stage": args.stage,
               "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
           for m, vs in values.items()}
    path = exp_dir / "noise_band.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**existing, **out}, indent=2))
    for m, b in out.items():
        print(f"NOISE_BAND {m}={b['band']} (std={b['std']}, half_range={b['half_range']})")
    print(f"-> {path} (explog.decide now enforces effect > band)")


if __name__ == "__main__":
    main()
