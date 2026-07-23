#!/usr/bin/env python3
"""variance_check — repeated train → independent eval runs → noise band (R5).

    python -m studio.tools.variance_check --stage cpt --config configs/cpt.yaml --seeds 3

For each seed, this tool first runs the fixed training stage in the current training
environment. After that process exits, it runs ``tools/eval_probes.py`` with
``NEKAISE_EVAL_PYTHON`` and parses the gym process's METRIC line. CPT/SFT never import
or launch gym themselves.

The resulting per-metric band is written to ``experiments/<exp>/noise_band.json``. The
adjudication rule refuses proper keep/revert verdicts until this file exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STAGES = {"cpt", "sft"}


def band(values: list[float]) -> dict:
    """Noise band from repeated runs: sample std (n≥2). Half-range reported alongside —
    with 2–3 seeds the std underestimates tails; the CONSERVATIVE band is max(std, hr)."""
    if len(values) < 2:
        raise ValueError("need ≥2 seed runs for a band")
    std = statistics.stdev(values)
    hr = (max(values) - min(values)) / 2
    return {"values": [round(v, 6) for v in values], "std": round(std, 6),
            "half_range": round(hr, 6), "band": round(max(std, hr), 6)}


def run_seed(stage: str, config: str, seed: int, eval_python: str,
             exp: str) -> tuple[str, str, float] | None:
    train = subprocess.run(
        [sys.executable, "-m", "studio.cli", "train", stage, "--config", config,
         "--seed", str(seed), "--quiet"], capture_output=True, text=True, cwd=REPO)
    sys.stdout.write(train.stdout[-1500:])
    if train.returncode:
        sys.stderr.write(train.stderr[-3000:])
        return None
    run_id = None
    for line in train.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "train_result":
            run_id = payload["run"]["run_id"]
    if not run_id:
        return None

    eval_env = os.environ.copy()
    eval_env["PATH"] = f"{Path(eval_python).resolve().parent}:{eval_env.get('PATH', '')}"
    evaluated = subprocess.run(
        [eval_python, str(REPO / "tools" / "eval_probes.py"),
         "--run-id", run_id, "--split", "dev"],
        capture_output=True, text=True, cwd=REPO, env=eval_env)
    sys.stdout.write(evaluated.stdout[-3000:])
    if evaluated.returncode:
        sys.stderr.write(evaluated.stderr[-3000:])
        return None
    m = re.search(r"METRIC (\S+)=([0-9.]+)", evaluated.stdout)
    return (run_id, m.group(1), float(m.group(2))) if m else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=3407)
    args = ap.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = cfg["run"]
    exp = run["experiment"]
    sys.path.insert(0, str(REPO / "lib"))
    from workspace import Workspace
    exp_dir = Workspace.resolve(REPO).experiment_dir(exp)
    eval_python = os.environ.get("NEKAISE_EVAL_PYTHON")
    if not eval_python:
        sys.exit("NEKAISE_EVAL_PYTHON is required (install requirements-eval.txt in an "
                 "isolated environment and point this variable at its Python)")
    if not Path(eval_python).is_file():
        sys.exit(f"NEKAISE_EVAL_PYTHON is not a file: {eval_python}")

    values: dict[str, list[float]] = {}
    run_ids: list[str] = []
    for i in range(args.seeds):
        seed = args.base_seed + i * 1013
        print(f"[variance] seed {seed} ({i + 1}/{args.seeds})")
        got = run_seed(args.stage, args.config, seed, eval_python, exp)
        if got is None:
            sys.exit(f"seed {seed}: no METRIC line — fix the run before calibrating noise")
        run_id, metric, value = got
        run_ids.append(run_id)
        values.setdefault(metric, []).append(value)

    out = {m: {**band(vs), "seeds": args.seeds, "run_ids": run_ids,
               "stage": args.stage,
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
