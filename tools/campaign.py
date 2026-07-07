#!/usr/bin/env python3
"""campaign.py — run a declarative matrix of evals/train runs; collect results; print the table.

A campaign is what research actually looks like (N models x M treatments), and encoding it
as a spec kills the whole class of ad-hoc-bash bugs (env-prefix parsing, lost METRIC lines,
hand-assembled tables). Runs execute SEQUENTIALLY (one GPU), failures are recorded and
skipped, and a re-run of the same spec SKIPS runs that already succeeded (idempotent
resume). Every result lands in the experiment ledger (lib/results.py) tagged with the
campaign name.

    python tools/campaign.py workspace/my_campaign.json [--dry-run]

Spec:
{
  "name": "sub4b-distill-v2",
  "experiment": "ceiling-sub4b",              # ledger + default cwd for stages
  "runs": [
    {"id": "base-08b",    "kind": "eval",  "checkpoint": "unsloth/Qwen3.5-0.8B",
     "split": "dev"},                          # extra: limit, batch_size, milestone
    {"id": "distill-08b", "kind": "train", "recipe": "experiments/ceiling-sub4b/train.py",
     "env": {"NEKAISE_STAGE": "distill_qwen08", "NEKAISE_BASE_MODEL": "unsloth/Qwen3.5-0.8B",
             "NEKAISE_INIT_FROM": "..."}}      # env is a DICT — nothing for bash to mangle
  ]
}

Each run's stdout/stderr goes to workspace/campaigns/<name>/<run-id>.log; the campaign
state (one JSON line per finished run) is <name>/state.jsonl — delete a line (or the file)
to force a re-run. Exit code: number of failed runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
from results import log_result  # noqa: E402

METRIC_RE = re.compile(r"(?:METRIC\s+\w+=|nekaise_bench=)([0-9.]+)")


def run_command(run: dict) -> list[str]:
    if run["kind"] == "eval":
        cmd = [sys.executable, str(REPO / "tools" / "eval_bench.py"),
               "--checkpoint", run["checkpoint"], "--split", run.get("split", "dev")]
        for flag in ("limit", "batch_size", "max_new_tokens"):
            if run.get(flag):
                cmd += [f"--{flag.replace('_', '-')}", str(run[flag])]
        if run.get("milestone"):
            cmd.append("--milestone")
        if run.get("exp"):
            cmd += ["--exp", run["exp"]]
        return cmd
    if run["kind"] == "train":
        return [sys.executable, str(REPO / run["recipe"])]
    raise ValueError(f"unknown run kind: {run['kind']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="path to the campaign JSON spec")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    name, exp = spec["name"], spec.get("experiment")
    exp_dir = REPO / "experiments" / exp if exp else None
    cdir = REPO / "workspace" / "campaigns" / name
    cdir.mkdir(parents=True, exist_ok=True)
    state_path = cdir / "state.jsonl"
    done = {}
    if state_path.exists():
        for line in state_path.read_text().splitlines():
            row = json.loads(line)
            done[row["id"]] = row

    results, failures = [], 0
    for run in spec["runs"]:
        rid = run["id"]
        if rid in done and done[rid].get("status") == "ok":
            print(f"[skip] {rid} (already done: {done[rid].get('value')})")
            results.append(done[rid])
            continue
        cmd = run_command(run)
        env = {**os.environ, **{k: str(v) for k, v in run.get("env", {}).items()}}
        print(f"[run ] {rid}: {' '.join(cmd)}"
              + (f"  env+={list(run['env'])}" if run.get("env") else ""), flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        with (cdir / f"{rid}.log").open("a") as logf:
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                  cwd=REPO, text=True)
        tail = (cdir / f"{rid}.log").read_text()[-4000:]
        m = list(METRIC_RE.finditer(tail))
        value = float(m[-1].group(1)) if m else None
        status = "ok" if proc.returncode == 0 and value is not None else "failed"
        row = {"id": rid, "status": status, "value": value, "kind": run["kind"],
               "minutes": round((time.time() - t0) / 60, 1)}
        with state_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if status == "failed":
            failures += 1
            print(f"[FAIL] {rid} (exit {proc.returncode}) — see {cdir / (rid + '.log')}")
        else:
            print(f"[ ok ] {rid}: {value} ({row['minutes']} min)")
            if exp_dir and exp_dir.exists():
                log_result(exp_dir, kind="campaign_run", campaign=name, run_id=rid,
                           value=value, spec=run)
        results.append(row)

    if not args.dry_run:
        print(f"\n## campaign {name}\n\n| run | status | value | min |\n|---|---|---|---|")
        for r in results:
            print(f"| {r['id']} | {r['status']} | {r.get('value')} | {r.get('minutes', '')} |")
    return failures


if __name__ == "__main__":
    sys.exit(main())
