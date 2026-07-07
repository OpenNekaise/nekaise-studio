#!/usr/bin/env python3
"""doctor.py — preflight for a fresh clone. Zero dependencies, read-only, safe to run anytime.

    python tools/doctor.py

Checks the training stack (Python, GPU, packages), local config (.env, NEKAISE_HOLDOUT),
data (nekaise_data/), and optional services (Ollama, dashboard). Prints one line per check
with a fix; exits non-zero only if something hard-blocks training.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
results: list[str] = []  # "pass" | "warn" | "fail"


def check(status: str, msg: str, fix: str = "") -> None:
    results.append(status)
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[status]
    line = f"  [{icon}] {msg}"
    if fix:
        line += f"\n         fix: {fix}"
    print(line)


def dotenv_keys() -> set[str]:
    """Names of variables set in the repo-root .env (values never read into output)."""
    env = REPO / ".env"
    if not env.exists():
        return set()
    keys = set()
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                keys.add(k.strip())
    return keys


def configured(name: str, envfile: set[str]) -> bool:
    return bool(os.environ.get(name)) or name in envfile


def main() -> int:
    print("Nekaise Studio doctor\n")

    # ── training stack ──
    if sys.version_info >= (3, 10):
        check("pass", f"Python {sys.version.split()[0]}")
    else:
        check("fail", f"Python {sys.version.split()[0]} — need 3.10+", "install Python 3.10 or newer")

    missing = [p for p in ("unsloth", "trl", "transformers", "datasets", "accelerate", "torch")
               if importlib.util.find_spec(p) is None]
    if missing:
        check("fail", f"packages missing: {', '.join(missing)}", "pip install -r requirements.txt")
    else:
        check("pass", "training packages installed (unsloth, trl, transformers, datasets, accelerate)")
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                check("pass", f"CUDA GPU: {p.name}, {p.total_memory / 1e9:.0f} GB")
            else:
                check("warn", "no CUDA GPU visible — training needs one (evals/data prep run anywhere)")
        except Exception as e:  # torch import can fail on broken installs
            check("warn", f"torch present but unusable: {e}", "reinstall: pip install -r requirements.txt")

    # ── local config ──
    envfile = dotenv_keys()
    if (REPO / ".env").exists():
        check("pass", ".env present")
    else:
        check("warn", ".env missing (fine for the gsm8k bootstrap)", "cp .env.example .env")
    if configured("ANTHROPIC_API_KEY", envfile) or configured("OPENAI_API_KEY", envfile):
        check("pass", "frontier teacher key configured")
    else:
        check("warn", "no ANTHROPIC_API_KEY / OPENAI_API_KEY — teacher steps (build_data, judge) need one",
              "add a key to .env")

    # ── building data + holdout ──
    data = REPO / "nekaise_data"
    buildings = sorted(d.name for d in data.iterdir()
                       if d.is_dir() and d.name != "hvac_corpus") if data.exists() else []
    if buildings:
        check("pass", f"nekaise_data/: {len(buildings)} building(s) found")
        holdout = os.environ.get("NEKAISE_HOLDOUT") or ("NEKAISE_HOLDOUT" in envfile and "(.env)")
        if not holdout:
            check("fail", "NEKAISE_HOLDOUT unset — the eval silently picks the first folder by name, "
                          "which can be the wrong building AND leak training data into the exam",
                  "set NEKAISE_HOLDOUT=<building folder> in .env")
        elif holdout != "(.env)" and holdout not in buildings:
            check("fail", f"NEKAISE_HOLDOUT does not match any folder under nekaise_data/",
                  f"pick one of: {', '.join(buildings)}")
        else:
            check("pass", "NEKAISE_HOLDOUT set")
        if (data / "hvac_corpus").exists():
            check("pass", "nekaise_data/hvac_corpus/ present (ceiling material)")
    else:
        check("warn", "no buildings under nekaise_data/ — building pack idle; gsm8k bootstrap still works",
              "drop a building's data into nekaise_data/<building>/")

    # ── optional services ──
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as r:
            n = len(json.loads(r.read()).get("models", []))
        check("pass", f"Ollama reachable at {base} ({n} model(s))")
    except Exception:
        check("warn", f"Ollama not reachable at {base} — only needed for ollama:* backends and serving",
              "https://ollama.com/download")
    if (REPO / "dashboard-ui" / "node_modules").exists():
        check("pass", "dashboard-ui ready (npm run dev)")
    else:
        check("warn", "dashboard-ui not installed (optional live dashboard)",
              "cd dashboard-ui && npm install && npm run dev")

    n_fail, n_warn = results.count("fail"), results.count("warn")
    print(f"\n{results.count('pass')} pass, {n_warn} warn, {n_fail} fail.")
    if n_fail == 0:
        print("Ready. Next: read skills/run-experiment.md and drive the loop.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
