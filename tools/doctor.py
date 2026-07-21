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


def dotenv() -> dict[str, str]:
    """Variables set in the repo-root .env (secret values are only checked, never printed)."""
    env = REPO / ".env"
    if not env.exists():
        return {}
    out = {}
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                out[k.strip()] = v.strip()
    return out


def configured(name: str, envfile: dict[str, str]) -> bool:
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
        from importlib.metadata import version
        vers = ", ".join(f"{p} {version(p)}" for p in ("unsloth", "trl", "transformers"))
        check("pass", f"training packages installed ({vers})")
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
    envfile = dotenv()
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
        holdout = os.environ.get("NEKAISE_HOLDOUT") or envfile.get("NEKAISE_HOLDOUT")
        if not holdout:
            check("fail", "NEKAISE_HOLDOUT unset — the eval falls back to the first folder by name, "
                          "which can be the wrong building AND leak training data into the exam",
                  "set NEKAISE_HOLDOUT=<building folder> in .env")
        elif holdout not in buildings:
            check("fail", "NEKAISE_HOLDOUT does not match any folder under nekaise_data/",
                  f"pick one of: {', '.join(buildings)}")
        else:
            try:
                sys.path.insert(0, str(REPO / "gym" / "tasks" / "building"))
                import prepare  # noqa: PLC0415
                exam_b = prepare.exam_building()
            except Exception:
                exam_b = None
            if exam_b and exam_b != holdout:
                check("fail", f"NEKAISE_HOLDOUT is set but the frozen exam is about a different building",
                      f"set NEKAISE_HOLDOUT={exam_b} (the exam's building)")
            else:
                check("pass", "NEKAISE_HOLDOUT set" + (" and matches the frozen exam" if exam_b else ""))
        if (data / "hvac_corpus").exists():
            check("pass", "nekaise_data/hvac_corpus/ present (ceiling material)")
    else:
        check("warn", "no buildings under nekaise_data/ — building pack idle; gsm8k bootstrap still works",
              "drop real data into nekaise_data/<building>/, or try the synthetic demo: "
              "cp -r examples/example-building nekaise_data/")

    # ── optional services ──
    bench = Path(os.environ.get("NEKAISE_BENCH_DIR", REPO.parent / "nekaise-bench"))
    if (bench / "eval_ollama.py").exists():
        check("pass", f"nekaise-bench found ({bench.name}) — third metric available")
    else:
        check("warn", "nekaise-bench not found — ceiling-phase metric unavailable",
              "git clone https://github.com/OpenNekaise/nekaise-bench next to this repo "
              "(or set NEKAISE_BENCH_DIR)")
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
    try:
        import subprocess
        hooks = subprocess.run(["git", "-C", str(REPO), "config", "core.hooksPath"],
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        hooks = ""
    if hooks == "tools/hooks":
        check("pass", "privacy pre-commit hook active (core.hooksPath=tools/hooks)")
    else:
        check("warn", "privacy pre-commit hook not installed — leaks are only caught in CI",
              "git config core.hooksPath tools/hooks")

    n_fail, n_warn = results.count("fail"), results.count("warn")
    print(f"\n{results.count('pass')} pass, {n_warn} warn, {n_fail} fail.")
    if n_fail == 0:
        print("Ready. Next: read skills/run-experiment.md and drive the loop.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
