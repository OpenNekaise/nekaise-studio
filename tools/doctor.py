#!/usr/bin/env python3
"""doctor.py — preflight for a fresh clone. Zero dependencies, read-only, safe to run anytime.

    python tools/doctor.py

Checks the training stack (Python, GPU, packages), local config (.env, NEKAISE_HOLDOUT),
the sibling corpus, building data, and optional services. Prints one line per check
with a fix; exits non-zero only if something hard-blocks training.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
results: list[str] = []  # "pass" | "warn" | "fail"
TRAIN_PINS = {"unsloth": "2026.6.9", "trl": "0.24.0", "transformers": "5.5.0"}
EVAL_PINS = {"vllm": "0.25.1", "torch": "2.11.0", "transformers": "5.6.0",
             "ninja": "1.13.0"}


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

    sys.path.insert(0, str(REPO / "lib"))
    from workspace import Workspace
    try:
        workspace = Workspace.resolve(REPO).apply_environment()
        check("pass", f"workspace {workspace.workspace_id} "
              f"({'external' if workspace.external else 'legacy'})")
    except Exception as exc:
        check("fail", f"user workspace unavailable: {exc}",
              "initialize it with: python -m studio.cli workspace init <path>")
        workspace = Workspace.legacy(REPO)

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
        installed = {p: version(p) for p in TRAIN_PINS}
        drift = [f"{p}={installed[p]} (need {want})" for p, want in TRAIN_PINS.items()
                 if installed[p] != want]
        if drift:
            check("fail", f"training environment drift: {', '.join(drift)}",
                  "install the exact requirements.txt lock in the training environment")
        else:
            vers = ", ".join(f"{p} {installed[p]}" for p in TRAIN_PINS)
            check("pass", f"training lock matches ({vers})")
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
        check("warn", ".env missing", "cp .env.example .env")
    if configured("ANTHROPIC_API_KEY", envfile) or configured("OPENAI_API_KEY", envfile):
        check("pass", "frontier teacher key configured")
    else:
        check("warn", "no ANTHROPIC_API_KEY / OPENAI_API_KEY — teacher steps (build_data, judge) need one",
              "add a key to .env")

    eval_python = os.environ.get("NEKAISE_EVAL_PYTHON") or envfile.get("NEKAISE_EVAL_PYTHON")
    if not eval_python:
        check("warn", "NEKAISE_EVAL_PYTHON unset — training works, independent gym eval does not",
              "create an isolated env from requirements-eval.txt and set its Python path in .env")
    elif not Path(eval_python).is_file():
        check("fail", "NEKAISE_EVAL_PYTHON does not point to a file", "fix its path in .env")
    else:
        code = (
            "import json; from importlib.metadata import version; "
            f"print(json.dumps({{p: version(p) for p in {tuple(EVAL_PINS)!r}}}))"
        )
        proc = subprocess.run([eval_python, "-c", code], capture_output=True, text=True)
        if proc.returncode:
            check("fail", "gym eval environment is incomplete",
                  "install the exact requirements-eval.txt lock in that environment")
        else:
            installed_eval = json.loads(proc.stdout)
            drift = [f"{p}={installed_eval[p]} (need {want})"
                     for p, want in EVAL_PINS.items() if installed_eval[p] != want]
            if drift:
                check("fail", f"gym eval environment drift: {', '.join(drift)}",
                      "install the exact requirements-eval.txt lock in that environment")
            else:
                check("pass", "gym eval lock matches (vLLM 0.25.1, torch 2.11.0, "
                              "transformers 5.6.0)")

    # ── CPT corpus ──
    corpus = workspace.resolve_input("../nekaise-corpus")
    manifests = sorted((corpus / "manifest").glob("*.jsonl"))
    if manifests and (corpus / "text").is_dir():
        check("pass", f"nekaise-corpus found ({len(manifests)} manifest shard(s) + local text)")
    else:
        check("fail", "../nekaise-corpus is incomplete — CPT cannot build its data",
              "place the corpus beside this repository with manifest/ and text/")

    # ── building data + holdout ──
    data = workspace.data_dir
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
    else:
        check("warn", "no buildings under nekaise_data/ — building pack is idle",
              "drop real data into nekaise_data/<building>/, or try the synthetic demo: "
              "cp -r examples/example-building nekaise_data/")

    # ── optional services ──
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as r:
            n = len(json.loads(r.read()).get("models", []))
        check("pass", f"Ollama reachable at {base} ({n} model(s))")
    except Exception:
        check("warn", f"Ollama not reachable at {base} — only needed for ollama:* backends and serving",
              "https://ollama.com/download")
    try:
        sys.path.insert(0, str(REPO / "lib"))
        from runstore import RunStore
        integrity = RunStore(REPO).integrity()
        check("pass" if integrity["ok"] else "fail",
              f"agent run store integrity: {integrity['sqlite']}, "
              f"{integrity['counts']['runs']} indexed run(s)",
              "run: python -m studio.cli integrity" if not integrity["ok"] else "")
    except Exception as exc:
        check("fail", f"agent run store unavailable: {exc}",
              "run: python -m studio.cli reindex --reset")
    try:
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
        print("Ready. Next: read skills/coapt-round.md and drive the loop.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
