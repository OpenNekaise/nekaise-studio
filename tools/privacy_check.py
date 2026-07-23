#!/usr/bin/env python3
"""privacy_check.py — keep proprietary/personal strings out of tracked files. Zero dependencies.

    python tools/privacy_check.py             # scan ALL git-tracked files (CI mode)
    python tools/privacy_check.py --staged    # scan lines being added by the staged diff (hook mode)

What it flags:
  - personal absolute paths            /home/<user>/... , /Users/<user>/...
  - Tailscale hostnames                *.ts.net
  - secret-shaped strings              sk-ant-..., sk-..., ghp_..., AKIA...
  - real building names                folder names under nekaise_data/ (derived at runtime,
                                       never hardcoded here) + terms from .privacy-denylist
                                       (one per line, git-ignored — partner names, addresses)
                                       + NEKAISE_DENYLIST (absolute path to a wordlist kept
                                       entirely OUTSIDE the repo — R10; set it in .env)

A line ending in `privacy-ok` is exempt (for documented, deliberate exceptions).
Lockfiles are skipped (base64 blobs false-positive on short terms). Exit 1 on any finding.

Install as a pre-commit hook (doctor checks this):  git config core.hooksPath tools/hooks
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DENYLIST_FILE = REPO / ".privacy-denylist"
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock"}

GENERIC = [
    ("personal path", re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")),
    ("tailscale host", re.compile(r"[A-Za-z0-9-]+\.ts\.net")),
    ("secret-shaped", re.compile(r"\b(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{24,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})\b")),
]


def denied_terms() -> list[str]:
    """Building folder names (from local data) + the local denylist. Empty in CI — fine."""
    terms = []
    data = Path(os.environ.get("NEKAISE_DATA", REPO / "nekaise_data"))
    if data.exists():
        terms += [d.name for d in data.iterdir()
                  if d.is_dir() and not d.name.startswith((".", "_"))
                  and d.name not in ("hvac_corpus", "documentations", "example-building")]
    env_list = os.environ.get("NEKAISE_DENYLIST")
    for f in (DENYLIST_FILE, Path(env_list) if env_list else None):
        if f and f.exists():
            terms += [l.strip() for l in f.read_text().splitlines()
                      if l.strip() and not l.startswith("#")]
    return [t for t in terms if len(t) >= 4]  # very short terms false-positive too easily


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=True).stdout


def scan_line(line: str, terms: list[str]) -> list[str]:
    if line.rstrip().endswith("privacy-ok"):
        return []
    hits = []
    for label, rx in GENERIC:
        m = rx.search(line)
        if m and "XXXX" not in m.group(0):  # XXXX marks a documentation placeholder
            hits.append(label)
    low = line.lower()
    for t in terms:  # real names are never OK anywhere, synthetic examples included
        if t.lower() in low:
            hits.append(f"denylisted term '{t}'")
    return hits


def scan_tracked(terms: list[str]) -> list[str]:
    findings = []
    for f in git("ls-files").splitlines():
        if Path(f).name in SKIP_FILES or f == ".privacy-denylist":
            continue
        p = REPO / f
        try:
            blob = p.read_bytes()
        except OSError:
            continue
        if b"\0" in blob[:8192]:  # binary — flag denylisted terms in the raw bytes only
            if any(t.lower().encode() in blob.lower() for t in terms):
                findings.append(f"{f}: binary file contains a denylisted term")
            continue
        for i, line in enumerate(blob.decode(errors="replace").splitlines(), 1):
            for hit in scan_line(line, terms):
                findings.append(f"{f}:{i}: {hit}")
    return findings


def scan_staged(terms: list[str]) -> list[str]:
    findings = []
    diff = git("diff", "--cached", "--unified=0", "--no-color")
    path = ""
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
        elif raw.startswith("+") and not raw.startswith("+++"):
            if Path(path).name in SKIP_FILES:
                continue
            for hit in scan_line(raw[1:], terms):
                findings.append(f"{path} (staged): {hit}")
    return findings


def main() -> int:
    terms = denied_terms()
    staged = "--staged" in sys.argv
    findings = scan_staged(terms) if staged else scan_tracked(terms)
    if findings:
        print("PRIVACY CHECK FAILED — proprietary/personal strings in tracked content:", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        print("\nRemove them (or append `privacy-ok` to a deliberate, documented exception).",
              file=sys.stderr)
        return 1
    n = "staged diff" if staged else "all tracked files"
    print(f"privacy check OK ({n}; {len(terms)} local term(s) screened)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
