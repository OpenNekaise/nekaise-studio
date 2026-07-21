"""sandbox_tests — run gold unit tests against code in the response, sandboxed (R7).

meta: {"tests": <python source asserting on `import solution`>,
       "timeout_s": <int> = 10,
       "mem_mb": <int> = 512}

The response's python code (``` fence, else the whole text) is written as solution.py in
a throwaway tmpdir; the TESTS ARE NEVER WRITTEN TO DISK — they are piped to an isolated
interpreter's stdin (`python -I -`), so a policy that learns to read files during training
cannot read its own grader. Binary score: 1.0 iff the test process exits 0.

Sandbox: wall-clock timeout, CPU rlimit, address-space rlimit, empty environment (no
proxy/API vars → no ambient credentials; NOTE this is best-effort network denial —
true isolation needs a netns/container, tracked as a roadmap item), cwd inside the
tmpdir which is removed afterward.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.S | re.I)


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(str(text))
    return max(blocks, key=len).strip() if blocks else str(text).strip()


_BOOTSTRAP = "import sys, os\nsys.path.insert(0, os.getcwd())\n"   # -I drops cwd; restore
                                                                   # ONLY the sandbox dir


def _limits(mem_mb: int, cpu_s: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    return apply


def verify(prompt: str, response: str, meta: dict) -> float:
    tests = meta.get("tests")
    if not tests:
        return 0.0
    timeout = int(meta.get("timeout_s", 10))
    mem_mb = int(meta.get("mem_mb", 512))
    with tempfile.TemporaryDirectory(prefix="gym_sbx_") as td:
        (Path(td) / "solution.py").write_text(extract_code(response))
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-"], input=_BOOTSTRAP + tests, text=True,
                capture_output=True, timeout=timeout, cwd=td, env={},
                preexec_fn=_limits(mem_mb, timeout))
        except (subprocess.TimeoutExpired, OSError):
            return 0.0
        return 1.0 if proc.returncode == 0 else 0.0
