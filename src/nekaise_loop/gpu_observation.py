"""Read-only host GPU evidence for recovery, without model or benchmark imports."""
from __future__ import annotations

import csv
from pathlib import Path
import re
import subprocess
import tempfile

from .storage import now


def _number(value):
    if value in {"N/A", "[N/A]", "[Not Supported]"}:
        return None
    if not re.fullmatch(r"[0-9]{1,20}", value):
        raise ValueError("Invalid numeric observation")
    return int(value)


def _uuid(value):
    if not re.fullmatch(r"GPU-[a-zA-Z0-9-]{1,80}", value):
        raise ValueError("Invalid GPU identity")
    return value


def _query(kind, fields, converters):
    command = ["nvidia-smi", f"--query-{kind}={','.join(fields)}",
               "--format=csv,noheader,nounits"]
    result = {"command": command, "started_at": now(), "status": "unavailable",
              "rows": [], "error": None}
    try:
        # Keep output off the Python heap; read a bounded projection only. Never
        # preserve stderr or arbitrary command output (including on failures).
        with tempfile.TemporaryFile() as output:
            process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.DEVNULL, timeout=5, check=False)
            result["returncode"] = process.returncode
            if process.returncode:
                result["error"] = "query_failed"
                return result
            output.seek(0)
            data = output.read(65537)
        if len(data) > 65536:
            raise ValueError("Oversized observation")
        rows = list(csv.reader(data.decode("ascii").splitlines(), skipinitialspace=True))
        if len(rows) > 256 or any(len(row) != len(fields) for row in rows):
            raise ValueError("Invalid observation shape")
        result["rows"] = [dict(zip(fields, (convert(cell.strip()) for convert, cell
                             in zip(converters, row)))) for row in rows]
        result["status"] = "observed"
        if kind == "gpu" and not rows:
            result.update(status="unavailable", error="no_devices_reported")
    except subprocess.TimeoutExpired:
        result["error"] = "query_timeout"
    except FileNotFoundError:
        result["error"] = "query_not_found"
    except OSError:
        result["error"] = "query_os_error"
    except (ValueError, csv.Error):
        result["error"] = "invalid_query_output"
    finally:
        result["finished_at"] = now()
    return result


def _process_identity(pid, proc_root=Path("/proc")):
    """Numeric identity only: never read command lines, environment or workload files."""
    if pid is None or pid <= 0:
        return {"status": "unavailable"}
    directory = proc_root / str(pid)
    try:
        before = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        uid = directory.stat().st_uid
        after = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        start = _number(before[19])
        if start is None or before[19] != after[19]:
            return {"status": "changed_during_read"}
        return {"status": "observed", "start_ticks": str(start),
                "parent_pid": _number(after[1]), "uid": uid}
    except (OSError, ValueError, IndexError):
        return {"status": "unavailable"}


def gpu_observation():
    """Collect a bounded snapshot on the recovery host before entering its sandbox."""
    started = now()
    devices = _query("gpu", ["index", "uuid", "memory.total", "memory.used", "memory.free"],
                     [_number, _uuid, _number, _number, _number])
    processes = _query("compute-apps", ["gpu_uuid", "pid", "used_gpu_memory"],
                       [_uuid, _number, _number])
    for row in processes["rows"]:
        row["identity"] = _process_identity(row["pid"])
    return {"started_at": started, "finished_at": now(), "memory_unit": "MiB",
            "devices": devices, "compute_processes": processes,
            "limitations": "Point-in-time, non-atomic observations, not a memory reservation. "
            "Unknown memory is null, never zero. Compute processes omit graphics and may not "
            "identify all MPS/MIG consumers. Process identity is sampled after the GPU query "
            "and may be absent or reused; it does not establish ownership or authorize termination. "
            "No command lines, environment, workload files or benchmark evidence are read. "
            "The orchestrator decides recovery; no automatic capacity threshold is applied."}
