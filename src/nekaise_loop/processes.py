"""Owned subprocesses with bounded cancellation and crash recovery identity checks."""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
import tempfile
from pathlib import Path


class Cancelled(Exception):
    pass


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def stop_owned(pid: int, started: str | None) -> None:
    if not started or process_start(pid) != started:
        return
    try:
        if os.getpgid(pid) != pid:
            raise RuntimeError("Refusing to stop a process outside its recorded group")
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 3
        while process_start(pid) == started and time.monotonic() < deadline:
            time.sleep(0.1)
        if process_start(pid) == started:
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_child(proc: subprocess.Popen) -> None:
    """An unreaped child handle is stronger ownership evidence than /proc metadata."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    except ProcessLookupError:
        proc.wait(timeout=5)


class ProcessRunner:
    def __init__(self, store, stage_id, cancelled=lambda: False, *, table="stage_runs"):
        self.store, self.stage_id, self.cancelled = store, stage_id, cancelled
        if table not in {"stage_runs", "recoveries", "material_calls"}:
            raise ValueError("Invalid process owner table")
        self.table = table

    def run(self, command: list[str], *, cwd: Path, log: Path, timeout: int, stdin: str | None = None, on_message=None, env=None, check=True, max_output_bytes=None, on_output=None) -> str:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as log_file, tempfile.TemporaryFile() as input_file:
            if stdin is not None:
                input_file.write(stdin.encode())
                input_file.seek(0)
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=input_file, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            started = process_start(proc.pid)
            try:
                self.store.execute(f"UPDATE {self.table} SET process_pid=?,process_start=? WHERE id=?", (proc.pid, started, self.stage_id))
                selector = selectors.DefaultSelector()
                selector.register(proc.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + timeout
                output, pending, received = [], b"", 0
                while selector.get_map():
                    if self.cancelled():
                        raise Cancelled("Stopped by operator")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Process exceeded {timeout}s budget")
                    for key, _ in selector.select(0.25):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        received += len(chunk)
                        if max_output_bytes is not None and received > max_output_bytes:
                            raise RuntimeError("Process output exceeded byte budget")
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            value = line.decode(errors="replace")
                            log_file.write(value + "\n")
                            log_file.flush()
                            if on_output is not None:
                                on_output(value)
                            output.append(value)
                            if len(output) > 1000:
                                output = output[-1000:]
                            # Only model transports opt into LOOP events. Agent logs
                            # may quote the protocol or print its documentation.
                            if on_message is not None and value.startswith("LOOP "):
                                on_message(json.loads(value[5:]))
                if pending:
                    value = pending.decode(errors="replace")
                    output.append(value)
                    log_file.write(value + "\n")
                    if on_output is not None:
                        on_output(value)
                code = proc.wait(timeout=5)
                self.returncode = code
                if code and check:
                    message = "\n".join(output[-12:])
                    for line in reversed(output):
                        try:
                            envelope = json.loads(line)
                            if isinstance(envelope, dict) and envelope.get("is_error") and envelope.get("result"):
                                message = str(envelope["result"])
                                break
                        except json.JSONDecodeError:
                            continue
                    raise RuntimeError(f"Process exited {code}: " + message[-2000:])
                return "\n".join(output)
            except BaseException:
                stop_child(proc)
                raise
            finally:
                if 'selector' in locals():
                    selector.close()
                proc.stdout.close()
                self.store.execute(f"UPDATE {self.table} SET process_pid=NULL,process_start=NULL WHERE id=?", (self.stage_id,))
