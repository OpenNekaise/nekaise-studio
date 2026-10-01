import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from nekaise_loop import gpu_observation as gpu


def fake_queries(monkeypatch, *outputs):
    pending = iter(outputs)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 5
        assert kwargs["stdin"] == kwargs["stderr"] == subprocess.DEVNULL
        item = next(pending)
        if isinstance(item, Exception):
            raise item
        code, output = item
        kwargs["stdout"].write(output)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(gpu.subprocess, "run", run)
    return calls


def test_capacity_and_process_identity_are_separate_observations(monkeypatch):
    calls = fake_queries(monkeypatch, (0, b"0, GPU-abc, 48476, 34648, 13828\n"),
                         (0, b"GPU-abc, 179123, 34120\nGPU-abc, 53956, [N/A]\n"))
    monkeypatch.setattr(gpu, "_process_identity", lambda pid: {"status": "unavailable"})
    result = gpu.gpu_observation()
    assert len(calls) == 2
    assert result["devices"]["rows"][0]["memory.free"] == 13828
    processes = result["compute_processes"]["rows"]
    assert processes[0]["pid"] == 179123 and processes[0]["used_gpu_memory"] == 34120
    assert processes[1]["used_gpu_memory"] is None
    assert processes[0]["identity"] == {"status": "unavailable"}
    assert result["memory_unit"] == "MiB" and result["finished_at"] >= result["started_at"]


@pytest.mark.parametrize("failure,error", [
    (FileNotFoundError("private path"), "query_not_found"),
    (PermissionError("private path"), "query_os_error"),
    (subprocess.TimeoutExpired("private command", 5), "query_timeout"),
    ((9, b"private output"), "query_failed"),
    ((0, b"private output"), "invalid_query_output"),
    ((0, b"0, GPU-abc, 10, 2, invalid\n"), "invalid_query_output"),
    ((0, b"x" * 65537), "invalid_query_output"),
    ((0, b"0, GPU-abc, 10, 2, 8\n" * 257), "invalid_query_output"),
    ((0, b""), "no_devices_reported"),
])
def test_unavailable_is_not_zero_and_does_not_block_other_query(monkeypatch, failure, error):
    fake_queries(monkeypatch, failure, (0, b""))
    result = gpu.gpu_observation()
    assert result["devices"]["status"] == "unavailable"
    assert result["devices"]["error"] == error
    assert result["devices"]["rows"] == []
    assert result["compute_processes"]["status"] == "observed"
    assert result["compute_processes"]["rows"] == []
    assert "private" not in json.dumps(result)


def stat_line(start):
    return "123 (name with ) spaces) " + " ".join(["S", "99"] + ["0"] * 17 + [str(start)])


def test_numeric_identity_reads_only_stat_and_directory_owner(tmp_path):
    directory = tmp_path / "123"
    directory.mkdir()
    (directory / "stat").write_text(stat_line(456))
    assert gpu._process_identity(123, tmp_path) == {
        "status": "observed", "start_ticks": "456", "parent_pid": 99,
        "uid": directory.stat().st_uid}
    assert gpu._process_identity(124, tmp_path) == {"status": "unavailable"}
    assert gpu._process_identity(None, tmp_path) == {"status": "unavailable"}


def test_pid_reuse_during_identity_read_is_explicit(tmp_path, monkeypatch):
    (tmp_path / "123").mkdir()
    contents = iter([stat_line(456), stat_line(789)])
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: next(contents))
    assert gpu._process_identity(123, tmp_path) == {"status": "changed_during_read"}
