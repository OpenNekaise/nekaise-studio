import json
import os
import subprocess
import sys
import time

import pytest

from nekaise_loop.processes import ProcessRunner, stop_child, stop_owned, process_start
from nekaise_loop.storage import Store


def test_owned_child_stops_with_no_persisted_metadata():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        start = time.monotonic()
        stop_child(proc)
        assert proc.poll() is not None
        assert time.monotonic()-start < 5
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_recovery_does_not_signal_reused_or_unrelated_identity():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        stop_owned(proc.pid, "wrong-start-identity")
        assert proc.poll() is None
    finally:
        stop_child(proc)


def test_model_transport_decodes_events_and_rejects_invalid_protocol(tmp_path):
    runner = ProcessRunner(Store(tmp_path), 0)
    messages = []
    event = {"type":"metric", "data":{"step":1,"loss":2.5}}
    output = runner.run([sys.executable, "-c", f"print({'LOOP ' + json.dumps(event)!r})"],
                        cwd=tmp_path, log=tmp_path/"model.log", timeout=10, on_message=messages.append)
    assert messages == [event] and "LOOP " in output
    with pytest.raises(json.JSONDecodeError):
        runner.run([sys.executable, "-c", "print('LOOP invalid-json')"],
                   cwd=tmp_path, log=tmp_path/"bad-model.log", timeout=10, on_message=messages.append)
