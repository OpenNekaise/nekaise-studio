import os
import subprocess
import sys
import time

import pytest

from nekaise_loop.processes import stop_child, stop_owned, process_start


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
