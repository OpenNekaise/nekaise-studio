"""One source tree may have training readers or a repair writer, never both."""
from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path

from .config import ROOT

LOCK_DIRECTORY = ROOT / "workspace"


def source_fingerprint():
    paths = sorted((ROOT / "src/nekaise_loop").rglob("*.py")) + sorted((ROOT / "prompts").glob("*.txt")) + [ROOT/"docs/COAPT.md"]
    return hashlib.sha256(b"".join(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() for p in paths)).hexdigest()


@contextmanager
def source_lock(exclusive=False):
    directory = LOCK_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "source.lock").open("a+") as handle:
        fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        yield


def locked(path: Path):
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True
