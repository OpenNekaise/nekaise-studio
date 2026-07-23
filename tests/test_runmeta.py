from __future__ import annotations

import hashlib

from studio.tools import runmeta


def test_file_fingerprints_are_content_based(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    assert runmeta.file_fingerprints(tmp_path, ["a.txt"]) == {
        "a.txt": hashlib.sha256(b"alpha").hexdigest()
    }


def test_environment_has_training_lock_packages():
    env = runmeta.environment()
    assert {"unsloth", "trl", "transformers", "torch"} <= set(env)
