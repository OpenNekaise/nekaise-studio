"""Resume / checkpoint guards for multi-week CPT runs (CPU-only, no torch).

A resume must be refused on ANY missing or mismatched identity input (never fail-open),
a torn Trainer save must never be selected as the resume point, and the wall-clock box
must be one clock for both the trainer callback and the recorded verdict.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from studio.stages import _common, cpt


def _zip_bytes() -> bytes:
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.pkl", b"x" * 64)
    return buffer.getvalue()


def _safetensors_bytes() -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    return len(header).to_bytes(8, "little") + header + b"\x00" * 8


def _write_checkpoint(directory: Path, step: int, *, sharded: bool = False,
                      torn: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (directory / name).write_bytes(_zip_bytes())
    if sharded:
        (directory / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors"}}))
        (directory / "model-00001-of-00002.safetensors").write_bytes(_safetensors_bytes())
        (directory / "model-00002-of-00002.safetensors").write_bytes(_safetensors_bytes())
    else:
        (directory / "model.safetensors").write_bytes(_safetensors_bytes())
    if torn:                                   # a save cut short: half the bytes
        payload = (directory / torn).read_bytes()
        (directory / torn).write_bytes(payload[: len(payload) // 2])
    return directory


def test_complete_checkpoint_accepts_intact_saves(tmp_path):
    assert cpt._complete_checkpoint(_write_checkpoint(tmp_path / "checkpoint-10", 10))
    assert cpt._complete_checkpoint(
        _write_checkpoint(tmp_path / "checkpoint-20", 20, sharded=True))


@pytest.mark.parametrize("torn", ["optimizer.pt", "scheduler.pt", "rng_state.pth",
                                  "model.safetensors"])
def test_complete_checkpoint_rejects_zero_length_state(tmp_path, torn):
    assert not cpt._complete_checkpoint(
        _write_checkpoint(tmp_path / "checkpoint-10", 10, torn=torn))


def test_complete_checkpoint_rejects_missing_shard_and_step_mismatch(tmp_path):
    d = _write_checkpoint(tmp_path / "checkpoint-10", 10, sharded=True)
    (d / "model-00002-of-00002.safetensors").unlink()
    assert not cpt._complete_checkpoint(d)
    assert not cpt._complete_checkpoint(_write_checkpoint(tmp_path / "checkpoint-11", 12))


def test_complete_checkpoint_rejects_orphan_shards_and_wrong_rng_name(tmp_path):
    d = _write_checkpoint(tmp_path / "checkpoint-10", 10, sharded=True)
    (d / "model.safetensors.index.json").unlink()        # shards without their index
    assert not cpt._complete_checkpoint(d)
    e = _write_checkpoint(tmp_path / "checkpoint-20", 20)
    (e / "rng_state.pth").rename(e / "rng_state_0.pth")   # multi-process name, 1 GPU
    assert not cpt._complete_checkpoint(e)


def test_safetensors_size_must_match_header(tmp_path):
    good = tmp_path / "model.safetensors"
    good.write_bytes(_safetensors_bytes())
    assert cpt._valid_safetensors(good)
    (tmp_path / "long.safetensors").write_bytes(_safetensors_bytes() + b"\x00")
    assert not cpt._valid_safetensors(tmp_path / "long.safetensors")
    (tmp_path / "junk.safetensors").write_bytes(b"\xff" * 32)
    assert not cpt._valid_safetensors(tmp_path / "junk.safetensors")


def test_last_complete_checkpoint_skips_torn_newest(tmp_path):
    trainer = tmp_path / "_trainer"
    good = _write_checkpoint(trainer / "checkpoint-2000", 2000)
    _write_checkpoint(trainer / "checkpoint-4000", 4000, torn="optimizer.pt")
    assert cpt._last_complete_checkpoint(trainer) == good


class _Store:
    def __init__(self, run: dict, root: Path):
        self.run, self.root = run, root

    def get_run(self, run_id):
        return self.run

    def run_dir(self, experiment, run_id):
        return self.root


def _prior(fingerprints, environment, **overrides) -> dict:
    run = {"dataset_id": "abc", "seed": 3407, "config_sha256": "cfg", "experiment": "cpt",
           "status": "running", "environment": dict(environment), "model_revision": "rev1",
           "metadata": {"file_fingerprints": dict(fingerprints)}}
    run.update(overrides)
    return run


@pytest.fixture
def identity():
    fingerprints = {name: f"fp-{name}" for name in cpt.RESUME_STRICT_FILES}
    environment = {pkg: "1.0" for pkg in cpt.RESUME_STRICT_PACKAGES}
    return fingerprints, environment


def _attempt(tmp_path, prior, fingerprints, environment):
    _write_checkpoint(tmp_path / "scratch" / "_trainer" / "checkpoint-10", 10)
    logger = types.SimpleNamespace(store=_Store(prior, tmp_path))
    return cpt._resume_checkpoint(
        logger, "run-0", dataset_id="abc", config_sha256="cfg", seed=3407,
        experiment="cpt", fingerprints=fingerprints, environment=environment)


def test_resume_accepts_identical_identity(tmp_path, identity):
    fingerprints, environment = identity
    path, revision = _attempt(tmp_path, _prior(fingerprints, environment),
                              fingerprints, environment)
    assert path.endswith("checkpoint-10") and revision == "rev1"


def test_resume_refuses_missing_fingerprint_either_side(tmp_path, identity):
    fingerprints, environment = identity
    prior = _prior(fingerprints, environment)
    del prior["metadata"]["file_fingerprints"]["studio/stages/cpt.py"]
    with pytest.raises(SystemExit, match="fingerprint missing on the interrupted run"):
        _attempt(tmp_path, prior, fingerprints, environment)
    current = {k: v for k, v in fingerprints.items() if k != "lib/trainkit.py"}
    with pytest.raises(SystemExit, match="fingerprint missing on this invocation"):
        _attempt(tmp_path, _prior(fingerprints, environment), current, environment)


def test_resume_refuses_changed_code_environment_or_finished_run(tmp_path, identity):
    fingerprints, environment = identity
    changed = {**fingerprints, "studio/stages/cpt.py": "other"}
    with pytest.raises(SystemExit, match="studio/stages/cpt.py changed"):
        _attempt(tmp_path, _prior(fingerprints, environment), changed, environment)
    with pytest.raises(SystemExit, match="accelerate 1.0 != 1.1"):
        _attempt(tmp_path, _prior(fingerprints, environment), fingerprints,
                 {**environment, "accelerate": "1.1"})
    with pytest.raises(SystemExit, match="version missing"):
        _attempt(tmp_path, _prior(fingerprints, {**environment, "torch": ""}),
                 fingerprints, environment)
    with pytest.raises(SystemExit, match="finished training"):
        _attempt(tmp_path, _prior(fingerprints, environment, status="trained"),
                 fingerprints, environment)


def test_resume_refuses_while_prior_trainer_is_alive(tmp_path, identity, monkeypatch):
    fingerprints, environment = identity
    monkeypatch.setattr(cpt, "_prior_process_alive", lambda prior: True)
    with pytest.raises(SystemExit, match="still alive"):
        _attempt(tmp_path, _prior(fingerprints, environment, pid=12345), fingerprints,
                 environment)


def test_prior_process_alive_reads_this_host_cmdline():
    import os
    import socket
    me = {"pid": os.getpid(), "host": socket.gethostname()}
    assert cpt._prior_process_alive(me) is False        # pytest is not a CPT trainer
    assert cpt._prior_process_alive({**me, "host": "elsewhere"}) is False
    assert cpt._prior_process_alive({"pid": None, "host": socket.gethostname()}) is False


def test_resume_refuses_without_complete_checkpoint(tmp_path, identity):
    fingerprints, environment = identity
    _write_checkpoint(tmp_path / "scratch" / "_trainer" / "checkpoint-10", 10,
                      torn="scheduler.pt")
    logger = types.SimpleNamespace(store=_Store(_prior(fingerprints, environment), tmp_path))
    with pytest.raises(SystemExit, match="no complete Trainer checkpoint"):
        cpt._resume_checkpoint(
            logger, "run-0", dataset_id="abc", config_sha256="cfg", seed=3407,
            experiment="cpt", fingerprints=fingerprints, environment=environment)


class _Callback:
    """Stands in for trainkit's wall-clock callback; its own verdict must be ignored."""
    def on_step_end(self, args, state, control, **kwargs):
        control.should_training_stop = True      # a clock jump would stop training here
        return control


def _stub_trainkit(monkeypatch, seen):
    stub = types.ModuleType("trainkit")

    def time_budget_callback(minutes):
        seen["minutes"] = minutes
        return _Callback()
    stub.time_budget_callback = time_budget_callback
    stub.save_checkpoint = lambda *a, **k: {"digest": "d", "path": "p"}
    monkeypatch.setitem(sys.modules, "trainkit", stub)
    return stub


def test_budget_callback_shares_the_budget_clock(monkeypatch):
    seen = {}
    _stub_trainkit(monkeypatch, seen)
    budget = _common.Budget(100)
    budget._m0 -= 30 * 60           # thirty minutes already spent (model load, packing)
    budget.callback()
    assert 69.9 < seen["minutes"] < 70.1


def test_budget_verdict_is_the_callback_firing_not_the_clock(monkeypatch):
    _stub_trainkit(monkeypatch, {})
    budget = _common.Budget(100)
    callback = budget.callback()
    control = types.SimpleNamespace(should_training_stop=False, should_save=False)
    callback.on_step_end(None, None, control)
    assert not budget.fired and not budget.exceeded and not control.should_save
    assert not control.should_training_stop      # the wall-clock verdict is not chained
    budget._m0 -= 101 * 60          # the box passes while training is still stepping
    callback.on_step_end(None, None, control)
    assert budget.fired and budget.exceeded
    assert control.should_training_stop and control.should_save   # full save requested
    late = _common.Budget(100)
    late.callback()
    late._m0 -= 101 * 60            # the box passes only after training finished
    assert not late.exceeded


def test_finish_stage_never_publishes_a_timeboxed_run_as_latest(monkeypatch, tmp_path):
    _stub_trainkit(monkeypatch, {})
    pointers = []
    store = types.SimpleNamespace(set_pointer=lambda *a: pointers.append(a))
    logger = types.SimpleNamespace(store=store, run_id="run-1")
    budget = _common.Budget(100)
    budget.callback()
    budget.fired = True
    artifact = _common.finish_stage(exp_dir=tmp_path / "cpt", stage="cpt", metric="m",
                                    value=None, cfg={}, budget=budget, run_logger=logger,
                                    model=object(), tok=object())
    assert artifact["digest"] == "d" and pointers == []
    budget.fired = False
    _common.finish_stage(exp_dir=tmp_path / "cpt", stage="cpt", metric="m", value=None,
                         cfg={}, budget=budget, run_logger=logger, model=object(),
                         tok=object())
    assert pointers and pointers[0][1] == "latest"
