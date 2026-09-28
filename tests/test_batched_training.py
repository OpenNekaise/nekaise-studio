import json

import pytest

from nekaise_loop.config import CampaignConfig
from nekaise_loop.training import recipe_hash, training_code_hash
from nekaise_loop.training_runtime import optimizer_transition, runtime_hash, SERIAL_PREDECESSOR
from nekaise_loop.workers.batched_training import microbatches


def test_physical_batches_do_not_change_update_order_or_recipe():
    rows = list(range(11))
    assert list(microbatches(rows, 4)) == [rows[:4], rows[4:8], rows[8:]]
    old = CampaignConfig().model_dump()
    new = {**old, "training_execution": "batched_v1", "training_microbatch_size": 4,
           "training_activation_checkpointing": False}
    assert recipe_hash(old) == recipe_hash(new)
    assert runtime_hash(old) == training_code_hash() == SERIAL_PREDECESSOR
    assert runtime_hash(new) != runtime_hash(old)


def test_optimizer_bridge_is_explicit_and_rejects_unknown_runtime_or_recipe():
    config = CampaignConfig(training_execution="batched_v1").model_dump()
    prior = {"recipe_hash": recipe_hash(config), "training_code": SERIAL_PREDECESSOR}
    assert optimizer_transition(prior, config)["policy"] == "serial_to_batched_v1"
    assert optimizer_transition({**prior, "training_code": runtime_hash(config)}, config)["policy"] == "identical_runtime"
    for wrong in ({"recipe_hash": "other"}, {"training_code": "unreviewed"}):
        with pytest.raises(ValueError):
            optimizer_transition({**prior, **wrong}, config)
    with pytest.raises(ValueError):
        optimizer_transition(prior, {**config, "tokens_per_update": 4096})


def test_continuation_records_bridge_without_reset_or_budget_renewal(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    stage = service.store.one("SELECT s.* FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='train' ORDER BY s.id DESC LIMIT 1", (campaign["id"],))
    trained = service.artifacts.get(stage["artifact"])
    trained["manifest"].update(recipe_hash=recipe_hash(campaign["config"]), training_code=SERIAL_PREDECESSOR)
    from pathlib import Path
    (Path(trained["checkpoint"]) / "checkpoint.json").write_text(json.dumps(trained["manifest"]))
    service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (service.artifacts.put(trained), stage["id"]))
    child = service.continue_campaign(campaign["id"], {"training_execution": "batched_v1", "training_microbatch_size": 4})
    assert child["config"]["inherit_optimizer"] is True
    assert child["teacher_budget_since"] == (campaign["teacher_budget_since"] or campaign["created_at"])
    assert service.artifacts.get(child["context_artifact"])["optimizer_transition"]["policy"] == "serial_to_batched_v1"
    from nekaise_loop.service import Conflict
    with pytest.raises(Conflict, match="recipe"):
        service.continue_campaign(campaign["id"], {"training_execution": "batched_v1", "tokens_per_update": 4096})
    trained["manifest"].update(training_code=runtime_hash(child["config"]), config=child["config"])
    (Path(trained["checkpoint"]) / "checkpoint.json").write_text(json.dumps(trained["manifest"]))
    service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (service.artifacts.put(trained), stage["id"]))
    with pytest.raises(Conflict, match="Unreviewed"):
        service.continue_campaign(campaign["id"], {"training_execution": "serial_v1"})


def test_profile_requires_quiescent_worker_and_bounded_work(setup_loop):
    from nekaise_loop.training_profile import run_profile
    settings, service, campaign, _ = setup_loop
    with pytest.raises(ValueError, match="bounded"):
        run_profile(settings, "missing", steps=1001)
    service.store.set_status(campaign["id"], "running")
    with pytest.raises(ValueError, match="Pause or stop"):
        run_profile(settings, "missing")


def test_profile_counts_useful_targets_separately_from_padding():
    from nekaise_loop.workers.training_profile import batch_statistics
    rows = [{"input_ids": [0] * n} for n in (512, 6, 127, 13, 62, 3, 37, 2, 9)]
    four, eight = (batch_statistics(rows, size) for size in (4, 8))
    assert four["forward_backward_calls"] == 3
    assert eight["forward_backward_calls"] == 2
    assert four["valid_targets"] == eight["valid_targets"] == 762
    assert four["unpadded_input_tokens"] == eight["unpadded_input_tokens"] == 771
    assert eight["padded_input_tokens"] > four["padded_input_tokens"]
    assert eight["max_physical_rows"] == 8


@pytest.mark.parametrize("equivalence_only", [False, True])
def test_headroom_profile_preserves_saved_recipe_and_official_history(setup_loop, monkeypatch, equivalence_only):
    from pathlib import Path
    from nekaise_loop.training_profile import run_profile
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    stage = service.store.one("SELECT s.* FROM stage_runs s JOIN rounds r ON r.id=s.round_id WHERE r.campaign_id=? AND s.stage='train' ORDER BY s.id DESC LIMIT 1", (campaign["id"],))
    trained = service.artifacts.get(stage["artifact"])
    trained["manifest"]["config"] = campaign["config"]
    (Path(trained["checkpoint"]) / "checkpoint.json").write_text(json.dumps(trained["manifest"]))
    service.store.execute("UPDATE stage_runs SET artifact=? WHERE id=?", (service.artifacts.put(trained), stage["id"]))
    before_rounds = service.store.query("SELECT * FROM rounds ORDER BY id")
    before_stages = service.store.query("SELECT * FROM stage_runs ORDER BY id")
    received = []
    def fake_run(self, argv, **kwargs):
        payload = json.loads(Path(argv[-1]).read_text())
        received.append(payload)
        kwargs["on_message"]({"type": "result", "data": {"kind": "test_fixture", "passed": True}})
    monkeypatch.setattr("nekaise_loop.training_profile.ProcessRunner.run", fake_run)
    monkeypatch.setattr("nekaise_loop.training_profile.signal.signal", lambda *args: None)
    directory = run_profile(settings, stage["round_id"], mixed=True, headroom=True, equivalence_only=equivalence_only)
    assert len(received) == (1 if equivalence_only else 4)
    for payload in received:
        assert recipe_hash(payload["config"]) == recipe_hash(campaign["config"])
        assert payload["config"]["inherit_optimizer"] is True
        assert payload["config"]["training_activation_checkpointing"] is False
        assert payload["profile_case"]["headroom"] is True
        assert "output" not in payload  # No deployable checkpoint output.
    evidence = json.loads((Path(directory) / "evidence.json").read_text())
    assert evidence["checkpoint_verified_after"] is True
    assert evidence["coverage_advanced"] is False
    assert service.store.query("SELECT * FROM rounds ORDER BY id") == before_rounds
    assert service.store.query("SELECT * FROM stage_runs ORDER BY id") == before_stages
