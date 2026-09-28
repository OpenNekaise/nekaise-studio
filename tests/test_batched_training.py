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
