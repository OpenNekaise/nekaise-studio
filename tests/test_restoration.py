"""Fixture-only recovery branches: provenance, command ownership and resume integrity."""
import hashlib
import json

import pytest

from conftest import FakeModel, FakeTeacher
from test_checkpoint_comparison import input_comparison_setup
from nekaise_loop.engine import Engine
from nekaise_loop.recovery import RecoveryDecision, apply_recovery
from nekaise_loop.service import Conflict
from nekaise_loop.teacher_tools import latest_strategy


def branch_setup(setup_loop, tmp_path):
    settings, service, child, reference, snapshot, _ = input_comparison_setup(setup_loop, tmp_path)
    recovery_id = service.store.recover(child["id"], "status_review", "Fixture Base branch experiment")
    decision = RecoveryDecision.model_validate({
        "action": "continue", "reason": "Fixture Base branch experiment", "report": "Fixture evidence only.",
        "retry_seconds": 30, "history_review_after_rounds": 1, "run_retention": [], "log_removals": [],
        "config_updates": [{"field": "restore_base_from_round", "value": json.dumps(reference["id"])},
                           {"field": "inherit_optimizer", "value": "false"}],
    }).model_dump()
    service.store.execute("UPDATE recoveries SET status='decided',decision=? WHERE id=?", (json.dumps(decision), recovery_id))
    return settings, service, child, reference, snapshot, recovery_id, decision


def test_base_branch_queues_once_preserves_teaching_parent_and_resets_optimizer(setup_loop, tmp_path):
    settings, service, parent, reference, snapshot, recovery_id, _ = branch_setup(setup_loop, tmp_path)
    parent_before = service.store.campaign(parent["id"])
    prior_rounds = service.store.query("SELECT * FROM rounds")
    apply_recovery(settings, recovery_id)
    apply_recovery(settings, recovery_id)
    recovery = service.store.one("SELECT * FROM recoveries WHERE id=?", (recovery_id,))
    assert recovery["status"] == "resolved"
    branch = service.store.campaign(recovery["continuation_id"])
    assert branch["parent_campaign_id"] == parent["id"]
    assert branch["config"]["student_model"] == str(snapshot)
    assert branch["config"]["inherit_optimizer"] is False
    assert "restore_base_from_round" not in branch["config"]  # one-use selector
    assert service.store.campaign(parent["id"])["config"] == parent_before["config"]
    assert service.store.query("SELECT * FROM rounds") == prior_rounds
    assert latest_strategy(settings.workspace, branch["id"]) == latest_strategy(settings.workspace, parent["id"])
    context = service.artifacts.get(branch["context_artifact"])
    assert context["restoration"]["reference"]["round_id"] == reference["id"]
    assert context["restoration"]["retained_checkpoint"] == parent_before["config"]["student_model"]
    assert service.store.query("SELECT kind,actor FROM actions WHERE campaign_id=?", (branch["id"],)) == [{"kind": "start", "actor": "orchestrator"}]
    assert len(service.store.query("SELECT id FROM events WHERE campaign_id=? AND kind='checkpoint_restoration'", (branch["id"],))) == 1
    seen = []
    class Model(FakeModel):
        def train(self, checkpoint, *args):
            seen.append((checkpoint, self.config.inherit_optimizer))
            return super().train(checkpoint, *args)
    Engine(settings, FakeTeacher, Model).run(branch["id"])
    assert seen == [(str(snapshot), False)]
    continuation = service.continue_campaign(branch["id"])
    assert continuation["config"]["student_model"] != str(snapshot)
    assert "restoration" not in service.artifacts.get(continuation["context_artifact"])


@pytest.mark.parametrize("defect", ["missing", "incomplete", "parent", "artifact", "weights", "reset", "null"])
def test_invalid_branch_never_queues_a_child(setup_loop, tmp_path, defect):
    settings, service, parent, reference, snapshot, recovery_id, decision = branch_setup(setup_loop, tmp_path)
    if defect == "missing":
        decision["config_updates"][0]["value"] = '"unknown"'
    elif defect == "incomplete":
        service.store.execute("UPDATE rounds SET status='failed' WHERE id=?", (reference["id"],))
    elif defect == "parent":
        service.store.execute("UPDATE rounds SET model_before='foreign' WHERE id=?", (reference["id"],))
    elif defect == "artifact":
        key = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='train'", (reference["id"],))["artifact"]
        (service.artifacts.root/key[:2]/f"{key}.json").write_text('{}')
    elif defect == "weights":
        (snapshot/"model-00001.safetensors").write_bytes(b'CORRUPT FIXTURE')
    elif defect == "reset":
        decision["config_updates"][1]["value"] = 'true'
    else:
        decision["config_updates"][0]["value"] = 'null'
    service.store.execute("UPDATE recoveries SET decision=? WHERE id=?", (json.dumps(decision), recovery_id))
    apply_recovery(settings, recovery_id)
    assert not service.store.query("SELECT id FROM campaigns WHERE parent_campaign_id=?", (parent["id"],))
    assert not service.store.query("SELECT id FROM actions WHERE kind='start'")
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (recovery_id,))["status"] == "waiting"


@pytest.mark.parametrize("control", ["pause", "review"])
def test_new_operator_control_supersedes_branch(setup_loop, tmp_path, control):
    settings, service, parent, _, _, recovery_id, _ = branch_setup(setup_loop, tmp_path)
    service.store.execute("UPDATE campaigns SET config=? WHERE id=?", (json.dumps({**parent["config"], "auto_recover": True}), parent["id"]))
    service.action(parent["id"], control, spawn=False, reason="Fixture operator control")
    apply_recovery(settings, recovery_id)
    assert not service.store.query("SELECT id FROM campaigns WHERE parent_campaign_id=?", (parent["id"],))
    assert service.store.one("SELECT status FROM recoveries WHERE id=?", (recovery_id,))["status"] == "cancelled"


def test_branch_rejects_non_orchestrator_request(setup_loop, tmp_path):
    _, service, parent, reference, _, _, _ = branch_setup(setup_loop, tmp_path)
    with pytest.raises(Conflict, match="explicit orchestrator"):
        service.continue_campaign(parent["id"], {"restore_base_from_round": reference["id"], "inherit_optimizer": False})


@pytest.mark.parametrize("diagnostic_continuation", [False, True])
def test_snapshot_retargeting_after_selection_fails_before_model_work(setup_loop, tmp_path, diagnostic_continuation):
    settings, service, _, _, snapshot, recovery_id, _ = branch_setup(setup_loop, tmp_path)
    apply_recovery(settings, recovery_id)
    branch_id = service.store.one("SELECT continuation_id FROM recoveries WHERE id=?", (recovery_id,))["continuation_id"]
    if diagnostic_continuation:
        class DiagnosticTeacher(FakeTeacher):
            def curriculum(self, brief):
                return {**super().curriculum(brief), "train_epochs": 0, "token_mix": {"teacher": 0, "corpus": 0, "replay": 0}}
        Engine(settings, DiagnosticTeacher, FakeModel).run(branch_id)
        child = service.continue_campaign(branch_id)
        assert child["config"]["inherit_optimizer"] is False
        assert service.artifacts.get(child["context_artifact"])["restoration"]
        branch_id = child["id"]
    data = b'VALID BUT DIFFERENT FIXTURE WEIGHTS'
    blob = snapshot.parent.parent/"blobs"/hashlib.sha256(data).hexdigest()
    blob.write_bytes(data)
    (snapshot/"model-00001.safetensors").unlink()
    (snapshot/"model-00001.safetensors").symlink_to(blob)
    class NoModel(FakeModel):
        def __init__(self, *args):
            pytest.fail("Changed Base must fail before model initialization")
    Engine(settings, FakeTeacher, NoModel).run(branch_id)
    campaign = service.store.campaign(branch_id)
    assert campaign["status"] == "failed"
    assert "changed after branch selection" in campaign["error"]
