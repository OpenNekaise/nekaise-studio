"""Identity wiring fixtures validate provenance, never learned model behavior."""
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import FakeModel, FakeTeacher
from nekaise_loop.config import CampaignConfig
from nekaise_loop.engine import Engine
from nekaise_loop.identity import StudentIdentity
from nekaise_loop.model_chat import latest_snapshot
from test_material_authors import configured, response


def identity(**changes):
    return StudentIdentity(name="Kai", version="0.0", developer="Nekaise", origin="Sweden",
        foundation_model="openbmb/MiniCPM5-1B-SFT", charter="Fixture charter: honest Swedish AI.\n", **changes)


@pytest.mark.parametrize("version", [0.0, "0", "00.0", "0.00", "0.0.1", "v0.0", "0.0\n"])
def test_version_requires_two_string_components(version):
    with pytest.raises(ValidationError):
        StudentIdentity.model_validate({**identity().model_dump(), "version": version})


def test_public_metadata_preserves_minor_version_and_hashes_exact_charter():
    profile = StudentIdentity.model_validate({**identity().model_dump(), "version": "0.10"})
    public = profile.public_metadata()
    assert public["display_name"] == "Kai 0.10" and "charter" not in public
    assert public["charter_sha256"] == hashlib.sha256(profile.charter.encode()).hexdigest()
    assert public["origin"] == "Sweden"


def test_continuation_preserves_history_and_requires_new_version_for_changed_contract(setup_loop):
    _, service, parent, engine = setup_loop
    engine.run(parent["id"])
    child = service.continue_campaign(parent["id"], {"student_identity": identity().model_dump()})
    assert service.store.campaign(parent["id"])["config"]["student_identity"] is None
    assert child["config"]["student_identity"] == identity().model_dump()
    count = len(service.list_campaigns())
    changed = {**child["config"]["student_identity"], "charter": "Revised fixture charter"}
    with pytest.raises(ValueError, match="requires a new"):
        service.continue_campaign(child["id"], {"student_identity": changed})
    assert len(service.list_campaigns()) == count
    revised = service.continue_campaign(child["id"], {"student_identity": {**changed, "version": "0.1"}})
    assert revised["config"]["student_identity"]["version"] == "0.1"
    assert service.store.campaign(child["id"])["config"]["student_identity"] == identity().model_dump()


def test_authors_receive_exact_immutable_student_identity(setup_loop):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        task = json.loads(body["messages"][1]["content"])["task"]
        seen.append(task["student_identity"])
        return response(request)
    settings, service, old, engine = configured(setup_loop, handler, review_policy="trusted_author_v1")
    profile = identity().model_dump()
    campaign = service.create("Identity fixture", CampaignConfig.model_validate({**old["config"], "student_identity": profile}))
    engine.run(campaign["id"])
    assert service.store.campaign(campaign["id"])["status"] == "complete"
    assert seen == [profile, profile]
    jobs = service.store.query("SELECT input_artifact FROM material_jobs j JOIN rounds r ON r.id=j.round_id WHERE r.campaign_id=?", (campaign["id"],))
    assert len(jobs) == 2
    assert all(service.artifacts.get(j["input_artifact"])["spec"]["student_identity"] == profile for j in jobs)


def test_chat_attribution_comes_from_weights_not_new_campaign_or_diagnostic(setup_loop):
    settings, service, original, _ = setup_loop
    class ManifestModel(FakeModel):
        def train(self, *args):
            result = super().train(*args)
            result["manifest"]["config"] = self.config.model_dump()
            Path(result["checkpoint"], "checkpoint.json").write_text(json.dumps(result["manifest"]))
            return result
    class DiagnosticTeacher(FakeTeacher):
        def curriculum(self, brief):
            plan = super().curriculum(brief)
            plan["train_epochs"] = 0
            return plan
    # Legacy checkpoint is still unattributed even when its child targets Kai.
    config = CampaignConfig.model_validate({**original["config"], "rounds": 1, "student_format": "chat_template"})
    parent = service.create("Legacy fixture", config)
    Engine(settings, FakeTeacher, ManifestModel).run(parent["id"])
    assert latest_snapshot(service)[0]["identity"] is None
    child = service.continue_campaign(parent["id"], {"student_identity": identity().model_dump()})
    service.store.execute("UPDATE campaigns SET status='queued' WHERE id=?", (child["id"],))
    assert latest_snapshot(service)[0]["identity"] is None
    Engine(settings, DiagnosticTeacher, ManifestModel).run(child["id"])
    assert latest_snapshot(service)[0]["identity"] is None
    trained = service.continue_campaign(child["id"])
    Engine(settings, FakeTeacher, ManifestModel).run(trained["id"])
    snapshot = latest_snapshot(service)[0]
    assert snapshot["identity"] == identity().public_metadata()
    # A future release target must not relabel the previous checkpoint either.
    new = service.continue_campaign(trained["id"], {"student_identity": {**identity().model_dump(), "version": "0.1"}})
    service.store.execute("UPDATE campaigns SET status='queued' WHERE id=?", (new["id"],))
    assert latest_snapshot(service)[0] == snapshot
