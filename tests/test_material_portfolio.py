import copy

import pytest

from nekaise_loop.config import CampaignConfig
from nekaise_loop.material_portfolio import completed_portfolio, portfolio
from nekaise_loop.materials import validate_expansion_plan
from nekaise_loop.teaching import MaterialScopeMix


def package():
    rows = [
        {"id":"seed", "stream":"sft", "material_scope":"general_chat", "training_tokenization":"chat_response", "training_response":"Hello."},
        {"id":"author", "stream":"cpt", "material_scope":"general_prose", "material_origin":{"author_id":"writer"}},
        {"id":"domain", "stream":"sft", "material_scope":"domain"},
        {"id":"old", "stream":"replay"},
    ]
    samples = [{"row_id":r["id"], "stream":"replay" if r["stream"]=="replay" else "teacher", "input_ids":[1,2,3,4,5]} for r in rows]
    return {"rows":rows, "samples":samples, "ledger":{"total_tokens":16}}


def config(**updates):
    return {**CampaignConfig().model_dump(), "general_material_policy":"required_v1", "tokens_per_update":8, "train_epochs":2, **updates}


def test_scopes_are_separate_from_origin_and_repetitions():
    p = portfolio(package(), config(), {"general_chat":.38,"general_prose":.12,"domain":.5,"unspecified":0})
    assert p["prepared_targets"] == 16
    assert p["requested_exposure"] == 32
    assert p["requested_updates"] == 4
    assert p["by_scope"]["general_chat"]["prepared_share"] == .25
    assert p["by_scope"]["general_prose"]["requested_by_origin"]["author"] == 8
    assert p["by_scope"]["unspecified"]["prepared_by_origin"]["replay"] == 4
    done = completed_portfolio(p, {"tokens":32,"steps":4})
    assert done["status"] == "verified"
    assert done["by_scope"]["general_chat"] == 8


@pytest.mark.parametrize("origin", ["teacher", "author"])
def test_positive_round_cannot_omit_either_new_general_origin(origin):
    frozen = package()
    frozen["rows"][0 if origin=="teacher" else 1]["material_scope"] = "domain"
    with pytest.raises(ValueError, match="zero"):
        portfolio(frozen, config())


def test_general_prose_alone_does_not_satisfy_chat_requirement():
    frozen = package()
    frozen["rows"][0]["material_scope"] = "general_prose"
    with pytest.raises(ValueError, match="zero general-chat"):
        portfolio(frozen, config())


def test_step_override_checks_actual_scheduled_rows_not_just_prepared_mix():
    frozen = package()
    # Find a deterministic one-token prefix that misses the required general groups.
    with pytest.raises(ValueError, match="zero"):
        portfolio(frozen, config(tokens_per_update=1, train_steps=1))
    p = portfolio(frozen, config(train_steps=5))
    assert p["requested_exposure"] == 40
    assert p["requested_updates"] == 5


def test_zero_pass_diagnostic_and_historical_unknown_scope_remain_supported():
    frozen = package()
    for row in frozen["rows"]:
        row.pop("material_scope", None)
    p = portfolio(frozen, config(train_epochs=0))
    assert p["by_scope"]["unspecified"]["prepared_targets"] == 16
    assert p["requested_exposure"] == 0
    p = portfolio(frozen, config(general_material_policy="legacy_optional"))
    assert p["by_scope"]["unspecified"]["requested_exposure"] == 32


@pytest.mark.parametrize("change", [{"training_tokenization":"full_text"}, {"training_response":""}])
def test_general_chat_requires_native_serialization_and_answer(change):
    frozen = package()
    frozen["rows"][0].update(change)
    with pytest.raises(ValueError, match="native chat_response"):
        portfolio(frozen, config())


def test_counters_must_match_before_exposure_is_reported_completed():
    p = portfolio(package(), config())
    assert completed_portfolio(p, {})["status"] == "unverified"
    mismatch = completed_portfolio(p, {"tokens":31,"steps":4})
    assert mismatch == {"status":"mismatch", "reason":"Completed trainer counters differ from the material portfolio traversal",
                        "observed":{"tokens":31,"updates":4}, "expected":{"tokens":32,"updates":4}}


def test_unknown_and_duplicate_rows_do_not_fabricate_accounting():
    frozen = package()
    frozen["rows"].append(copy.deepcopy(frozen["rows"][0]))
    with pytest.raises(ValueError, match="unique"):
        portfolio(frozen, config())


def test_scope_plan_is_a_valid_share_vector_not_a_ratio_gate():
    with pytest.raises(ValueError, match="sum to one"):
        MaterialScopeMix(general_chat=.5,general_prose=.5,domain=.5)
    # Realized 25/25/25/25 above is accepted despite a different declared plan.
    assert portfolio(package(), config(), {"general_chat":.8,"domain":.2})["prepared_targets"] == 16


def test_new_campaign_defaults_required_but_explicit_legacy_is_preserved(setup_loop):
    _, service, _, _ = setup_loop
    c = service.create("New general scope", CampaignConfig())
    assert c["config"]["general_material_policy"] == "required_v1"
    c = service.create("Legacy", CampaignConfig(general_material_policy="legacy_optional"))
    assert c["config"]["general_material_policy"] == "legacy_optional"


def test_missing_general_plan_fails_before_paid_expansion():
    c = CampaignConfig(general_material_policy="required_v1", expansion_policy="legacy_optional")
    with pytest.raises(ValueError, match="Teacher-authored"):
        validate_expansion_plan(c,{"train_epochs":1,"lessons":[],"expansion_jobs":[]})


@pytest.mark.parametrize("review_policy", ["trusted_author_v1", "teacher_review_v1"])
def test_required_general_package_survives_expansion_freeze_training_and_replay(setup_loop, review_policy):
    import json
    from conftest import FakeModel
    from test_chat_serialization import ChatTokenizer
    from test_material_authors import AuthorTeacher, configured
    from nekaise_loop.training import prepare_dataset
    from nekaise_loop.teacher_tools import replay_lesson
    from nekaise_loop.learning_work import round_work

    class Teacher(AuthorTeacher):
        def curriculum(self, brief):
            result = super().curriculum(brief)
            result['lessons'][0].update(kind='sft', material_scope='general_chat', student_prompt='Hello?')
            result['lessons'][1]['material_scope'] = 'domain'
            result['expansion_jobs'][0]['material_scope'] = 'general_prose'
            result['expansion_jobs'][1]['material_scope'] = 'domain'
            result.update(token_mix={'teacher':1,'corpus':0,'replay':0}, train_epochs=1)
            return result

        def revise(self, lessons):
            result = super().revise(lessons)
            # The real Revision schema omits scope; the host must retain the plan's label.
            for row in result:
                row.pop('material_scope', None)
            result[0].update(training_tokenization='chat_response', training_text='', training_response='Hello! What would you like to discuss?')
            return result

    class Model(FakeModel):
        def prepare(self, checkpoint, rows):
            return prepare_dataset(rows, ChatTokenizer(), self.config.model_dump(), eos_ids=[4])

        def train(self, checkpoint, dataset, dataset_hash, on_metric):
            result = super().train(checkpoint, dataset, dataset_hash, on_metric)
            expected = dataset['material_portfolio']
            result['manifest'].update(tokens=expected['requested_exposure'], steps=expected['requested_updates'])
            from pathlib import Path
            (Path(result['checkpoint'])/'checkpoint.json').write_text(json.dumps(result['manifest']))
            return result

    settings, service, old, engine = configured(setup_loop, teacher=Teacher, review_policy=review_policy)
    campaign = service.create('Required scope fixture', CampaignConfig.model_validate({**old['config'], 'general_material_policy':'required_v1', 'train_steps':0}))
    engine.model_factory = Model
    engine.run(campaign['id'])
    assert service.store.campaign(campaign['id'])['status'] == 'complete'
    rid = service.store.one('SELECT id FROM rounds WHERE campaign_id=?',(campaign['id'],))['id']
    frozen = engine.artifacts.get(service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='freeze'",(rid,))['artifact'])
    assert frozen['material_portfolio']['by_scope']['general_chat']['prepared_by_origin']['teacher'] > 0
    assert frozen['material_portfolio']['by_scope']['general_prose']['prepared_by_origin']['author'] > 0
    assert replay_lesson(settings.workspace, rid, 'l1')['material_scope'] == 'general_chat'
    assert replay_lesson(settings.workspace, rid, 'material-one:variant')['material_scope'] == 'general_prose'
    assert round_work(service.store, engine.artifacts, rid)['material_portfolio']['completed_training']['status'] == 'verified'


@pytest.mark.parametrize("change", [{"training_tokenization":"full_text", "training_text":"Hello."}, {"training_response":""}])
def test_general_chat_author_validation_returns_actionable_retry_paths(change):
    import json
    from types import SimpleNamespace
    from nekaise_loop.material_jobs import candidates, CandidateValidationError
    row = {"id":"hello", "kind":"sft", "concept":"greeting", "student_prompt":"Hello?",
           "training_tokenization":"chat_response", "training_response":"Hello!", "rationale":"Short chat", **change}
    spec = {"sources":{}, "job":{"seed_ids":[], "material_scope":"general_chat"}}
    with pytest.raises(CandidateValidationError) as failure:
        candidates(SimpleNamespace(content=json.dumps({"rows":[row]}), complete=True), spec)
    assert failure.value.diagnostics == [{"path":["rows",0,"training_tokenization"], "type":"general_chat_requires_chat_response"}]


def test_general_chat_kind_is_checked_before_paid_expansion():
    with pytest.raises(ValueError, match="kind=sft"):
        validate_expansion_plan(CampaignConfig(), {"train_epochs":1, "lessons":[{"material_scope":"general_chat", "kind":"cpt"}]})
