import copy
import json

import pytest

from nekaise_loop.author_config import AuthorPool, merge_pool
from nekaise_loop.config import CampaignConfig


def spec(name, **updates):
    return {"id": name, "label": name, "model": "fixture-model",
            "base_url": "https://fixture.invalid", **updates}


def pool():
    return AuthorPool.model_validate({"authors": [spec("existing", resource_pool="shared", options={"temperature": .3})],
        "resource_limits": {"shared": 2}, "concurrency": 2,
        "max_calls_per_round": 7, "max_output_tokens_per_round": 65536})


def test_append_preserves_authors_limits_order_and_input_objects():
    original = pool()
    before = original.model_dump()
    patch = {"authors": [spec("new", resource_pool="another")], "resource_limits": {"another": 3}}
    untouched = copy.deepcopy(patch)
    result = merge_pool(original, patch)
    assert [a.id for a in result.authors] == ["existing", "new"]
    assert result.authors[0] == original.authors[0]
    assert result.resource_limits == {"shared": 2, "another": 3}
    assert result.concurrency == 2 and result.max_calls_per_round == 7
    assert result.max_output_tokens_per_round == 65536
    assert original.model_dump() == before and patch == untouched
    assert merge_pool(result, patch) == result  # idempotent upsert, no duplicate registration


def test_named_updates_preserve_other_fields_and_replace_explicit_options():
    original = pool()
    result = merge_pool(original, {"authors": [{"id":"existing", "model":"fixture-new"}], "concurrency":3})
    assert result.authors[0] == original.authors[0].model_copy(update={"model":"fixture-new"})
    assert result.concurrency == 3 and result.max_calls_per_round == 7
    assert merge_pool(result, {"authors":[{"id":"existing", "options":{}}]}).authors[0].options == {}


@pytest.mark.parametrize("patch", [{}, {"authors": []}, {"resource_limits": {}}])
def test_omission_and_empty_collections_are_not_removal(patch):
    assert merge_pool(pool(), patch) == pool()


def test_deliberate_replacement_names_removed_id_and_retains_shared_limit():
    result = merge_pool(pool(), {"authors":[spec("replacement")]}, ["existing"])
    assert [a.id for a in result.authors] == ["replacement"]
    assert result.resource_limits == {"shared":2}
    assert merge_pool(pool(), {}, ["existing"]).authors == []


@pytest.mark.parametrize("patch,removals", [
    ({"authors":[spec("new"),spec("new")]}, []),
    ({}, ["missing"]), ({}, ["existing","existing"]),
    ({"authors":[{"id":"existing","label":"rename"}]}, ["existing"]),
    ({"authors":[{"id":"new"}]}, []),
    ({"authors":[spec("new", resource_pool="missing")]}, []),
    ({"authors":None}, []), ({"authors":[{}]}, []),
    ({"resource_limits":None}, []), ({"typo_budget":12}, []),
    ({}, "existing"), ({}, [False]),
])
def test_invalid_or_ambiguous_updates_fail_without_mutating_pool(patch, removals):
    original = pool()
    before = original.model_dump()
    with pytest.raises(ValueError):
        merge_pool(original, patch, removals)
    assert original.model_dump() == before


def test_many_registered_authors_do_not_expand_execution_budgets():
    original = pool()
    result = merge_pool(original, {"authors":[spec(f"author-{n}") for n in range(100)]})
    assert len(result.authors) == 101
    assert result.model_dump(exclude={"authors"}) == original.model_dump(exclude={"authors"})


def test_continuation_upserts_pool_and_records_explicit_removal_without_mutating_history(setup_loop):
    _, service, _, _ = setup_loop
    parent = service.create("Author fixture", CampaignConfig(material_authors=pool()))
    child = service.continue_campaign(parent["id"], {"material_authors":{"authors":[spec("new")]}})
    assert [a['id'] for a in child['config']['material_authors']['authors']] == ['existing','new']
    assert service.store.campaign(parent['id'])['config'] == parent['config']
    receipt = service.artifacts.get(child['context_artifact'])['material_author_update']
    assert receipt['added'] == ['new'] and receipt['updated'] == receipt['removed'] == []
    assert receipt['before_hash'] != receipt['after_hash']
    final = service.continue_campaign(child['id'], {'remove_material_author_ids':['existing']}, reason='Explicit fixture removal')
    assert [a['id'] for a in final['config']['material_authors']['authors']] == ['new']
    assert 'remove_material_author_ids' not in final['config']
    receipt = service.artifacts.get(final['context_artifact'])['material_author_update']
    assert receipt['removed'] == ['existing'] and receipt['reason'] == 'Explicit fixture removal'


@pytest.mark.parametrize("remove_existing", [False, True])
def test_orchestrator_partial_pool_update_preserves_existing_author_and_budget_epoch(setup_loop, remove_existing):
    from nekaise_loop.recovery import handle_recovery, apply_recovery
    from test_recovery import decision

    settings, service, _, _ = setup_loop
    parent = service.create('Recovery fixture', CampaignConfig(material_authors=pool()))
    service.store.set_status(parent['id'], 'paused')
    recovery_id = service.store.recover(parent['id'], 'status_review', 'Fixture additive author decision')
    updates = [{'field':'material_authors','value':json.dumps({'authors':[spec('new')]})}]
    if remove_existing:
        updates.append({'field':'remove_material_author_ids','value':json.dumps(['existing'])})
    proposal = decision('continue', updates)
    handle_recovery(settings, recovery_id, agent=lambda *args: proposal)
    apply_recovery(settings, recovery_id)
    child = service.store.campaign(service.store.one('SELECT continuation_id FROM recoveries WHERE id=?',(recovery_id,))['continuation_id'])
    assert [a['id'] for a in child['config']['material_authors']['authors']] == (['new'] if remove_existing else ['existing','new'])
    assert child['config']['material_authors']['max_calls_per_round'] == 7
    assert child['teacher_budget_since'] == parent['created_at']
    assert service.store.campaign(parent['id'])['config'] == parent['config']
