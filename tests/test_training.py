from collections import Counter

import pytest

from conftest import FakeTokenizer
from nekaise_loop.config import CampaignConfig
from nekaise_loop.training import prepare_dataset, update_batches


def material(stream, text):
    return {"id": stream, "stream": stream, "text": text}


def test_default_recipe_and_unlimited_settings():
    config = CampaignConfig()
    assert (config.rounds, config.max_teacher_calls, config.train_steps) == (-1, -1, 0)
    assert config.teacher_model != config.orchestrator_model
    for field in ("rounds", "max_teacher_calls"):
        for invalid in (0, -2):
            with pytest.raises(ValueError):
                CampaignConfig(**{field: invalid})
    with pytest.raises(ValueError):
        CampaignConfig(token_mix={"teacher": .6, "corpus": .3, "replay": .2})


def test_mix_counts_causal_targets_and_does_not_repeat_teacher():
    # BOS + 59 characters + EOS gives 60 causal targets.
    rows = [material("cpt", "x"*59), material("corpus", "y"*299), material("replay", "z"*299)]
    result = prepare_dataset(rows, FakeTokenizer(), CampaignConfig().model_dump())
    ledger = result["ledger"]
    assert {k: v["prepared_tokens"] for k,v in ledger["streams"].items()} == {"teacher": 60, "corpus": 20, "replay": 20}
    assert ledger["total_tokens"] == 100
    assert ledger["streams"]["teacher"]["repeated_tokens"] == 0
    assert sum(len(x["input_ids"])-1 for x in result["samples"]) == 100
    duplicate = prepare_dataset(rows + [rows[0]], FakeTokenizer(), CampaignConfig().model_dump())
    assert duplicate["ledger"]["streams"]["teacher"]["prepared_tokens"] == 120
    assert duplicate["ledger"]["total_tokens"] == 200


def test_cold_start_renormalizes_and_records_missing_replay():
    rows = [material("cpt", "x"*59), material("corpus", "y"*299)]
    ledger = prepare_dataset(rows, FakeTokenizer(), CampaignConfig().model_dump())["ledger"]
    assert ledger["missing_streams"] == ["replay"]
    assert ledger["streams"]["teacher"]["prepared_tokens"] == 60
    assert ledger["streams"]["corpus"]["prepared_tokens"] == 20
    assert ledger["total_tokens"] == 80


def test_accumulation_preserves_each_causal_transition_and_short_final_update():
    samples = [{"row_id": "a", "stream": "teacher", "input_ids": list(range(12))}, {"row_id": "b", "stream": "corpus", "input_ids": list(range(20,27))}]
    batches = list(update_batches(samples, 8, 2, 42))
    assert [sum(len(s["input_ids"])-1 for s in batch) for batch in batches] == [8,8,1,8,8,1]
    def transitions(rows):
        return Counter((r["row_id"], a,b) for r in rows for a,b in zip(r["input_ids"], r["input_ids"][1:]))
    expected = transitions(samples)
    assert transitions([r for batch in batches for r in batch]) == expected + expected
    assert len(list(update_batches(samples, 8, 1, 42, step_limit=7))) == 7
