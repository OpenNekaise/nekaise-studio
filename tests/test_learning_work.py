"""Accounting fixtures are not student learning measurements."""
import hashlib
import json
from pathlib import Path

import pytest

from nekaise_loop.artifacts import verify_checkpoint
from nekaise_loop.learning_work import latest_work, prepared_coverage, round_work
from nekaise_loop.reports import current_status
from nekaise_loop.work_accounting import generation_work, preparation_work


def test_actual_work_uses_attempt_counters_not_sum_of_cumulative_metrics(setup_loop):
    _, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    rnd = service.store.one("SELECT id FROM rounds WHERE campaign_id=? ORDER BY number DESC LIMIT 1", (campaign["id"],))
    rid = rnd["id"]
    work = round_work(service.store, service.artifacts, rid)
    assert work["measured_work_all_attempts"] == {"tokens": 300, "updates": 3, "elapsed_seconds": 3}
    assert work["prepared_targets_per_pass"] == sum(work["prepared_target_coverage"][k] for k in ("prompt", "continuation", "unassigned"))
    assert work["timing_complete"] is True
    assert work["checkpoint_bytes_present"] == len(b"TEST FAKE WEIGHTS")
    service.store.execute("INSERT INTO metrics(round_id,attempt,step,data,created_at) VALUES(?,2,1,?,?)", (rid, json.dumps({"tokens": 1000, "elapsed_seconds": 2, "global_tokens": 99999999}), "2026-09-17T00:00:00+00:00"))
    service.store.execute("UPDATE stage_runs SET started_at=?,finished_at=? WHERE round_id=? AND stage='train'", ("2026-09-17T00:00:00+00:00", "2026-09-17T00:00:20+00:00", rid))
    service.store.execute("UPDATE stage_runs SET started_at=?,finished_at=? WHERE round_id=? AND stage='grade'", ("2026-09-17T00:00:20+00:00", "2026-09-17T00:00:50+00:00", rid))
    work = round_work(service.store, service.artifacts, rid)
    assert work["measured_work_all_attempts"] == {"tokens": 1300, "updates": 4, "elapsed_seconds": 5}
    assert work["stage_seconds"]["train"] == 20 and work["stage_seconds"]["grade"] == 30
    assert work["finished_stage_seconds"] == pytest.approx(sum(work["stage_seconds"].values()))
    assert work["teacher_stage_seconds"] == pytest.approx(sum(work["stage_seconds"][s] for s in ("select", "revise", "evaluate", "grade", "adapt")))
    assert work["update_work"]["updates_with_known_size"] == 4
    assert work["update_work"]["mean_targets"] == 325
    assert current_status(service)["learning_work"] == work


def test_preparation_exposes_anchor_truncation_repetition_and_partial_updates():
    from conftest import FakeTokenizer
    from nekaise_loop.config import CampaignConfig
    from nekaise_loop.training import prepare_dataset, update_batches
    config = CampaignConfig(token_mix={"teacher": .5, "corpus": .25, "replay": .25}, tokens_per_update=64, train_epochs=2).model_dump()
    data = prepare_dataset([{"id": "lesson", "stream": "sft", "text": "x"*39},
        {"id": "reading", "stream": "corpus", "text": "y"*299},
        {"id": "review", "stream": "replay", "text": "z"*9}], FakeTokenizer(), config)
    work = preparation_work(data, config)
    assert work["streams"]["corpus"] == {"available_targets": 300, "prepared_targets": 20, "unused_targets": 280, "repeated_targets": 0}
    assert work["streams"]["replay"]["repeated_targets"] == 10
    assert work["targets_per_pass"] == 80 and work["expected_target_exposure"] == 160
    assert work["expected_updates"] == 4 and work["expected_short_updates"] == 2
    assert work["expected_mean_update_fill"] == 160/256
    for epochs, override in [(0, 0), (0, 7), (2, 0), (1, 1), (1, 3), (1, 8)]:
        chosen = {**config, "train_epochs": epochs, "train_steps": override}
        estimate = preparation_work(data, chosen)
        batches = list(update_batches(data["samples"], 64, epochs, config["seed"], override)) if epochs else []
        sizes = [sum(len(row["input_ids"])-1 for row in batch) for batch in batches]
        assert estimate["expected_updates"] == len(sizes)
        assert estimate["expected_target_exposure"] == sum(sizes)
        assert estimate["expected_short_updates"] == sum(n < 64 for n in sizes)


def test_generation_accounting_deduplicates_shared_time_and_unchanged_pairs():
    batch = {"index": 0, "size": 2, "row_ids": ["a", "b"], "generation_seconds": 2.5}
    a = {"id": "a", "text": "", "tokens": 3, "generation_batch": batch,
         "generation_audit": {"tokens_checked": 3, "mismatches": [{"position": 0}], "finite_logits": True}, "stop_reason": "max_new_tokens"}
    b = {"id": "b", "tokens": 5, "generation_batch": batch}
    b["comparison"] = {"weights_changed": False, "generation": dict(b)}
    work = generation_work([("answer", 1, {"items": [a, b]})])
    assert work["answers"] == 2 and work["generated_tokens"] == 8
    assert work["batches"] == 1 and work["generation_seconds"] == 2.5
    assert work["audit"] == {"tokens_checked": 3, "mismatched_tokens": 1, "nonfinite_answers": 0,
                             "empty_answers": 1, "max_new_tokens_answers": 1}
    a["comparison"] = {"weights_changed": True, "generation": {**a, "generation_batch": {**batch, "size": 1, "row_ids": ["a"]}}}
    work = generation_work([("answer", 1, {"items": [a, b]})])
    assert work["answers"] == 3 and work["batches"] == 2 and work["generation_seconds"] == 5


def test_latest_work_follows_lineage_and_corrupt_evidence_does_not_block_reports(setup_loop):
    _, service, parent, engine = setup_loop
    engine.run(parent["id"])
    child = service.continue_campaign(parent["id"], {"rounds": 1})
    expected = latest_work(service.store, service.artifacts, parent["id"])
    assert latest_work(service.store, service.artifacts, child["id"])["id"] == expected["id"]
    stage = service.store.one("SELECT artifact FROM stage_runs WHERE round_id=? AND stage='freeze'", (expected["id"],))
    key = stage["artifact"]
    (service.artifacts.root/key[:2]/f"{key}.json").write_text('corrupt fixture')
    report = current_status(service, campaign_id=child["id"])
    assert "integrity failed" in report["learning_work"]["error"]
    assert report["campaign"]["id"] == child["id"]


def test_prepared_coverage_separates_known_prefixes_and_ambiguous_chunks():
    frozen = {"rows": [{"id": "shared", "stream": "sft", "serialization": {"prompt_token_ids": [1, 2, 3]}},
                       {"id": "shared", "stream": "replay", "serialization": {"prompt_token_ids": [1, 5, 6]}}],
              "samples": [{"row_id": "shared", "stream": "teacher", "input_ids": [1, 2, 3, 4, 0]},
                          {"row_id": "shared", "stream": "replay", "input_ids": [1, 5, 6, 8, 9, 0]},
                          {"row_id": "shared", "stream": "teacher", "input_ids": [9, 9, 0]},
                          {"row_id": "shared", "stream": "replay", "input_ids": [1, 5]}]}
    counts = prepared_coverage(frozen)
    assert counts == {"basis": "prepared_causal_targets_per_pass_not_consumed_exposure", "prompt": 4, "continuation": 5, "unassigned": 3}
    assert sum(counts[k] for k in ("prompt", "continuation", "unassigned")) == sum(len(s["input_ids"])-1 for s in frozen["samples"])


def test_inference_skips_optimizer_contents_but_resumption_and_missing_files_stay_strict(tmp_path, monkeypatch):
    files = {"model.safetensors": b"fixture weights", "training_state.pt": b"fixture optimizer"}
    manifest = {"files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    for name, data in files.items():
        (tmp_path/name).write_bytes(data)
    (tmp_path/"checkpoint.json").write_text(json.dumps(manifest))
    result = {"checkpoint": str(tmp_path), "manifest": manifest}
    opened, original = [], Path.open
    def tracked(path, *args, **kwargs):
        opened.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tracked)
    verify_checkpoint(result, require_optimizer=False)
    assert "model.safetensors" in opened and "training_state.pt" not in opened
    (tmp_path/"training_state.pt").write_bytes(b"corrupted fixture optimizer")
    verify_checkpoint(result, require_optimizer=False)
    with pytest.raises(ValueError, match="integrity failed"):
        verify_checkpoint(result)
    (tmp_path/"training_state.pt").unlink()
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(result, require_optimizer=False)
    (tmp_path/"training_state.pt").write_bytes(files["training_state.pt"])
    (tmp_path/"model.safetensors").write_bytes(b"corrupted fixture model")
    with pytest.raises(ValueError, match="integrity failed"):
        verify_checkpoint(result, require_optimizer=False)
