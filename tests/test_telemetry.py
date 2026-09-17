import json

from fastapi.testclient import TestClient

from nekaise_loop.api import create_app
from nekaise_loop.config import CampaignConfig
from nekaise_loop.storage import encode, now
from nekaise_loop.telemetry import call_usage, normalize_usage, sampled, training_telemetry


def test_usage_counts_cache_and_reasoning_once():
    assert normalize_usage({"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20, "reasoning_output_tokens": 8}) == {"input": 100, "output": 20, "cached": 80}
    assert normalize_usage({"models": {"fixture": {"inputTokens": 10, "outputTokens": 3, "cacheReadInputTokens": 20, "cacheCreationInputTokens": 5}}}) == {"input": 35, "output": 3, "cached": 20}
    assert normalize_usage({"cost_usd": None}) is None
    assert normalize_usage([]) is None
    assert normalize_usage({"models": {"bad": None}}) is None
    assert normalize_usage({"input_tokens": -1, "output_tokens": 5}) is None


def test_legacy_log_usage_refreshes_and_remains_available_after_cleanup(setup_loop):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    rid = service.snapshot(campaign["id"])["round"]["id"]
    path = settings.workspace/"runs"/rid/"select-1/teacher-999/provider.log"
    path.parent.mkdir(parents=True)
    call = {"id": 999, "round_id": rid, "usage": "{}"}
    path.write_text(json.dumps({"type": "item.completed", "usage": {"input_tokens": 9999, "output_tokens": 9999}})+"\n")
    assert call_usage(service, call)[0] is None
    with path.open("a") as f:
        f.write(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 80}})+"\n")
    assert call_usage(service, call)[0] == {"input": 100, "output": 20, "cached": 80}
    recovery_id = service.store.recover(campaign["id"], "history_review", "fixture review")
    relative = str(path.relative_to(settings.workspace))
    trash = settings.workspace/"trash/history"/str(recovery_id)/relative
    trash.parent.mkdir(parents=True);path.rename(trash)
    service.store.execute("INSERT INTO log_cleanup(recovery_id,path,sha256,summary,reason,bytes,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'removed',?,?)", (recovery_id, relative, "fixture", "fixture", "fixture", trash.stat().st_size, now(), now()))
    assert call_usage(service, call)[0]["input"] == 100


def test_lineage_totals_include_actual_retries_without_recounting_cumulative_steps(setup_loop):
    settings, service, parent, engine = setup_loop
    engine.run(parent["id"])
    rid = service.snapshot(parent["id"])["round"]["id"]
    for step, tokens in [(1, 10), (2, 30)]:
        service.store.execute("INSERT INTO metrics(round_id,attempt,step,data,created_at) VALUES(?,?,?,?,?)", (rid, 2, step, encode({"tokens": tokens, "global_tokens": 999999}), now()))
    service.store.execute("INSERT INTO teacher_calls(campaign_id,round_id,purpose,status,usage,created_at) VALUES(?,?,?,'failed',?,?)", (parent["id"], rid, "revise", encode({"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 80}), now()))
    child = service.create("Continuation fixture", CampaignConfig.model_validate(parent["config"]), parent_id=parent["id"])
    engine.run(child["id"])
    t = training_telemetry(service, child["id"])
    assert t["campaign_count"] == 2 and t["completed_rounds"] == 4
    assert t["training"]["total"] == 1230  # 300 per round, plus 30 actually consumed on retry
    assert t["training"]["updates"] == 14
    assert t["training"]["series"][-1]["tokens"] == 1230
    assert t["teacher"]["total"] == 120 and t["teacher"]["cached"] == 80
    assert t["teacher"]["series"][-1]["tokens"] == 120
    assert not t["clock_running"]


def test_elapsed_excludes_explicit_pause_and_carries_through_continuation(setup_loop, monkeypatch):
    _, service, parent, _ = setup_loop
    parent_id = parent["id"]
    service.store.execute("DELETE FROM events WHERE campaign_id=?", (parent_id,))
    service.store.execute("UPDATE campaigns SET created_at='2026-09-15T00:00:00+00:00',status='paused',updated_at='2026-09-15T00:00:40+00:00' WHERE id=?", (parent_id,))
    child = service.create("Continuation fixture", CampaignConfig.model_validate(parent["config"]), parent_id=parent_id)
    service.store.execute("DELETE FROM events WHERE campaign_id=?", (child["id"],))
    service.store.execute("UPDATE campaigns SET created_at='2026-09-15T00:01:00+00:00',status='running' WHERE id=?", (child["id"],))
    for cid, seconds, status in [(parent_id, 0, "running"), (parent_id, 10, "paused"), (parent_id, 30, "running"), (parent_id, 40, "paused"), (child["id"], 60, "running")]:
        stamp = f"2026-09-15T00:{seconds//60:02d}:{seconds%60:02d}+00:00"
        service.store.execute("INSERT INTO events(campaign_id,kind,message,data,created_at) VALUES(?,'campaign','fixture',?,?)", (cid, encode({"status": status}), stamp))
    monkeypatch.setattr("nekaise_loop.telemetry.now", lambda: "2026-09-15T00:01:10+00:00")
    t = training_telemetry(service, child["id"])
    assert t["elapsed_seconds"] == 30 and t["clock_running"]


def test_unreported_usage_is_unknown_and_read_api_does_not_launch_work(setup_loop, monkeypatch):
    settings, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    rid = service.snapshot(campaign["id"])["round"]["id"]
    service.store.execute("INSERT INTO teacher_calls(campaign_id,round_id,purpose,status,usage,created_at) VALUES(?,?,?,'running','{}',?)", (campaign["id"], rid, "revise", now()))
    monkeypatch.setattr("nekaise_loop.service.Service.ensure_worker", lambda *args: None)
    before = service.store.query("SELECT * FROM actions")
    with TestClient(create_app(settings)) as client:
        response = client.get(f"/api/campaigns/{campaign['id']}/telemetry")
        assert response.status_code == 200
        t = response.json()
        assert t["teacher"]["total"] is None and t["teacher"]["pending_calls"] == 1
        assert t["teacher"]["series"] == []
        assert client.get("/api/campaigns/missing/telemetry").status_code == 404
    assert service.store.query("SELECT * FROM actions") == before


def test_bounded_series_preserves_initial_and_final_totals():
    points = [{"tokens": i} for i in range(1000)]
    reduced = sampled(points)
    assert len(reduced) == 240 and reduced[0] == points[0] and reduced[-1] == points[-1]
