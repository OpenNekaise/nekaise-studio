"""Accounting fixtures, not student training results."""
import pytest
from nekaise_loop.efficiency import add_call, efficiency, usage_summary
from nekaise_loop.learning_work import round_work
from nekaise_loop.storage import encode, now
from nekaise_loop.telemetry import training_telemetry


def test_missing_pending_zero_and_reasoning_do_not_inflate_complete_ratio():
    usage = usage_summary()
    assert efficiency(100, usage)["ratio"] is None
    add_call(usage, {"status": "failed"}, {"input": 100, "output": 20, "cached": 80})
    result = efficiency(60, usage)
    assert result["ratio"] == .5 and result["teacher_tokens"] == 120
    add_call(usage, {"status": "failed"}, None)
    assert efficiency(60, usage)["ratio"] is None
    assert efficiency(60, usage)["reported_ratio"] == .5
    assert efficiency(60, usage)["status"] == "missing"
    add_call(usage, {"status": "running"}, {"input": 50, "output": 10, "cached": 0})
    assert efficiency(60, usage)["status"] == "pending"
    assert efficiency(60, usage)["ratio"] is None
    assert efficiency(60, usage)["teacher_tokens"] == 180


def test_teacher_feedback_and_plot_share_scope_and_ratio_of_sums(setup_loop):
    _, service, campaign, engine = setup_loop
    engine.run(campaign["id"])
    rounds = service.store.query("SELECT id FROM rounds WHERE campaign_id=? ORDER BY number", (campaign["id"],))
    for r, tokens in zip(rounds, (100, 200)):
        service.store.execute("INSERT INTO teacher_calls(campaign_id,round_id,purpose,status,usage,created_at) VALUES(?,?,?,'complete',?,?)", (campaign["id"], r["id"], "reflect", encode({"input_tokens": tokens, "output_tokens": 0}), now()))
    t = training_telemetry(service, campaign["id"])
    assert t["training"]["total"] == 600 and t["teacher"]["total"] == 300
    assert t["efficiency"]["ratio"] == 2  # not the mean of 3 and 1.5
    assert [p["ratio"] for p in t["efficiency"]["series"]] == [3, 1.5]
    for r, point in zip(rounds, t["efficiency"]["series"]):
        feedback = round_work(service.store, service.artifacts, r["id"])["teacher_efficiency"]
        assert feedback["ratio"] == point["ratio"] and feedback["final"]
    service.store.execute("INSERT INTO metrics(round_id,attempt,step,data,created_at) VALUES(?,2,1,?,?)", (rounds[0]["id"], encode({"tokens": 50, "global_tokens": 999999}), now()))
    assert training_telemetry(service, campaign["id"])["efficiency"]["ratio"] == pytest.approx(650/300)
    service.store.execute("INSERT INTO teacher_calls(campaign_id,round_id,purpose,status,usage,created_at) VALUES(?,?,?,'failed','{}',?)", (campaign["id"], rounds[0]["id"], "revise", now()))
    t = training_telemetry(service, campaign["id"])
    assert t["efficiency"]["ratio"] is None and t["efficiency"]["missing_observations"] == 1
    assert t["efficiency"]["series"][0]["ratio"] is None
    assert t["efficiency"]["series"][1]["ratio"] == 1.5
    assert t["efficiency"]["reported_ratio"] == pytest.approx(t["training"]["total"]/t["teacher"]["total"])
    service.store.execute("INSERT INTO teacher_calls(campaign_id,round_id,purpose,status,usage,created_at) VALUES(?,?,?,'running',?,?)", (campaign["id"], rounds[1]["id"], "reflect", encode({"input_tokens": 100, "output_tokens": 0}), now()))
    t = training_telemetry(service, campaign["id"])
    assert t["teacher"]["pending_calls"] == t["efficiency"]["teacher_usage"]["pending_calls"] == 1


def test_diagnostic_cost_counts_and_running_round_is_not_a_finished_observation(setup_loop):
    from nekaise_loop.efficiency import efficiency_history
    usage = usage_summary()
    add_call(usage, {"status": "complete"}, {"input": 100, "output": 10, "cached": 0})
    row = {"id": "a", "campaign_id": "c", "number": 1, "status": "complete", "updated_at": now()}
    series = efficiency_history([row, {**row, "id": "b", "status": "running"}], {"a": usage}, {})
    assert len(series) == 1 and series[0]["ratio"] == 0
    assert series[0]["teacher_tokens"] == 110


def test_efficiency_observations_do_not_cross_ancestor_branch_boundary(setup_loop):
    from nekaise_loop.config import CampaignConfig
    _, service, parent, engine = setup_loop
    engine.run(parent["id"])
    child = service.create("Fixture branch", CampaignConfig.model_validate(parent["config"]), parent_id=parent["id"])
    parent_round = service.store.one("SELECT id FROM rounds WHERE campaign_id=? ORDER BY number DESC LIMIT 1", (parent["id"],))
    service.store.execute("UPDATE rounds SET updated_at='2099-01-01T00:00:00+00:00' WHERE id=?", (parent_round["id"],))
    result = training_telemetry(service, child["id"])
    assert result["efficiency"]["completed_observations"] == 1
    assert all(p["round_id"] != parent_round["id"] for p in result["efficiency"]["series"])
