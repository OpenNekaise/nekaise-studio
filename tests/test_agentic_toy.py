"""R11 done-when: trajectory collection → outcome verifier → score, end to end on the toy
task — no training involved. Also pins the step cap and the rejection filter."""
from __future__ import annotations

import json

from gym import tasks as gym_tasks
from studio.stages.agentic import FROZEN, parse_action, run_episode
from studio.stages.agentic.collect import collect


def scripted_policy():
    """list → read trend.csv → compute the average from the observed content → answer."""
    def policy(obs: str, trajectory: list[dict]) -> str:
        if obs.startswith("timestamp,supply_temp_c"):
            vals = [float(l.split(",")[1]) for l in obs.splitlines()[1:] if "," in l]
            avg = round(sum(vals) / len(vals), 4)
            return json.dumps({"action": "answer", "text": str(avg)})
        if not trajectory:
            return '{"action": "list"}'
        return '{"action": "read", "path": "trend.csv"}'
    return policy


def test_episode_solves_toy_task(tmp_path):
    task = gym_tasks.load("toy_agentic", split="dev", n=1)[0]
    ep = run_episode(task, scripted_policy(), tmp_path / "ws")
    assert ep["solved"] and ep["score"] == 1.0
    assert ep["steps"] <= task.tags["max_steps"] <= FROZEN["max_steps"]


def test_step_cap_enforced(tmp_path):
    task = gym_tasks.load("toy_agentic", split="dev", n=1)[0]
    ep = run_episode(task, lambda obs, tr: '{"action": "list"}', tmp_path / "ws")
    assert not ep["solved"] and ep["steps"] <= FROZEN["max_steps"]


def test_collect_rejection_filter(tmp_path):
    rows = gym_tasks.load("toy_agentic", split="dev", n=3)
    out = tmp_path / "traj.jsonl"
    stats = collect(rows, scripted_policy(), out)
    assert stats["episodes"] == 3 and stats["solved"] == 3
    kept = [json.loads(l) for l in out.read_text().splitlines()]
    assert all(k["messages"][-1]["role"] == "assistant" for k in kept)

    stats = collect(rows, lambda obs, tr: '{"action": "list"}', tmp_path / "none.jsonl")
    assert stats["solved"] == 0                       # unsolved episodes are rejected


def test_parse_action_robust():
    assert parse_action('{"action": "read", "path": "x"}')["action"] == "read"
    assert parse_action("garbage")["action"] == "list"
    assert parse_action('{"no_action": 1}')["action"] == "list"
