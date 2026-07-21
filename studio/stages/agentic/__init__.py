"""Agentic stage — trajectory rejection SFT → short-horizon GRPO. Card row 5 (R11).

Episode protocol (the toy substrate until opennekaise grows a headless mode — see
docs/opennekaise-headless-issue.md):

    task.meta["workspace"] files are materialized into a scratch dir; the POLICY (an LLM
    or a scripted function) emits one JSON action per step:

        {"action": "list"}                      → observation: file listing
        {"action": "read", "path": "<file>"}    → observation: file content
        {"action": "answer", "text": "<final>"} → episode ends

    The OUTCOME (final answer) is scored by the task's gym verifier. Steps are capped at
    FROZEN["max_steps"]; there are no process rewards and no per-step supervision — the
    card forbids both.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from gym import tasks as gym_tasks
from gym.tasks import Task

FROZEN = {
    "max_steps": 15,                 # hard cap; tasks target 5–15 steps
    "reward": "outcome_only",
    "process_rewards": False,
    "dense_supervision": False,
}

_JSON = re.compile(r"\{.*\}", re.S)


def parse_action(text: str) -> dict:
    """First JSON object in the policy's reply; malformed → no-op 'list' (costs a step)."""
    m = _JSON.search(str(text))
    if not m:
        return {"action": "list"}
    try:
        act = json.loads(m.group(0))
        return act if isinstance(act, dict) and "action" in act else {"action": "list"}
    except json.JSONDecodeError:
        return {"action": "list"}


class WorkspaceEnv:
    def __init__(self, task: Task, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for rel, content in (task.meta.get("workspace") or {}).items():
            p = self.root / rel
            if not p.resolve().is_relative_to(self.root.resolve()):
                raise ValueError(f"workspace path escapes root: {rel}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    def observe(self, action: dict) -> str:
        kind = action.get("action")
        if kind == "list":
            return "\n".join(sorted(p.name for p in self.root.iterdir()))
        if kind == "read":
            p = (self.root / str(action.get("path", ""))).resolve()
            if not p.is_relative_to(self.root.resolve()) or not p.is_file():
                return f"ERROR: no such file: {action.get('path')}"
            return p.read_text()[:8000]
        return f"ERROR: unknown action {kind!r} (use list/read/answer)"


def run_episode(task: Task, policy, root: Path,
                max_steps: int | None = None) -> dict:
    """Roll one episode; returns trajectory + outcome score (gym verifier)."""
    cap = min(int(task.tags.get("max_steps", FROZEN["max_steps"])),
              max_steps or FROZEN["max_steps"], FROZEN["max_steps"])
    env = WorkspaceEnv(task, root)
    trajectory, answer = [], ""
    obs = f"TASK: {task.prompt}\n(actions: " \
          '{"action":"list"} | {"action":"read","path":...} | {"action":"answer","text":...})'
    for step in range(cap):
        reply = policy(obs, trajectory)
        action = parse_action(reply)
        trajectory.append({"observation": obs, "reply": reply, "action": action})
        if action["action"] == "answer":
            answer = str(action.get("text", ""))
            break
        obs = env.observe(action)
    score = gym_tasks.score(task, answer) if answer else 0.0
    return {"task_id": task.id, "trajectory": trajectory, "answer": answer,
            "steps": len(trajectory), "score": score, "solved": score >= 0.999}
