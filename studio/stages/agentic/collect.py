"""Trajectory collection → rejection SFT dataset (agentic stage, phase A).

    python -m studio.stages.agentic.collect --config configs/agentic.yaml \
        --policy server:<model>@<base_url> --out experiments/<exp>/data/trajectories.jsonl

Runs episodes over the configured task set, KEEPS only outcome-solved ones (rejection on
the verifier score — outcome only, per the card), and writes chat-format SFT rows ready
for the sft stage (`data.sources: [{jsonl: ...}]`). The policy is any OpenAI-compatible
endpoint via gym.runner (no transformers.generate — R4).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from studio.stages import _common  # noqa: F401  (sys.path side effect)
from studio.stages._common import REPO
from studio.stages.agentic import FROZEN, run_episode


def llm_policy(gen):
    """Wrap an OpenAI-compatible generator into the episode-policy callable."""
    def policy(obs: str, trajectory: list[dict]) -> str:
        msgs = [{"role": "system",
                 "content": "You are an agent in a file workspace. Reply with ONE JSON "
                            "action only: {\"action\":\"list\"} or "
                            "{\"action\":\"read\",\"path\":\"...\"} or "
                            "{\"action\":\"answer\",\"text\":\"...\"}."}]
        for t in trajectory:
            msgs.append({"role": "user", "content": t["observation"]})
            msgs.append({"role": "assistant", "content": t["reply"]})
        msgs.append({"role": "user", "content": obs})
        return gen.chat([msgs], max_tokens=200, temperature=0.7)[0]
    return policy


def to_sft_rows(episode: dict) -> dict:
    """One solved episode → one multi-turn SFT row (full trajectory as chat)."""
    messages = []
    for t in episode["trajectory"]:
        messages.append({"role": "user", "content": t["observation"]})
        messages.append({"role": "assistant", "content": t["reply"]})
    return {"messages": messages, "task_id": episode["task_id"],
            "steps": episode["steps"]}


def collect(rows, policy, out_path: Path, episodes_per_task: int = 1) -> dict:
    kept, total = [], 0
    for task in rows:
        for _ in range(episodes_per_task):
            with tempfile.TemporaryDirectory(prefix="agentic_ep_") as td:
                ep = run_episode(task, policy, Path(td))
            total += 1
            if ep["solved"]:
                kept.append(to_sft_rows(ep))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    return {"episodes": total, "solved": len(kept),
            "solve_rate": round(len(kept) / total, 4) if total else 0.0,
            "out": str(out_path)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "agentic.yaml"))
    ap.add_argument("--policy", required=True, help="server:<model>@<base_url>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes-per-task", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = _common.load_config(args.config)
    _common.assert_frozen(cfg, FROZEN, "agentic")

    from gym import tasks as gym_tasks
    from gym.runner import generator_for
    rows = []
    for spec in cfg["data"]["tasks"]:
        rows += gym_tasks.load(spec["set"], split=spec.get("split", "dev"),
                               n=int(spec.get("n", 0)))
    stats = collect(rows, llm_policy(generator_for(args.policy)), Path(args.out),
                    args.episodes_per_task)
    print(f"COLLECT episodes={stats['episodes']} solved={stats['solved']} "
          f"rate={stats['solve_rate']} -> {stats['out']}")


if __name__ == "__main__":
    main()
