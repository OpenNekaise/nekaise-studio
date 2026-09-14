"""Run a bounded REAL two-round campaign through the production worker queue.

Uses cached MiniCPM5 1B weights, the existing corpus, and Claude Code. Creates real
training/checkpoint artifacts and incurs up to 14 teacher calls. This is a plumbing
check, not evidence of model quality. Results stay visible in the dashboard.
"""
import json
import time
import argparse

from nekaise_loop.config import CampaignConfig, Settings
from nekaise_loop.service import Service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=["claude", "codex"], default="codex")
    parser.add_argument("--teacher-model", default=None)
    args = parser.parse_args()
    service = Service(Settings())
    config = CampaignConfig(teacher_provider=args.teacher, teacher_model=args.teacher_model or ("claude-fable-5-1" if args.teacher=="claude" else "gpt-5.6-terra"), rounds=2, lessons_per_round=2, eval_questions=2, train_steps=0, max_seq_len=256, max_new_tokens=96, passage_chars=1500, max_stage_seconds=600, max_teacher_calls=14, auto_recover=False)
    campaign = service.create(f"Building energy · {args.teacher.title()} smoke", config)
    print(json.dumps({"campaign_id": campaign["id"], "config": config.model_dump()}), flush=True)
    service.action(campaign["id"], "start")
    last_event, deadline = 0, time.monotonic()+2400
    while time.monotonic() < deadline:
        for event in service.store.events(campaign["id"], after=last_event):
            print(f"{event['created_at']} {event['kind']}: {event['message']}", flush=True)
            last_event = event["id"]
        state = service.store.campaign(campaign["id"])
        if state["status"] in {"complete", "failed", "stopped", "interrupted", "waiting"}:
            snapshot = service.snapshot(campaign["id"])
            print(json.dumps({"status": state["status"], "error": state["error"], "rounds": len(snapshot["rounds"]), "teacher_usage": snapshot["teacher_usage"]}), flush=True)
            return 0 if state["status"] == "complete" else 1
        time.sleep(2)
    service.action(campaign["id"], "stop")
    raise TimeoutError("Smoke budget expired; stop was requested")


if __name__ == "__main__":
    raise SystemExit(main())
