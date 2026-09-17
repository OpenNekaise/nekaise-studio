"""Small command interface for agents and humans; shares operations with the API."""
import argparse
import json

from .config import CampaignConfig, Settings
from .service import Service


def main():
    parser = argparse.ArgumentParser(prog="nekaise-loop")
    parser.add_argument("--workspace", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1", help="Bind address; use this server's Tailscale IP for private remote access")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--allow-host", action="append", default=[], help="Exact hostname allowed through a private reverse proxy; repeatable")
    sub.add_parser("worker")
    sub.add_parser("supervisor")
    for verb in ("recover", "apply-recovery"):
        sub.add_parser(verb).add_argument("recovery_id", type=int)
    sub.add_parser("shutdown-worker")
    sub.add_parser("doctor")
    sub.add_parser("list")
    sub.add_parser("history-inventory")
    sub.add_parser("history-reviews")
    reports = sub.add_parser("reports")
    reports.add_argument("--before", type=int)
    reports.add_argument("--limit", type=int, choices=range(1, 101), default=30)
    sub.add_parser("report").add_argument("recovery_id", type=int)
    sub.add_parser("restore-log").add_argument("cleanup_id", type=int)
    create = sub.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--config", required=True, help="JSON campaign recipe")
    status = sub.add_parser("status")
    status.add_argument("campaign_id")
    continuation = sub.add_parser("continue")
    continuation.add_argument("campaign_id")
    continuation.add_argument("--rounds", type=int, default=-1)
    for verb in ("start", "pause", "resume", "stop", "review"):
        p = sub.add_parser(verb)
        p.add_argument("campaign_id")
        p.add_argument("--reason", default=None)
    args = parser.parse_args()
    settings = Settings(args.workspace)
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(settings, allowed_hosts=(args.host, *args.allow_host)), host=args.host, port=args.port, access_log=False, proxy_headers=True, forwarded_allow_ips="127.0.0.1")
        return
    if args.command == "worker":
        from .worker import run_worker
        run_worker(settings)
        return
    if args.command == "supervisor":
        from .supervisor import run_supervisor
        run_supervisor(settings)
        return
    if args.command in {"recover", "apply-recovery"}:
        from .recovery import handle_recovery, apply_recovery
        (handle_recovery if args.command == "recover" else apply_recovery)(settings, args.recovery_id)
        return
    service = Service(settings)
    if args.command == "doctor":
        result = service.readiness()
    elif args.command == "shutdown-worker":
        result = service.shutdown_worker()
    elif args.command == "list":
        result = service.list_campaigns()
    elif args.command == "history-inventory":
        from .history import inventory
        result = inventory(service)
    elif args.command == "history-reviews":
        result = service.store.query("SELECT * FROM history_reviews ORDER BY applied_at DESC")
    elif args.command == "reports":
        from .reports import catalog, current_status
        result = {"current": current_status(service), **catalog(service, before=args.before, limit=args.limit)}
    elif args.command == "report":
        from .reports import detail
        result = detail(service, args.recovery_id)
    elif args.command == "restore-log":
        from .history import restore_log
        result = restore_log(service, args.cleanup_id)
    elif args.command == "create":
        from pathlib import Path
        result = service.create(args.name, CampaignConfig.model_validate_json(Path(args.config).read_text()))
    elif args.command == "status":
        result = service.snapshot(args.campaign_id)
    elif args.command == "continue":
        campaign = service.continue_campaign(args.campaign_id, {"rounds": args.rounds}, start=True)
        service.ensure_worker()
        result = {"campaign_id": campaign["id"], "status": campaign["status"]}
    else:
        result = service.action(args.campaign_id, args.command, reason=args.reason)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
