#!/usr/bin/env python3
"""Nekaise Studio dashboard — a zero-dependency, fully-local training viewer.

    python dashboard/server.py        # then open http://localhost:8765

Serves a single vanilla-JS page that polls the run files written by
dashboard/runlog.py (experiments/*/runs/*/{meta.json,events.jsonl}). No external
services, no CDN. The agent starts this when it kicks off an experiment.
"""
from __future__ import annotations

import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
HTML = Path(__file__).parent / "index.html"
PORT = int(os.environ.get("NEKAISE_DASH_PORT", "8765"))
HOST = os.environ.get("NEKAISE_DASH_HOST", "0.0.0.0")  # 0.0.0.0 = reachable over LAN


def list_runs() -> list[dict]:
    runs = []
    for meta in EXP.glob("*/runs/*/meta.json"):
        try:
            runs.append(json.loads(meta.read_text()))
        except Exception:
            pass
    runs.sort(key=lambda r: r.get("started", 0), reverse=True)
    return runs


def read_events(exp: str, run_id: str) -> list[dict]:
    f = EXP / exp / "runs" / run_id / "events.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str) -> None:
        body = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/runs":
            self._send(200, json.dumps(list_runs()), "application/json")
        elif u.path == "/api/events":
            q = parse_qs(u.query)
            data = read_events(q.get("exp", [""])[0], q.get("run", [""])[0])
            self._send(200, json.dumps(data), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, *a) -> None:  # quiet
        pass


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return HOST
    finally:
        s.close()


def main() -> None:
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"nekaise-studio dashboard: http://localhost:{PORT}  (LAN: http://{_lan_ip()}:{PORT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
