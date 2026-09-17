"""Dashboard API behind loopback or an explicitly allowed private proxy hostname."""
from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import CampaignConfig, Settings
from .service import Service, Conflict


class CreateCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    config: CampaignConfig


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["start", "pause", "resume", "stop", "review"]
    reason: str | None = Field(default=None, max_length=12000)


def create_app(settings: Settings | None = None, *, allowed_hosts: tuple[str, ...] = ()):
    service = Service(settings or Settings())
    hosts = {"localhost", "127.0.0.1", "::1", "testserver", *(h.lower() for h in allowed_hosts)}
    @asynccontextmanager
    async def lifespan(app):
        if service.store.one("SELECT id FROM actions WHERE handled_at IS NULL LIMIT 1") or service.store.one("SELECT id FROM recoveries WHERE status IN ('pending','waiting','running','decided') LIMIT 1") or service.store.one("SELECT id FROM campaigns WHERE status IN ('running','queued','pausing','stopping') OR (operator_hold IS NULL AND status NOT IN ('ready','complete') AND json_extract(config,'$.auto_recover')=1) LIMIT 1"):
            service.ensure_worker()
        yield
    app = FastAPI(title="Nekaise Studio", version="0.1.0", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        hostname = request.url.hostname
        if hostname not in hosts:
            return JSONResponse({"detail": "This dashboard hostname is not allowed"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.headers.get('host')}"
            if origin and origin != expected:
                return JSONResponse({"detail": "Cross-origin changes are not allowed"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-site changes are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        return response

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse({"detail": "Record not found"}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/system")
    def system():
        return service.readiness()

    @app.get("/api/campaigns")
    def campaigns():
        return service.list_campaigns()

    @app.post("/api/campaigns", status_code=201)
    def create(body: CreateCampaign):
        if not body.name.strip():
            raise HTTPException(status_code=422, detail="Campaign name cannot be blank")
        return service.create(body.name.strip(), body.config)

    @app.post("/api/campaigns/{campaign_id}/actions", status_code=202)
    def action(campaign_id: str, body: Action):
        return service.action(campaign_id, body.action, reason=body.reason)

    @app.get("/api/campaigns/{campaign_id}")
    def snapshot(campaign_id: str, round_id: str | None = None):
        return service.snapshot(campaign_id, round_id)

    @app.get("/api/campaigns/{campaign_id}/telemetry")
    def telemetry(campaign_id: str):
        from .telemetry import training_telemetry
        return training_telemetry(service, campaign_id)

    @app.get("/api/campaigns/{campaign_id}/benchmark")
    def benchmark(campaign_id: str):
        from .benchmark import read_benchmark
        service.store.campaign(campaign_id)
        return JSONResponse(read_benchmark(campaign_id), headers={"Cache-Control": "no-store"})

    @app.get("/api/campaigns/{campaign_id}/benchmark/history")
    def benchmark_history(campaign_id: str, history: str = Query(pattern=r"^[a-f0-9]{64}$"),
                          page: int = Query(ge=0)):
        from .benchmark import read_benchmark_history
        from fastapi.responses import JSONResponse
        return JSONResponse(read_benchmark_history(campaign_id, history, page),
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/rounds/{round_id}")
    def round_detail(round_id: str):
        return service.round_detail(round_id)

    @app.get("/api/rounds/{round_id}/materials")
    def materials(round_id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        return service.materials(round_id, offset, limit)

    @app.get("/api/events")
    def events(campaign_id: str | None = None, after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        return service.store.events(campaign_id, after, limit)

    @app.get("/api/history-reviews")
    def history_reviews():
        import json
        rows = service.store.query("SELECT h.*,r.decision FROM history_reviews h JOIN recoveries r ON r.id=h.recovery_id ORDER BY h.applied_at DESC LIMIT 100")
        for row in rows:
            row["result"] = json.loads(row["result"])
            row["decision"] = json.loads(row["decision"]) if row["decision"] else None
        return rows

    @app.get("/api/reports")
    def reports(before: int | None = Query(None, ge=1), limit: int = Query(30, ge=1, le=100)):
        from .reports import catalog, current_status
        return {"current": current_status(service), **catalog(service, before=before, limit=limit)}

    @app.get("/api/reports/{recovery_id}")
    def report(recovery_id: int):
        from .reports import detail
        return detail(service, recovery_id)

    app.mount("/", StaticFiles(directory=service.settings.dashboard, html=True), name="dashboard")
    return app
