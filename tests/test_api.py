import pytest
from fastapi.testclient import TestClient

from nekaise_loop.api import create_app


def test_api_dashboard_and_control_contract(setup_loop, monkeypatch):
    settings, service, campaign, engine = setup_loop
    app = create_app(settings)
    monkeypatch.setattr(app.state.service, "ensure_worker", lambda: None)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/styles.css").status_code == 200
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/campaigns").json()[0]["id"] == campaign["id"]
        assert client.post(f"/api/campaigns/{campaign['id']}/actions",json={"action":"start"}).status_code == 202
        assert client.post(f"/api/campaigns/{campaign['id']}/actions",json={"action":"start"}).status_code == 409
        assert client.get("/api/campaigns/missing").status_code == 404
        assert client.get("/api/events?limit=9999").status_code == 422
        assert client.post("/api/campaigns",json={"name":"Bad", "config":{"rounds":0}}).status_code == 422


def test_cross_origin_and_dns_rebinding_blocked(setup_loop):
    settings, _, campaign, _ = setup_loop
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health",headers={"Host":"attacker.example"}).status_code == 403
        assert client.post(f"/api/campaigns/{campaign['id']}/actions",json={"action":"start"},headers={"Origin":"https://attacker.example"}).status_code == 403


def test_snapshot_contains_live_records_without_secret_settings(setup_loop):
    settings, _, campaign, engine = setup_loop
    engine.run(campaign["id"])
    with TestClient(create_app(settings)) as client:
        result = client.get(f"/api/campaigns/{campaign['id']}").json()
        assert result["round"]["lessons"][0]["student"]
        assert result["round"]["lessons"][0]["teacher"]
        assert result["round"]["evaluations"][0]["reference"]
        assert len(result["round"]["metrics"]) == 3
        assert "API_KEY" not in str(result)
        assert client.get(f"/api/campaigns/{campaign['id']}?round_id=missing").status_code == 404


def test_private_proxy_host_keeps_same_origin_controls(setup_loop, monkeypatch):
    settings, _, campaign, _ = setup_loop
    host = "studio.example.ts.net"
    app = create_app(settings, allowed_hosts=(host,))
    monkeypatch.setattr(app.state.service, "ensure_worker", lambda: None)
    with TestClient(app, base_url=f"https://{host}") as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").status_code == 200
        path = f"/api/campaigns/{campaign['id']}/actions"
        assert client.post(path, json={"action": "start"}, headers={"Origin": "https://unrelated.example.ts.net"}).status_code == 403
        assert client.get("/api/health", headers={"Host": "unrelated.example.ts.net"}).status_code == 403
        assert client.post(path, json={"action": "start"}, headers={"Origin": f"https://{host}"}).status_code == 202
