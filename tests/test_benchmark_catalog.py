import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from nekaise_loop.api import create_app
from nekaise_loop.benchmark_catalog import GPQA_DATASET, read_gpqa


def fixture():
    stamp = datetime.now(timezone.utc).isoformat()
    row = {"schema_version": 1, "benchmark": "gpqa-diamond", "run_id": "20260925T123456123456Z-abcdef01",
           "status": "complete", "started_at": stamp, "updated_at": stamp, "model_id": "a" * 64,
           "model_label": "TEST FIXTURE", "round_id": None, "round_number": None, "campaign_id": None,
           "protocol": "gpqa-diamond-gen-1", "protocol_id": "b" * 64, "dataset_sha256": GPQA_DATASET,
           "max_new_tokens": 1024, "max_input_tokens": 4096, "n": 198, "completed": 198,
           "correct": 0, "score": 0., "ci95": [0., .03], "invalid": 198, "budget_exhausted": 0,
           "seal_sha256": "c" * 64, "question": "PRIVATE", "response": "PRIVATE", "gold": "PRIVATE"}
    return {"schema_version": 1, "benchmark": "gpqa-diamond", "runs": [row], "questions": ["PRIVATE"]}


def write(root, data):
    (root / "gpqa-diamond.json").write_text(json.dumps(data))


def test_whitelist_and_missing_vs_real_zero(tmp_path):
    assert read_gpqa(tmp_path)["status"] == "empty"
    write(tmp_path, fixture()); value = read_gpqa(tmp_path)
    assert value["status"] == "ok" and value["runs"][0]["score"] == 0
    assert "PRIVATE" not in json.dumps(value)


@pytest.mark.parametrize("key,value", [("n", 2), ("completed", 197), ("score", .5), ("correct", True),
    ("protocol", "unknown"), ("dataset_sha256", "d" * 64), ("ci95", [-.1, .5]),
    ("status", "running"), ("seal_sha256", None), ("model_id", None)])
def test_inconsistent_full_score_rejected(tmp_path, key, value):
    data = fixture(); data["runs"][0][key] = value; write(tmp_path, data)
    assert read_gpqa(tmp_path)["status"] == "unavailable"


def test_progress_never_has_a_score_and_stale_is_explicit(tmp_path):
    data = fixture(); row = data["runs"][0]
    row.update(status="running", completed=3, started_at="2020-01-01T00:00:00Z", updated_at="2020-01-01T00:00:00Z")
    for key in ("correct", "score", "ci95", "invalid", "budget_exhausted", "seal_sha256"):
        row[key] = None
    write(tmp_path, data)
    value = read_gpqa(tmp_path)
    assert value["status"] == "ok" and value["stale"] and value["runs"][0]["score"] is None


def test_reader_bounds_and_symlink_escape(tmp_path):
    root = tmp_path / "projection"; root.mkdir()
    write(tmp_path, fixture()); (root / "gpqa-diamond.json").symlink_to(tmp_path / "gpqa-diamond.json")
    assert read_gpqa(root)["status"] == "unavailable"
    (root / "gpqa-diamond.json").unlink(); (root / "gpqa-diamond.json").write_bytes(b" " * (256 * 1024 + 1))
    assert read_gpqa(root)["status"] == "unavailable"


def test_endpoint_does_not_mutate_training_or_enter_snapshots(setup_loop, tmp_path, monkeypatch):
    settings, service, campaign, engine = setup_loop
    write(tmp_path, fixture()); monkeypatch.setenv("NEKAISE_BENCH_GPQA_PROJECTION_DIR", str(tmp_path))
    monkeypatch.setattr("nekaise_loop.service.Service.ensure_worker", lambda *_: None)
    tables = ("events", "records", "metrics", "actions", "recoveries")
    before = {t: service.store.query(f"SELECT * FROM {t}") for t in tables}
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/benchmarks/gpqa-diamond")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert response.json()["runs"][0]["correct"] == 0
        assert "PRIVATE" not in response.text
        assert "gpqa" not in json.dumps(client.get(f"/api/campaigns/{campaign['id']}").json()).lower()
    assert before == {t: service.store.query(f"SELECT * FROM {t}") for t in tables}


def test_reader_only_imported_by_dedicated_api():
    source = Path(__file__).parents[1] / "src/nekaise_loop"
    references = [p.name for p in source.rglob("*.py") if p.name != "benchmark_catalog.py" and "benchmark_catalog import" in p.read_text()]
    assert references == ["api.py"]
