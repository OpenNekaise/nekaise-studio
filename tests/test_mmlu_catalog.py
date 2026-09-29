import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from nekaise_loop.api import create_app
from nekaise_loop.mmlu_catalog import MMLU_DATASET, CATEGORY_COUNTS, read_mmlu_pro


def fixture():
    stamp = datetime.now(timezone.utc).isoformat()
    row = {"schema_version": 1, "benchmark": "mmlu-pro", "run_id": "20260925T123456123456Z-abcdef01",
           "status": "complete", "started_at": stamp, "updated_at": stamp, "model_id": "a" * 64,
           "model_label": "TEST FIXTURE", "round_id": None, "round_number": None, "campaign_id": None,
           "protocol": "mmlu-pro-choice-1", "protocol_id": "b" * 64, "dataset_sha256": MMLU_DATASET,
           "max_input_tokens": 4096, "n": 12032, "completed": 12032,
           "correct": 0, "score": 0., "ci95": [0., .03], 'chance_score':.11, 'subjects':[{'category':k,'n':v,'correct':0,'score':0.,'question':'PRIVATE'} for k,v in CATEGORY_COUNTS.items()],
           "seal_sha256": "c" * 64, "question": "PRIVATE", "response": "PRIVATE", "gold": "PRIVATE"}
    return {"schema_version": 1, "benchmark": "mmlu-pro", "runs": [row], "questions": ["PRIVATE"]}


def write(root, data):
    (root / "mmlu-pro.json").write_text(json.dumps(data))


def test_whitelist_and_missing_vs_real_zero(tmp_path):
    assert read_mmlu_pro(tmp_path)["status"] == "empty"
    write(tmp_path, fixture()); value = read_mmlu_pro(tmp_path)
    assert value["status"] == "ok" and value["runs"][0]["score"] == 0
    assert "PRIVATE" not in json.dumps(value)


@pytest.mark.parametrize("key,value", [("n", 2), ("completed", 12031), ("score", .5), ("correct", True),
    ("protocol", "unknown"), ("dataset_sha256", "d" * 64), ("ci95", [-.1, .5]),
    ("status", "running"), ("seal_sha256", None), ("model_id", None)])
def test_inconsistent_full_score_rejected(tmp_path, key, value):
    data = fixture(); data["runs"][0][key] = value; write(tmp_path, data)
    assert read_mmlu_pro(tmp_path)["status"] == "unavailable"


def test_progress_never_has_a_score_and_stale_is_explicit(tmp_path):
    data = fixture(); row = data["runs"][0]
    row.update(status="running", completed=3, started_at="2020-01-01T00:00:00Z", updated_at="2020-01-01T00:00:00Z")
    for key in ("correct", "score", "ci95", "chance_score", "subjects", "seal_sha256"):
        row[key] = None
    write(tmp_path, data)
    value = read_mmlu_pro(tmp_path)
    assert value["status"] == "ok" and value["stale"] and value["runs"][0]["score"] is None


def test_reader_bounds_and_symlink_escape(tmp_path):
    root = tmp_path / "projection"; root.mkdir()
    write(tmp_path, fixture()); (root / "mmlu-pro.json").symlink_to(tmp_path / "mmlu-pro.json")
    assert read_mmlu_pro(root)["status"] == "unavailable"
    (root / "mmlu-pro.json").unlink(); (root / "mmlu-pro.json").write_bytes(b" " * (256 * 1024 + 1))
    assert read_mmlu_pro(root)["status"] == "unavailable"


def test_bad_or_stale_history_cannot_hide_a_valid_new_result(tmp_path):
    data = fixture()
    good = copy.deepcopy(data["runs"][0])
    bad = copy.deepcopy(good); bad["run_id"] = "20260925T123456123456Z-abcdef02"; bad["score"] = .5
    old = copy.deepcopy(good); old.update(run_id="20260925T123456123456Z-abcdef03", status="running", completed=7,
                                         started_at="2020-01-01T00:00:00Z", updated_at="2020-01-01T00:00:00Z")
    for key in ("correct", "score", "ci95", "chance_score", "subjects", "seal_sha256"):
        old[key] = None
    data["runs"] = [good, bad, old]
    write(tmp_path, data); value = read_mmlu_pro(tmp_path)
    assert value["status"] == "ok" and value["rejected_entries"] == 1 and len(value["runs"]) == 2
    assert not value["stale"] and value["runs"][1]["stale"]


def test_numerical_roundoff_at_zero_does_not_reject_real_zero(tmp_path):
    data = fixture(); data["runs"][0]["ci95"][0] = 1e-18
    write(tmp_path, data)
    assert read_mmlu_pro(tmp_path)["runs"][0]["score"] == 0


def test_endpoint_does_not_mutate_training_or_enter_snapshots(setup_loop, tmp_path, monkeypatch):
    settings, service, campaign, engine = setup_loop
    write(tmp_path, fixture()); monkeypatch.setenv("NEKAISE_BENCH_MMLU_PRO_PROJECTION_DIR", str(tmp_path))
    monkeypatch.setattr("nekaise_loop.service.Service.ensure_worker", lambda *_: None)
    tables = ("events", "records", "metrics", "actions", "recoveries")
    before = {t: service.store.query(f"SELECT * FROM {t}") for t in tables}
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/benchmarks/mmlu-pro")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert response.json()["runs"][0]["correct"] == 0
        assert "PRIVATE" not in response.text
        assert "mmlu-pro" not in json.dumps(client.get(f"/api/campaigns/{campaign['id']}").json()).lower()
    assert before == {t: service.store.query(f"SELECT * FROM {t}") for t in tables}


def test_reader_only_imported_by_dedicated_api():
    source = Path(__file__).parents[1] / "src/nekaise_loop"
    references = [p.name for p in source.rglob("*.py") if p.name != "mmlu_catalog.py" and "mmlu_catalog import" in p.read_text()]
    assert references == ["api.py"]


def test_subjects_must_match_total_and_exact_category_counts(tmp_path):
    for mutate in (lambda r:r['subjects'][0].update(n=1),
                   lambda r:r['subjects'][0].update(correct=1),
                   lambda r:r['subjects'][0].update(category='private subject'),
                   lambda r:r.update(subjects=r['subjects'][:-1])):
        data=fixture();mutate(data['runs'][0]);write(tmp_path,data)
        assert read_mmlu_pro(tmp_path)['status']=='unavailable'
