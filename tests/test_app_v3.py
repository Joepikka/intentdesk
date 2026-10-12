from types import SimpleNamespace

import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from intentdesk.app_v3 import create_app


HEADERS = {"X-IntentDesk-Client": "local-web"}


class Original:
    def __init__(self, root):
        self.metadata = {"run_id": "fixture", "gate": {"passed": False}}
        self.models = SimpleNamespace(classes=["report_lost_card"])
    def route(self, text):
        raise AssertionError("V3 requests must not enter the original suggestion pipeline")


class Hybrid:
    def __init__(self, root):
        pass
    def route(self, text):
        return {"label": "uncertain", "name": "Needs clarification", "reason": "Choose one issue.",
                "candidates": [], "model_version": "hybrid_fixture", "action_executed": False,
                "route_recommendation": "review_required"}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, Original, card_factory=Hybrid), base_url="http://127.0.0.1") as client:
        yield client


def test_v3_review_is_separate_and_stale_saves_are_rejected(client):
    result = client.post("/api/card-v3/requests", json={"text": "fixture multiple issues"}, headers=HEADERS)
    assert result.status_code == 201
    item = result.json()
    assert client.get("/api/status").json()["counts"] == {"pending": 0, "reviewed": 0}
    assert client.get("/api/card-v3/requests").json()["total"] == 1
    body = {"label": "uncertain", "expected_version": 0, "note": "fixture automated verification"}
    path = f"/api/card-v3/requests/{item['id']}/review"
    saved = client.post(path, json=body, headers=HEADERS)
    assert saved.status_code == 200 and saved.json()["handoff"]["action_executed"] is False
    assert client.post(path, json=body, headers=HEADERS).status_code == 409
    current = client.get(f"/api/card-v3/requests/{item['id']}").json()
    assert len(current["decisions"]) == 1 and current["decisions"][0]["label"] == "uncertain"
    assert client.get("/api/pilot/report").json()["completed_human_sessions"] == 0


def test_v3_inherits_local_boundaries_and_validates_fields(client):
    path = "/api/card-v3/requests"
    assert client.post(path, json={"text": "fiction"}).status_code == 403
    assert client.post(path, json={"text": "fiction"}, headers={**HEADERS, "Origin": "https://other.example"}).status_code == 403
    assert client.post(path, json={"text": " "}, headers=HEADERS).status_code == 422
    assert client.get("/card-v3").status_code == 200
    assert client.get("/assets/card_v3.js").status_code == 200
    assert client.get("/assets/card_v3.css").status_code == 200


def test_busy_and_unavailable_models_do_not_create_requests(client):
    lock = client.app.state.inference_lock; lock.acquire()
    try:
        assert client.post("/api/card-v3/requests", json={"text": "fiction"}, headers=HEADERS).status_code == 503
    finally:
        lock.release()
    assert client.get("/api/card-v3/requests").json()["total"] == 0


def test_unknown_label_and_missing_record(client):
    response = client.post("/api/card-v3/requests/missing/review", json={"label": "weather", "expected_version": 0}, headers=HEADERS)
    assert response.status_code == 422
    assert client.get("/api/card-v3/requests/missing").status_code == 404
