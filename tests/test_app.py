from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from intentdesk.app import create_app
from intentdesk.reviews import ReviewConflict, ReviewStore

HEADERS = {"X-IntentDesk-Client": "local-web"}


class FakeRouter:
    def __init__(self, root):
        self.metadata = {"run_id": "fixture", "gate": {"passed": False}}
        self.models = SimpleNamespace(classes=["weather", "report_lost_card"])

    def route(self, text):
        return {"policy_version": "fixture", "route_recommendation": "route_to_intent",
                "action_executed": True, "candidates": [{"intent": "weather", "score": .99}],
                "reason": "Fixture only."}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, FakeRouter), base_url="http://127.0.0.1") as client:
        yield client


def submit(client, text="What is the weather?"):
    return client.post("/api/requests", json={"text": text}, headers=HEADERS)


def test_request_review_history_and_persistence(client, tmp_path):
    item = submit(client).json()
    assert item["status"] == "pending" and item["version"] == 0
    assert item["analysis"]["route_recommendation"] == "review_required"
    assert item["analysis"]["action_executed"] is False
    result = client.post(f"/api/requests/{item['id']}/review", headers=HEADERS,
                         json={"label": "weather", "note": "Reviewed fixture", "expected_version": 0})
    assert result.status_code == 200
    assert result.json()["decisions"][0]["label"] == "weather"
    store = ReviewStore(tmp_path/"local/reviews.sqlite3")
    assert store.get(item["id"])["version"] == 1
    assert store.counts() == {"pending": 0, "reviewed": 1}


def test_stale_decision_cannot_overwrite_or_append_history(client):
    item = submit(client).json()
    path = f"/api/requests/{item['id']}/review"
    body = {"label": "oos", "expected_version": 0}
    assert client.post(path, json=body, headers=HEADERS).status_code == 200
    assert client.post(path, json=body, headers=HEADERS).status_code == 409
    current = client.get(f"/api/requests/{item['id']}").json()
    assert len(current["decisions"]) == 1 and current["version"] == 1


@pytest.mark.parametrize("text", ["", "  ", "x"*4001, 17, None])
def test_invalid_input_never_enters_queue(client, text):
    assert submit(client, text).status_code == 422
    assert client.get("/api/status").json()["counts"]["pending"] == 0


def test_unknown_label_rejected_without_changing_request(client):
    item = submit(client).json()
    result = client.post(f"/api/requests/{item['id']}/review", headers=HEADERS,
                         json={"label": "invented", "expected_version": 0})
    assert result.status_code == 422
    assert client.get(f"/api/requests/{item['id']}").json()["version"] == 0


def test_origin_host_and_custom_header_boundaries(client):
    assert client.post("/api/requests", json={"text": "weather"}).status_code == 403
    assert client.post("/api/requests", json={"text": "weather"}, headers={**HEADERS,"Origin":"https://other.example"}).status_code == 403
    assert client.get("/api/status", headers={"Host": "other.example"}).status_code == 400
    assert client.post("/api/requests", json={"text": "weather"}, headers={**HEADERS,"Origin":"http://127.0.0.1"}).status_code == 201


def test_large_body_and_missing_record_are_handled(client):
    assert client.post("/api/requests", content=b"x"*32769, headers=HEADERS).status_code == 413
    assert client.get("/api/requests/not-a-record").status_code == 404
    assert client.post("/api/requests/not-a-record/review", headers=HEADERS,
                       json={"label":"oos","expected_version":0}).status_code == 404


def test_pagination_and_review_only_metadata(client):
    for i in range(3):
        assert submit(client, f"weather {i}").status_code == 201
    first = client.get("/api/requests?limit=2").json()
    second = client.get("/api/requests?limit=2&offset=2").json()
    assert first["total"] == 3 and len(second["items"]) == 1
    assert {x["id"] for x in first["items"]}.isdisjoint(x["id"] for x in second["items"])
    assert client.get("/api/requests?limit=1000").status_code == 422
    status = client.get("/api/status").json()
    assert not status["automatic_actions"] and not status["feedback_used_for_training"]


def test_busy_model_returns_retry_without_creating_a_record(client):
    lock = client.app.state.inference_lock
    lock.acquire()
    try:
        assert submit(client).status_code == 503
    finally:
        lock.release()
    assert client.get("/api/status").json()["counts"]["pending"] == 0


def test_local_static_assets_are_served_with_security_headers(client):
    result = client.get("/")
    assert result.status_code == 200 and "Request review" in result.text
    assert "frame-ancestors 'none'" in result.headers["content-security-policy"]
    assert result.headers["cache-control"] == "no-store"
    assert client.get("/assets/app.js").status_code == 200


def test_missing_search_index_is_explicit_and_query_is_not_saved(client):
    assert not client.get("/api/search/status").json()["ready"]
    assert client.post("/api/search", json={"query":"python"}, headers=HEADERS).status_code == 503
    assert client.get("/api/status").json()["counts"] == {"pending":0,"reviewed":0}


def test_sql_text_is_data_and_feedback_is_append_only(tmp_path):
    store = ReviewStore(tmp_path/"reviews.sqlite3")
    item = store.create("'); DROP TABLE requests; --", {"candidates": []})
    store.review(item["id"], "oos", "first", 0)
    store.review(item["id"], "uncertain", "revised", 1)
    assert [d["label"] for d in store.get(item["id"])["decisions"]] == ["oos", "uncertain"]
    with pytest.raises(ReviewConflict):
        store.review(item["id"], "oos", "stale", 0)
