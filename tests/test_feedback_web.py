from types import SimpleNamespace

import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from intentdesk.app_v3 import create_app
from intentdesk.feedback_web import create_lab_app


HEADERS={"X-IntentDesk-Client":"local-web"}


class Original:
    def __init__(self,root):
        self.metadata={"run_id":"fixture","gate":{"passed":False}}
        self.models=SimpleNamespace(classes=["report_lost_card"])


def test_lab_readiness_is_explicit_and_assets_are_local(tmp_path):
    with TestClient(create_app(tmp_path,Original),base_url="http://127.0.0.1") as client:
        status=client.get("/api/ml-lab/status").json()
        assert status["ready"] is False and status["automatic_actions"] is False
        assert client.get("/ml-lab").status_code==200
        assert client.get("/assets/feedback_lab.js").status_code==200
        assert client.post("/api/ml-lab/analyze",json={"text":"fiction"}).status_code==403
        assert client.post("/api/ml-lab/analyze",json={"text":" "},headers=HEADERS).status_code==422
        response=client.post("/api/ml-lab/analyze",json={"text":"fiction"},headers=HEADERS)
        assert response.status_code==503
        assert client.get("/api/status").json()["counts"]=={"pending":0,"reviewed":0}


def test_standalone_lab_needs_no_original_models_or_pilot_records(tmp_path):
    with TestClient(create_lab_app(tmp_path),base_url="http://127.0.0.1") as client:
        assert client.get("/").status_code==200
        assert client.get("/api/ml-lab/status").json()["ready"] is False
        assert client.get("/ml-lab",headers={"Host":"other.example"}).status_code==400
        assert client.post("/api/ml-lab/analyze",json={"text":"fiction"},headers={**HEADERS,"Origin":"https://other.example"}).status_code==403
        assert not (tmp_path/"local/pilot.sqlite3").exists()
