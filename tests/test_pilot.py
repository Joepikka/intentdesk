import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from intentdesk.app import create_app
from intentdesk.pilot import CASES, CATEGORIES, PROTOCOL, PilotStore, assignment, suggest
from intentdesk.reviews import ReviewConflict

HEADERS = {"X-IntentDesk-Client": "local-web"}


class PilotRouter:
    def __init__(self, root=None):
        self.calls = []
        self.metadata = {"run_id": "fixture", "gate": {"passed": False}}
        self.models = SimpleNamespace(classes=list(CATEGORIES))

    def route(self, text):
        self.calls.append(text)
        return {"policy_version": "fixture", "scope_assessment": {"status": "supported_candidate"},
                "candidates": [{"intent": "report_lost_card", "score": .99}]}


def store(tmp_path):
    return PilotStore(tmp_path/"pilot.sqlite3", tmp_path, PilotRouter())


def finish(pilot, sid, *, interrupted=False, wrong=False):
    # Test fixtures consult the synthetic oracle; runtime clients never receive it.
    expected = {c["text"]: c["label"] for c in CASES}
    while not pilot.view(sid)["finished"]:
        trial = pilot.next(sid)["trial"]
        pilot.decide(sid, trial["id"], "uncertain" if wrong else expected[trial["text"]], interrupted)


def test_balanced_unique_cases_and_adjacent_reviewer_counterbalance():
    first, second = assignment(0), assignment(1)
    assert len({x["id"] for x in first}) == 14
    assert {x["id"] for x in first} == {x["id"] for x in second}
    assert sum(x["condition"] == "manual" for x in first) == 7
    assert {(x["label"], x["condition"]) for x in first} == {
        (c["label"], condition) for c in CASES for condition in ("manual", "assisted")}
    paired = {x["id"]: x["condition"] for x in second}
    assert all(paired[x["id"]] != x["condition"] for x in first)


def test_projection_never_cherry_picks_a_lower_supported_category():
    analysis = {"scope_assessment": {"status": "supported_candidate"},
                "candidates": [{"intent": "weather"}, {"intent": "report_lost_card"}]}
    assert suggest(analysis)["label"] == "oos"
    analysis["scope_assessment"]["status"] = "unavailable"
    assert suggest(analysis)["label"] == "uncertain"
    analysis["scope_assessment"]["status"] = "likely_out_of_scope"
    assert suggest(analysis)["label"] == "oos"
    assert suggest({})["action_executed"] is False


def test_manual_blinding_assisted_inference_once_and_timer_persists(tmp_path):
    pilot = store(tmp_path); session = pilot.start("reviewer01")
    assert session["trial"] is None and "summary" not in session
    sid = session["id"]
    for _ in range(14):
        before = len(pilot.router.calls)
        value = pilot.next(sid); trial = value["trial"]
        assert "expected_label" not in json.dumps(value)
        assert len(pilot.router.calls) == before + (trial["condition"] == "assisted")
        assert (trial["suggestion"] is None) == (trial["condition"] == "manual")
        with pilot.connect() as db:
            shown = db.execute("SELECT shown_at FROM pilot_trials WHERE id=?", (trial["id"],)).fetchone()[0]
        assert pilot.next(sid)["trial"] == trial
        with pilot.connect() as db:
            assert db.execute("SELECT shown_at FROM pilot_trials WHERE id=?", (trial["id"],)).fetchone()[0] == shown
        pilot.decide(sid, trial["id"], "uncertain")
    assert pilot.view(sid)["finished"] and len(pilot.router.calls) == 7


def test_unopened_stale_and_invalid_decisions_cannot_change_history(tmp_path):
    pilot = store(tmp_path); sid = pilot.start("reviewer01")["id"]
    with pytest.raises(ReviewConflict): pilot.decide(sid, "not-opened", "oos")
    trial = pilot.next(sid)["trial"]
    with pytest.raises(ValueError): pilot.decide(sid, trial["id"], "invented")
    saved = pilot.decide(sid, trial["id"], "oos")
    assert saved["completed"] == 1 and saved["last_handoff"]["action_executed"] is False
    with pytest.raises(ReviewConflict): pilot.decide(sid, trial["id"], "uncertain")
    assert pilot.view(sid)["completed"] == 1


def test_resume_and_smoke_do_not_change_human_slots(tmp_path):
    pilot = store(tmp_path); first = pilot.start("reviewer01")
    assert pilot.start("reviewer01")["id"] == first["id"]
    pilot.start("smoke01", "smoke")
    second = pilot.start("reviewer02")
    with pilot.connect() as db:
        assert db.execute("SELECT slot FROM pilot_sessions WHERE id=?", (second["id"],)).fetchone()[0] == 1
    with pytest.raises(ReviewConflict): pilot.start("reviewer01", "smoke")
    assert pilot.report()["incomplete_human_sessions"] == 2


def test_interrupted_trials_remain_in_accuracy_but_not_timing(tmp_path):
    pilot = store(tmp_path); sid = pilot.start("reviewer01")["id"]
    finish(pilot, sid, interrupted=True)
    for condition in pilot.view(sid)["summary"].values():
        assert condition["accuracy"] == 1 and condition["count"] == 7
        assert condition["timed_count"] == 0 and condition["timing_exclusions"] == 7
        assert condition["mean_decision_seconds"] is None
    assert pilot.report()["paired_reviewer_seconds_difference"]["reviewers"] == 0


def test_long_pause_and_negative_clock_are_excluded(tmp_path, monkeypatch):
    pilot = store(tmp_path); sid = pilot.start("reviewer01")["id"]
    for clock, expected in ((1400., "over_300_seconds"), (900., "clock_reversal")):
        monkeypatch.setattr("intentdesk.pilot.time.time", lambda: 1000.)
        trial = pilot.next(sid)["trial"]
        monkeypatch.setattr("intentdesk.pilot.time.time", lambda: clock)
        pilot.decide(sid, trial["id"], "oos")
        with pilot.connect() as db:
            row = db.execute("SELECT * FROM pilot_trials WHERE id=?", (trial["id"],)).fetchone()
        assert row["timing_exclusion"] == expected and row["elapsed_seconds"] >= 0


def test_reports_exclude_smoke_and_incomplete_sessions_never_claim_savings(tmp_path):
    pilot = store(tmp_path)
    finish(pilot, pilot.start("smoke01", "smoke")["id"])
    pilot.start("unfinished")
    assert pilot.report()["conditions"]["manual"]["count"] == 0
    finish(pilot, pilot.start("reviewer01")["id"], wrong=True)
    report = pilot.report()
    assert report["smoke_sessions_excluded"] == 1 and report["completed_human_sessions"] == 1
    assert report["conditions"]["manual"]["accuracy"] == 0
    assert report["paired_reviewer_accuracy_difference"]["interval_95"] is None
    assert report["business_value_established"] is False and report["binding"]["protocol"]["data_is_real_customer_traffic"] is False


def test_protocol_change_preserves_previous_records_under_a_separate_binding(tmp_path, monkeypatch):
    original = store(tmp_path); sid = original.start("reviewer01")["id"]
    monkeypatch.setitem(PROTOCOL, "version", 2)
    changed = store(tmp_path)
    assert changed.run_id != original.run_id
    with pytest.raises(ReviewConflict): changed.view(sid)
    with changed.connect() as db:
        assert db.execute("SELECT count(*) FROM pilot_sessions").fetchone()[0] == 1


def test_pilot_api_limits_blinding_history_and_original_queue(tmp_path):
    with TestClient(create_app(tmp_path, PilotRouter), base_url="http://127.0.0.1") as client:
        assert client.get("/pilot").status_code == 200
        assert client.get("/assets/pilot.js").status_code == 200
        assert client.post("/api/pilot/sessions", json={"reviewer":"reviewer01"}).status_code == 403
        assert client.post("/api/pilot/sessions", json={"reviewer":"<script>"}, headers=HEADERS).status_code == 422
        sid = client.post("/api/pilot/sessions", json={"reviewer":"reviewer01"}, headers=HEADERS).json()["id"]
        opened = client.post(f"/api/pilot/sessions/{sid}/next", json={}, headers=HEADERS)
        assert opened.status_code == 200 and "expected_label" not in opened.text
        trial = opened.json()["trial"]
        payload = {"trial_id":trial["id"],"label":"oos","interrupted":False}
        assert client.post(f"/api/pilot/sessions/{sid}/decision", json=payload, headers=HEADERS).status_code == 200
        assert client.post(f"/api/pilot/sessions/{sid}/decision", json=payload, headers=HEADERS).status_code == 409
        assert client.get("/api/status").json()["counts"] == {"pending":0,"reviewed":0}
        assert client.get("/api/pilot/sessions/missing").status_code == 404
