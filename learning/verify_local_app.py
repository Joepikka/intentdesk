"""Smoke-test the running local server with labeled fictional assistant-authored examples."""

from pathlib import Path
import time

import httpx

from intentdesk.common import write_json

root = Path(__file__).resolve().parents[1]
headers = {"X-IntentDesk-Client":"local-web"}
with httpx.Client(base_url="http://127.0.0.1:8765", timeout=15) as client:
    status = client.get("/api/status")
    status.raise_for_status()
    start = time.perf_counter()
    response = client.post("/api/requests", json={"text":"Write a sonnet about the moon."}, headers=headers)
    response.raise_for_status()
    elapsed = (time.perf_counter()-start)*1000
    record = response.json()
    assert record["analysis"]["route_recommendation"] == "review_required"
    assert not record["analysis"]["action_executed"]
    decision = {"label":"oos", "note":"Assistant-authored smoke example; not independently adjudicated training/validation data.","expected_version":0}
    review = client.post(f"/api/requests/{record['id']}/review",json=decision,headers=headers)
    review.raise_for_status()
    assert len(review.json()["decisions"]) == 1
    stale = client.post(f"/api/requests/{record['id']}/review",json=decision,headers=headers)
    assert stale.status_code == 409
    invalid = client.post("/api/requests",json={"text":"  "},headers=headers)
    assert invalid.status_code == 422
    search = client.post("/api/search",json={"query":"photosynthesis chlorophyll"},headers=headers)
    search.raise_for_status()
    assert search.json()["results"] and not search.json()["answer_generated"]
    result = {"real_model_api": "passed", "review_persistence": "passed", "stale_review_409":"passed",
              "invalid_input_422":"passed", "full_corpus_search_api":"passed", "automatic_actions": False,
              "request_api_single_smoke_ms": elapsed, "search_api_single_smoke_ms": search.json()["elapsed_ms"],
              "model_policy":status.json()["policy_version"], "test_evaluated":status.json()["test_evaluated"],
              "scope":"Fictional assistant-authored local smoke examples; not load testing or human validation"}
    write_json(root/"reports/app_smoke.json", result)
    print(result)
