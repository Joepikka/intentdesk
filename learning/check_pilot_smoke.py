"""Exercise a SMOKE session only. Never automate a human measurement session."""
from pathlib import Path
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from intentdesk.pilot import CASES

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8765"


def api(path, body=None):
    request = Request(BASE+path, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type":"application/json", "X-IntentDesk-Client":"local-web"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


before = api("/api/pilot/report")["completed_human_sessions"]
session = api("/api/pilot/sessions", {"reviewer":"automated-check-v1", "kind":"smoke"})
assert session["kind"] == "smoke", "Refusing to automate a human session."
expected = {case["text"]: case["label"] for case in CASES}
sid = session["id"]
seen = []
last = None
while not session["finished"]:
    session = api(f"/api/pilot/sessions/{sid}/next", {})
    trial = session["trial"]
    assert "expected_label" not in trial
    assert (trial["suggestion"] is None) == (trial["condition"] == "manual")
    last = {"trial_id":trial["id"], "label":expected[trial["text"]], "interrupted":True}
    seen.append(trial["condition"])
    session = api(f"/api/pilot/sessions/{sid}/decision", last)
    assert session["last_handoff"]["action_executed"] is False
if last:
    try:
        api(f"/api/pilot/sessions/{sid}/decision", last)
        raise AssertionError("Duplicate decision unexpectedly accepted.")
    except HTTPError as error:
        assert error.code == 409
report = api("/api/pilot/report")
assert report["completed_human_sessions"] == before
assert report["business_value_established"] is False
result = {"session_id":sid, "run_id":session["run_id"], "kind":"smoke", "finished":session["finished"],
          "decisions_saved":session["completed"], "remaining_decisions_exercised":len(seen),
          "blinding_checked":True, "prepared_handoffs_no_actions":True,
          "duplicate_save_409":bool(last), "human_sessions_before":before,
          "human_sessions_after":report["completed_human_sessions"], "business_value_established":False,
          "qualification":"Oracle-driven API/browser verification; not human accuracy or speed evidence"}
(ROOT/"reports/pilot_smoke.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
