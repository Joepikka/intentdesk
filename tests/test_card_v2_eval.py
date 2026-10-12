import json
from pathlib import Path

import pytest

from intentdesk.card_v2 import CHECK_CASES
from intentdesk.card_v2_eval import evaluate_card
from intentdesk.common import sha256_file, write_json
from intentdesk.robustness import DEFAULT_CASES


def setup(root):
    payload = {"version": "fixture", "provenance": "assistant_authored_synthetic_development_cases",
        "label_status": "provisional_unreviewed", "cases": [{"id": "missing", "text": "fixture missing card",
        "expected_label": "report_lost_card", "tags": ["plain"]}, {"id": "failure", "text": "fixture failure",
        "expected_label": "uncertain", "tags": ["missing_context"]}]}
    for path in (DEFAULT_CASES, CHECK_CASES):
        write_json(root/path, payload)
    for name in ("policy_latest.json", "scope_latest.json"):
        write_json(root/"artifacts"/name, {"fixture": True})
    (root/"src/intentdesk").mkdir(parents=True)
    for name in ("scope.py", "policy.py", "pilot.py"):
        (root/"src/intentdesk"/name).write_text("fixture", encoding="utf-8")
    (root/"artifacts/fixture").mkdir()


class Card:
    calls = 0
    def __init__(self, root):
        self.metadata = {"run_id": "fixture", "model_sha256": "fixture",
            "binding": {"reserved_check_sha256": sha256_file(root/CHECK_CASES)}}
    def route(self, text):
        type(self).calls += 1
        return {"label": "uncertain" if "failure" in text else "report_lost_card",
                "action_executed": False, "route_recommendation": "review_required"}


class Baseline:
    def __init__(self, root):
        pass
    def route(self, text):
        if "failure" in text:
            raise RuntimeError("fixture")
        return {"scope_assessment": {"status": "supported_candidate"},
                "candidates": [{"intent": "report_lost_card"}],
                "action_executed": False, "route_recommendation": "review_required"}


def test_paired_comparison_counts_errors_and_reuses_completed_check(tmp_path):
    setup(tmp_path); Card.calls = 0
    report = evaluate_card(tmp_path, Baseline, Card)
    assert Card.calls == 4 and report["human_reviewed_labels"] == 0
    result = report["metrics"]["reserved_synthetic_48"]
    assert result["original_v1"]["runtime_errors"] == 1
    assert result["card_v2"]["correct"] == 2
    assert report["independent_accuracy_claim"] is False
    assert evaluate_card(tmp_path, Baseline, Card) == report and Card.calls == 4
    marker_path = tmp_path/"artifacts/fixture/check_experiment.json"
    marker = json.loads(marker_path.read_text())
    (tmp_path/marker["report"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="report changed"):
        evaluate_card(tmp_path, Baseline, Card)


def test_interrupted_or_changed_binding_is_not_reexposed(tmp_path):
    setup(tmp_path)
    report = evaluate_card(tmp_path, Baseline, Card)
    path = tmp_path/"artifacts/fixture/check_experiment.json"
    marker = json.loads(path.read_text()); marker["status"] = "started"
    write_json(path, marker)
    with pytest.raises(ValueError, match="already exposed"):
        evaluate_card(tmp_path, Baseline, Card)
    marker["status"] = "completed"; write_json(path, marker)
    (tmp_path/"src/intentdesk/pilot.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        evaluate_card(tmp_path, Baseline, Card)
