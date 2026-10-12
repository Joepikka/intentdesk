import json

import pytest

from intentdesk.card_v2 import CHECK_CASES as V2_CHECK
from intentdesk.card_v3 import CHECK_CASES
from intentdesk.card_v3_eval import evaluate_card
from intentdesk.common import sha256_file, write_json
from intentdesk.robustness import DEFAULT_CASES


class Fake:
    def __init__(self, original=False):
        self.original = original; self.calls = 0
    def route(self, text):
        self.calls += 1
        base = {"action_executed": False, "route_recommendation": "review_required"}
        if self.original:
            return {**base, "scope_assessment": {"status": "likely_out_of_scope"}, "candidates": []}
        return {**base, "label": "uncertain"}


def setup(root):
    payload = {"version": "fixture", "provenance": "assistant_authored_synthetic_development_cases",
        "label_status": "provisional_unreviewed", "cases": [{"id": "a", "text": "fixture unclear request",
        "expected_label": "uncertain", "tags": ["missing_context"]}]}
    for path in (DEFAULT_CASES, V2_CHECK, CHECK_CASES):
        write_json(root/path, payload)
    for name in ("policy_latest.json", "scope_latest.json", "card_v2_latest.json"):
        write_json(root/"artifacts"/name, {"fixture": True})
    for name in ("pilot.py", "scope.py", "card_v2_eval.py"):
        target = root/"src/intentdesk"/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture", encoding="utf-8")
    write_json(root/"artifacts/hybrid_fixture/metadata.json", {"fixture": True})
    routers = {"original_v1": Fake(True), "lexical_v2": Fake(), "hybrid_v3": Fake()}
    routers["hybrid_v3"].metadata = {"run_id": "hybrid_fixture",
        "binding": {"reserved_check_sha256": sha256_file(root/CHECK_CASES)}}
    return routers


def test_reserved_check_is_consumed_once_and_cached_report_is_verified(tmp_path):
    routers = setup(tmp_path)
    report = evaluate_card(tmp_path, routers)
    assert all(router.calls == 3 for router in routers.values())
    assert report["metrics"]["reserved_v3_synthetic_32"]["hybrid_v3"]["correct"] == 1
    assert report["human_reviewed_labels"] == 0 and report["independent_accuracy_claim"] is False
    assert evaluate_card(tmp_path, routers) == report
    assert all(router.calls == 3 for router in routers.values())
    marker = json.loads((tmp_path/"artifacts/hybrid_fixture/check_experiment.json").read_text())
    marker["status"] = "started"
    write_json(tmp_path/"artifacts/hybrid_fixture/check_experiment.json", marker)
    with pytest.raises(ValueError, match="consumed or interrupted"):
        evaluate_card(tmp_path, routers)


def test_changed_reserved_check_never_calls_a_model(tmp_path):
    routers = setup(tmp_path)
    (tmp_path/CHECK_CASES).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="reserved check changed"):
        evaluate_card(tmp_path, routers)
    assert all(router.calls == 0 for router in routers.values())
