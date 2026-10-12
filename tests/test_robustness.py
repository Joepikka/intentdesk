import csv
import json
from pathlib import Path

import pytest

from intentdesk.common import sha256_file
from intentdesk.robustness import DEFAULT_CASES, LABELS, SOURCES, export_template, load_cases, load_reviewed, metrics, run_suite


def payload():
    return {"version": "fixture_v1", "provenance": "assistant_authored_synthetic_development_cases",
            "label_status": "provisional_unreviewed", "cases": [
                {"id": "a", "text": "card missing fixture", "expected_label": "report_lost_card", "tags": ["plain"]},
                {"id": "b", "text": "toothbrush fixture", "expected_label": "oos", "tags": ["outside"]},
                {"id": "c", "text": "unclear fixture", "expected_label": "uncertain", "tags": ["ambiguous"]},
                {"id": "d", "text": "runtime failure fixture", "expected_label": "damaged_card", "tags": ["plain"]}]}


@pytest.fixture
def root(tmp_path):
    (tmp_path/"data/robustness").mkdir(parents=True)
    (tmp_path/DEFAULT_CASES).write_text(json.dumps(payload()), encoding="utf-8")
    source = tmp_path/"src/intentdesk"
    source.mkdir(parents=True)
    for name in SOURCES:
        (source/name).write_text("# fixture only\n", encoding="utf-8")
    return tmp_path


def test_shipped_suite_has_all_categories_and_explicit_synthetic_provenance():
    suite = load_cases(Path(__file__).parents[1]/DEFAULT_CASES)
    assert len(suite["cases"]) == 60
    assert {case["expected_label"] for case in suite["cases"]} == set(LABELS)
    assert {"negation", "typo", "multiple_intents", "keyword_trap", "missing_context"}.issubset(
        {tag for case in suite["cases"] for tag in case["tags"]})


@pytest.mark.parametrize("change", [
    lambda p: p["cases"].append(p["cases"][0]),
    lambda p: p["cases"][1].update(text="  CARD missing FIXTURE  "),
    lambda p: p["cases"][0].update(expected_label="not_a_category"),
    lambda p: p["cases"][0].update(tags=[]),
    lambda p: p.update(label_status="independently_verified"),
])
def test_invalid_or_misleading_suite_rejected(root, change):
    value = payload(); change(value)
    path = root/DEFAULT_CASES
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_label_template_blinds_draft_and_predictions_and_preserves_work(root):
    result = export_template(root)
    path = Path(result["label_sheet"])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert result["model_loaded"] is False
    assert not {"expected_label", "predicted_label", "proposed_label"}.intersection(rows[0])
    assert all(row["reviewed_label"] == row["reviewer"] == "" for row in rows)
    assert load_reviewed(path, payload()["cases"], sha256_file(root/DEFAULT_CASES)) == {}
    with pytest.raises(ValueError, match="already exists"):
        export_template(root)


def write_sheet(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_reviewed_labels_are_bound_to_exact_cases_and_require_reviewers(root):
    path = Path(export_template(root)["label_sheet"])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0].update(reviewed_label="report_fraud", reviewer="person01")
    write_sheet(path, rows)
    digest = sha256_file(root/DEFAULT_CASES)
    assert load_reviewed(path, payload()["cases"], digest) == {"a": "report_fraud"}
    for mutation in [{"reviewer": ""}, {"dataset_sha256": "stale"}, {"text": "changed"}, {"reviewed_label": "invalid"}]:
        original = rows[0].copy(); rows[0].update(mutation); write_sheet(path, rows)
        with pytest.raises(ValueError):
            load_reviewed(path, payload()["cases"], digest)
        rows[0] = original
    rows.append(rows[0]); write_sheet(path, rows)
    with pytest.raises(ValueError, match="repeated"):
        load_reviewed(path, payload()["cases"], digest)


def test_confusion_orientation_outside_errors_and_unreviewed_subset():
    rows = [
        {"provisional_label": "oos", "reviewed_label": None, "predicted_label": "report_lost_card"},
        {"provisional_label": "report_lost_card", "reviewed_label": "oos", "predicted_label": "error"},
        {"provisional_label": "uncertain", "reviewed_label": None, "predicted_label": "uncertain"}]
    result = metrics(rows, "provisional_label")
    assert result["samples"] == 3 and result["correct"] == 1
    assert result["label_agreement"] == pytest.approx(1/3)
    assert result["confusion_matrix"][LABELS.index("oos")][0] == 1
    assert result["confusion_matrix"][0][-1] == 1
    assert result["outside_suggested_supported_rate"] == 1
    assert result["runtime_errors"] == 1
    assert metrics(rows, "reviewed_label")["samples"] == 1
    assert metrics([], "reviewed_label")["label_agreement"] is None


class FakeRouter:
    def __init__(self, root):
        self.metadata = {"run_id": "fixture"}

    def route(self, text):
        if text == "runtime failure fixture":
            raise RuntimeError("fixture failure must not be hidden as a correct prediction")
        label = "report_lost_card"
        status = "supported_candidate"
        if text == "toothbrush fixture":
            status = "likely_out_of_scope"
        elif text == "unclear fixture":
            status = "uncertain"
        return {"action_executed": False, "route_recommendation": "review_required",
                "scope_assessment": {"status": status}, "candidates": [{"intent": label, "score": .9}]}


def test_end_to_end_report_retains_failures_without_touching_existing_results(root):
    (root/"reports").mkdir()
    protected = root/"reports/final_evaluation.json"
    protected.write_text('{"preserve":true}', encoding="utf-8")
    before = protected.read_bytes()
    result = run_suite(root, router_factory=FakeRouter)
    directory = Path(result["report"]).parent
    report = json.loads((directory/"summary.json").read_text())
    assert result["cases"] == 4 and result["runtime_errors"] == 1
    assert result["provisional_label_agreement"] == .75
    assert report["metrics"]["reviewer_declared_reference"]["samples"] == 0
    assert report["metrics"]["latency"]["successful_requests"] == 3
    assert report["original_benchmark_reopened"] is False
    assert report["reviewer_identity_or_independence_verified"] is False
    assert protected.read_bytes() == before and not (root/"local").exists()
    assert (directory/"robustness.png").stat().st_size > 1000
    assert "not independent test accuracy" in Path(result["report"]).read_text()
    assert "d" in (directory/"predictions.csv").read_text()
    another = run_suite(root, router_factory=FakeRouter)
    assert another["report"] != result["report"] and Path(result["report"]).exists()


def test_review_only_contract_violations_count_as_errors(root):
    class BadRouter(FakeRouter):
        def route(self, text):
            return {**super().route(text), "action_executed": True}
    result = run_suite(root, router_factory=BadRouter)
    assert result["runtime_errors"] == 4 and result["provisional_label_agreement"] == 0
