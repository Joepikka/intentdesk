"""The unattended wrapper records honest completion/failure without AI calls."""
import importlib.util
from pathlib import Path
import json
import pytest


def runner(tmp_path,monkeypatch):
    path=Path(__file__).parents[1]/"scripts/run_feedback_job.py"
    spec=importlib.util.spec_from_file_location("feedback_job_fixture",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,"__file__",str(tmp_path/"scripts/run_feedback_job.py"))
    monkeypatch.setattr(module,"start_extension",lambda root:{"run_id":"fixture"})
    monkeypatch.setattr(module,"train",lambda root:None)
    monkeypatch.setattr(module,"directory_for",lambda root:root/"artifacts/fixture")
    monkeypatch.setattr(module,"read_json",lambda path:{"report":"reports/fixture/summary.json"})
    return module


def test_unattended_job_saves_completed_status_and_actual_result(tmp_path,monkeypatch):
    module=runner(tmp_path,monkeypatch)
    monkeypatch.setattr(module,"evaluate",lambda root:{"selected_classifier":"fixture",
        "classification":{"fixture":{"accuracy":.7,"macro_f1":.6}}})
    module.main()
    status=json.loads((tmp_path/"local/feedback_job/status.json").read_text())
    assert status["state"]=="completed" and status["macro_f1"]==.6
    assert status["automatic_actions"] is False


def test_unattended_failure_is_not_reported_as_completion(tmp_path,monkeypatch):
    module=runner(tmp_path,monkeypatch)
    def fail(root):
        raise RuntimeError("fixture training failure")
    monkeypatch.setattr(module,"train",fail)
    with pytest.raises(RuntimeError):
        module.main()
    status=json.loads((tmp_path/"local/feedback_job/status.json").read_text())
    assert status["state"]=="failed" and "accuracy" not in status
