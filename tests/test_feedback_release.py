from types import SimpleNamespace
import pytest

from intentdesk.feedback_release import FeedbackRelease,improvement_report
from intentdesk.common import write_json,sha256_file
from intentdesk import feedback_v2


class Base:
    def __init__(self,root):
        pass
    def analyze(self,text):
        return {"experiment":"original","selected_classifier":"streaming_sgd","predicted_stars":2,
            "linear_regression_stars":2.1,"scratch_predicted_stars":1,"cluster":7,
            "automatic_actions":False,"human_review_required":True}


def test_checked_improvement_updates_classifier_but_preserves_other_methods(tmp_path):
    texts=[]
    def improved(root):
        return SimpleNamespace(metadata={"run_id":"new","selected_classifier":"word_char_balanced"},
            predict=lambda values:texts.extend(values) or [1])
    release=FeedbackRelease(tmp_path,Base,improved,lambda root:{"selected_classifier":"word_char_balanced"})
    result=release.analyze("one star this fictional product broke quickly")
    assert result["predicted_stars"]==1 and result["original_predicted_stars"]==2
    assert result["linear_regression_stars"]==2.1 and result["scratch_predicted_stars"]==1
    assert result["automatic_actions"] is False and result["release"]=="feedback_v2"
    assert "one star" not in texts[0]


def test_missing_or_retained_original_release_keeps_original_predictions(tmp_path):
    for report in (None,{"selected_classifier":"original_streaming"}):
        release=FeedbackRelease(tmp_path,Base,lambda root:pytest.fail("must not load new model"),lambda root:report)
        assert release.analyze("fictional review text")["selected_classifier"]=="streaming_sgd"
    assert improvement_report(tmp_path) is None


def test_completed_report_identity_and_path_are_checked(tmp_path,monkeypatch):
    directory=tmp_path/"artifacts/new";directory.mkdir(parents=True)
    write_json(tmp_path/"artifacts/feedback_v2_latest.json",{})
    metadata={"selected_classifier":"word_char_balanced"};write_json(directory/"metadata.json",metadata)
    monkeypatch.setattr(feedback_v2,"current_directory",lambda root:directory)
    monkeypatch.setattr(feedback_v2,"checked",lambda root:(directory,metadata))
    path=tmp_path/"reports/feedback_v2_runs/fixture/summary.json"
    report={"selected_classifier":"word_char_balanced","selection_unchanged_after_test":True}
    write_json(path,report)
    marker={"status":"completed","binding":{"metadata_sha256":sha256_file(directory/"metadata.json")},
        "report":path.relative_to(tmp_path).as_posix(),"report_sha256":sha256_file(path)}
    write_json(directory/"final_experiment.json",marker)
    assert improvement_report(tmp_path)==report
    path.write_text("changed")
    with pytest.raises(ValueError,match="report changed"):
        improvement_report(tmp_path)


def test_incomplete_improvement_is_not_deployed(tmp_path,monkeypatch):
    directory=tmp_path/"artifacts/new";directory.mkdir(parents=True)
    write_json(tmp_path/"artifacts/feedback_v2_latest.json",{})
    monkeypatch.setattr(feedback_v2,"current_directory",lambda root:directory)
    write_json(directory/"final_experiment.json",{"status":"started"})
    assert improvement_report(tmp_path) is None
