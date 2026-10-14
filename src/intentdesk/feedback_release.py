"""Compose checked feedback releases without changing their frozen model sources."""
from pathlib import Path

from intentdesk.common import read_json,sha256_file
from intentdesk.feedback_data import clean_rating_declarations
from intentdesk.predict import validate_text


def improvement_report(root):
    if not (root/"artifacts/feedback_v2_latest.json").exists():
        return None
    from intentdesk.feedback_v2 import current_directory,checked
    directory=current_directory(root)
    marker_path=directory/"final_experiment.json"
    if not marker_path.exists() or read_json(marker_path).get("status")!="completed":
        return None
    directory,metadata=checked(root)
    marker=read_json(marker_path)
    if marker["binding"]["metadata_sha256"]!=sha256_file(directory/"metadata.json"):
        raise ValueError("Completed V2 report metadata changed.")
    path=(root/marker["report"]).resolve()
    if not path.is_relative_to((root/"reports/feedback_v2_runs").resolve()) or sha256_file(path)!=marker["report_sha256"]:
        raise ValueError("Completed V2 report changed or has an invalid path.")
    report=read_json(path)
    if report["selected_classifier"]!=metadata["selected_classifier"] or report["selection_unchanged_after_test"] is not True:
        raise ValueError("V2 report changed the preselected classifier.")
    return report


class FeedbackRelease:
    def __init__(self,root,base_factory=None,improvement_factory=None,report_loader=improvement_report):
        if base_factory is None:
            from intentdesk.feedback_predict import FeedbackModels
            base_factory=FeedbackModels
        self.base=base_factory(root)
        report=report_loader(root)
        self.improved=None
        if report and report["selected_classifier"]!="original_streaming":
            if improvement_factory is None:
                from intentdesk.feedback_v2 import FeedbackV2
                improvement_factory=FeedbackV2
            self.improved=improvement_factory(root)

    def analyze(self,text):
        result={**self.base.analyze(text),"release":"original_feedback_lab"}
        if self.improved is not None:
            cleaned=clean_rating_declarations(validate_text(text))
            result.update(original_classifier=result["selected_classifier"],original_predicted_stars=result["predicted_stars"],
                original_experiment=result["experiment"],experiment=self.improved.metadata["run_id"],
                selected_classifier=self.improved.metadata["selected_classifier"],
                predicted_stars=int(self.improved.predict([cleaned])[0]),release="feedback_v2")
        return result
