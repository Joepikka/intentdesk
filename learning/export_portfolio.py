"""Export fixed milestone evidence, excluding raw data, inputs and model weights."""

from pathlib import Path
import shutil

from intentdesk.common import read_json, sha256_file, write_json

root = Path(__file__).resolve().parents[1]
neural = "neural_20261009T204150600532Z"
policy = "policy_20261010T043806079244Z"
figures = root/"reports/figures"
figures.mkdir(parents=True, exist_ok=True)
images = {}
for name, run, source in [("milestone_02_neural.png", neural, "neural_comparison.png"),
                          ("milestone_03_calibration.png", policy, "calibration_and_routing.png")]:
    shutil.copyfile(root/"reports/runs"/run/source, figures/name)
    images[name] = {"source_run": run, "sha256": sha256_file(figures/name)}
write_json(root/"reports/portfolio_evidence.json", {
    "neural_run": neural, "policy_run": policy, "test_evaluated": False,
    "neural_development_metrics": read_json(root/"reports/runs"/neural/"metrics.json"),
    "calibration_and_independent_check": read_json(root/"reports/runs"/policy/"calibration.json"),
    "policy_check": read_json(root/"reports/runs"/policy/"policy_check.json"),
    "shareable_figures": images,
    "scope": "Fixed milestone snapshots; no raw dataset or model weights. Primary unsupported-request gate failed; active routing remains review-all."
})
print("Shareable figures and evidence exported to reports/.")
