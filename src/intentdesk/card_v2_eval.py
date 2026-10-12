"""One-shot, bound comparison; new synthetic checks are still not independent data."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from intentdesk.card_v2 import CardRouter, CHECK_CASES
from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.pilot import suggest
from intentdesk.robustness import DEFAULT_CASES, LABELS, load_cases, metrics


def evaluate_card(root: Path, baseline_factory=None, card_factory=None):
    card = (card_factory or CardRouter)(root)
    directory = root/"artifacts"/card.metadata["run_id"]
    check_hash = sha256_file(root/CHECK_CASES)
    if check_hash != card.metadata["binding"]["reserved_check_sha256"]:
        raise ValueError("Reserved synthetic check changed after training protocol was frozen.")
    binding = {"model_sha256": card.metadata["model_sha256"], "reserved_check_sha256": check_hash,
               "development_sha256": sha256_file(root/DEFAULT_CASES),
               "baseline_source_sha256": {p: sha256_file(root/"src/intentdesk"/p) for p in ("scope.py", "policy.py", "pilot.py")},
               "baseline_pointer_sha256": {p: sha256_file(root/p) for p in
                   ("artifacts/policy_latest.json", "artifacts/scope_latest.json")},
               "evaluator_sha256": sha256_file(Path(__file__)),
               "protocol": "One sequential paired pass; same text to each model; development then reserved synthetic check; no tuning after check exposure"}
    marker_path = directory/"check_experiment.json"
    if marker_path.exists():
        previous = read_json(marker_path)
        if previous["binding"] != binding or previous["status"] != "completed":
            raise ValueError("Check already exposed or binding changed. Preserve results; use a new protocol and new check.")
        report_path = root/previous["report"]
        if sha256_file(report_path) != previous["report_sha256"]:
            raise ValueError("Stored card comparison report changed.")
        return read_json(report_path)
    if baseline_factory is None:
        from intentdesk.scope import ScopeRouter
        baseline_factory = ScopeRouter
    baseline = baseline_factory(root)
    suites = {"exposed_development_60": load_cases(root/DEFAULT_CASES)["cases"],
              "reserved_synthetic_48": load_cases(root/CHECK_CASES)["cases"]}
    run_id = card.metadata["run_id"]+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root/"reports/card_v2_runs"/run_id
    output.mkdir(parents=True, exist_ok=False)
    marker = {"binding": binding, "status": "started", "check_exposed": True}
    write_json(marker_path, marker)  # Refuse a second exposure after an interruption.
    rows, summary = [], {}
    for suite_name, cases in suites.items():
        evaluated = {"original_v1": [], "card_v2": []}
        for case in cases:
            for name, router in (("original_v1", baseline), ("card_v2", card)):
                started = time.perf_counter()
                error = ""
                try:
                    analysis = router.route(case["text"])
                    if analysis.get("action_executed") is not False or analysis.get("route_recommendation") != "review_required":
                        raise ValueError("Review-only contract violated.")
                    label = suggest(analysis)["label"] if name == "original_v1" else analysis["label"]
                    if label not in LABELS:
                        raise ValueError("Unknown category.")
                except Exception as failure:
                    label, error = "error", type(failure).__name__
                row = {"suite": suite_name, "model": name, "case_id": case["id"],
                       "text": case["text"], "tags": case["tags"], "reference": case["expected_label"],
                       "predicted_label": label, "latency_ms": (time.perf_counter()-started)*1000, "error_type": error}
                rows.append(row); evaluated[name].append(row)
        summary[suite_name] = {}
        for name, group in evaluated.items():
            times = [row["latency_ms"] for row in group if row["predicted_label"] != "error"]
            summary[suite_name][name] = {**metrics(group, "reference"),
                "phenomena": {tag: metrics([row for row in group if tag in row["tags"]], "reference")
                              for tag in sorted({tag for row in group for tag in row["tags"]})},
                "p50_ms": float(np.percentile(times, 50)) if times else None,
                "p95_ms": float(np.percentile(times, 95)) if times else None}
    report = {"run_id": run_id, "binding": binding, "training": card.metadata,
              "provenance": "AI-authored synthetic references, unreviewed by a person",
              "human_reviewed_labels": 0, "independent_accuracy_claim": False,
              "original_benchmarks_reopened": False, "automatic_actions": False,
              "timing_scope": "One sequential paired local CPU pass; first requests may be cold; not production latency or a controlled timing benchmark",
              "metrics": summary}
    write_json(output/"summary.json", report)
    with (output/"predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({**row, "tags": ";".join(row["tags"])} for row in rows)
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (suite, values) in zip(axes, summary.items()):
        names = list(values)
        axis.bar(names, [100*values[n]["label_agreement"] for n in names], color=["#84958a", "#146146"])
        axis.set(ylim=(0, 110), ylabel="Agreement with provisional labels (%)", title=suite.replace("_", " "))
        for i, name in enumerate(names):
            axis.text(i, 100*values[name]["label_agreement"]+2, f"{values[name]['correct']}/{values[name]['samples']}", ha="center")
    figure.suptitle("AI-authored / UNREVIEWED references / development evidence only")
    figure.tight_layout(); figure.savefig(output/"comparison.png", dpi=160); plt.close(figure)
    report_relative = (output/"summary.json").relative_to(root).as_posix()
    write_json(marker_path, {**marker, "status": "completed", "report": report_relative,
                             "report_sha256": sha256_file(output/"summary.json")})
    return report
