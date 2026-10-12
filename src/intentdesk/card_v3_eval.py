"""Freeze and consume the next synthetic check; preserve the unsuccessful V2 check."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import time

from intentdesk.card_v2 import CardRouter as LexicalRouter, CHECK_CASES as V2_CHECK
from intentdesk.card_v3 import CardRouter, CHECK_CASES
from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.pilot import suggest
from intentdesk.robustness import DEFAULT_CASES, LABELS, load_cases, metrics


def evaluate_card(root: Path, routers=None):
    if routers is None:
        from intentdesk.scope import ScopeRouter
        original = ScopeRouter(root)
        hybrid = CardRouter(root, shared_encoder=(original.tokenizer, original.encoder))
        routers = {"original_v1": original, "lexical_v2": LexicalRouter(root), "hybrid_v3": hybrid}
    hybrid = routers["hybrid_v3"]
    directory = root/"artifacts"/hybrid.metadata["run_id"]
    files = {"exposed_development_60": DEFAULT_CASES, "exposed_v2_check_48": V2_CHECK,
             "reserved_v3_synthetic_32": CHECK_CASES}
    binding = {"suite_sha256": {key: sha256_file(root/name) for key, name in files.items()},
               "v3_metadata_sha256": sha256_file(directory/"metadata.json"),
               "evaluator_sha256": sha256_file(Path(__file__)),
               "baseline_pointer_sha256": {name: sha256_file(root/"artifacts"/name) for name in
                    ("policy_latest.json", "scope_latest.json", "card_v2_latest.json")},
               "source_sha256": {s: sha256_file(root/"src/intentdesk"/s) for s in ("pilot.py", "scope.py", "card_v2_eval.py")},
               "protocol": "One paired pass per suite in fixed file order; original, lexical, hybrid; same request; no tuning after V3 check exposure"}
    if binding["suite_sha256"]["reserved_v3_synthetic_32"] != hybrid.metadata["binding"]["reserved_check_sha256"]:
        raise ValueError("V3 reserved check changed after training.")
    marker_path = directory/"check_experiment.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["status"] != "completed" or marker["binding"] != binding:
            raise ValueError("V3 check consumed or interrupted; preserve it and use fresh evaluation for a new version.")
        report_path = root/marker["report"]
        if sha256_file(report_path) != marker["report_sha256"]:
            raise ValueError("Completed V3 report changed.")
        return read_json(report_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root/"reports/card_v3_runs"/(hybrid.metadata["run_id"]+"_"+stamp)
    output.mkdir(parents=True, exist_ok=False)
    marker = {"status": "started", "check_exposed": True, "binding": binding}
    write_json(marker_path, marker)
    rows, summary = [], {}
    for suite_name, name in files.items():
        cases = load_cases(root/name)["cases"]
        groups = {name: [] for name in routers}
        for case in cases:
            for model_name, router in routers.items():
                started = time.perf_counter(); error = ""
                try:
                    result = router.route(case["text"])
                    if result.get("action_executed") is not False or result.get("route_recommendation") != "review_required":
                        raise ValueError("Review-only contract violated.")
                    label = suggest(result)["label"] if model_name == "original_v1" else result["label"]
                    if label not in LABELS:
                        raise ValueError("Unknown category.")
                except Exception as failure:
                    label, error = "error", type(failure).__name__
                row = {"suite": suite_name, "model": model_name, "case_id": case["id"], "text": case["text"],
                       "reference": case["expected_label"], "tags": case["tags"], "predicted_label": label,
                       "latency_ms": (time.perf_counter()-started)*1000, "error_type": error}
                groups[model_name].append(row); rows.append(row)
        summary[suite_name] = {model: {**metrics(group, "reference"),
            "phenomena": {tag: metrics([row for row in group if tag in row["tags"]], "reference")
                          for tag in sorted({tag for row in group for tag in row["tags"]})}}
            for model, group in groups.items()}
    report = {"run_id": output.name, "binding": binding, "training": hybrid.metadata,
              "human_reviewed_labels": 0, "independent_accuracy_claim": False, "automatic_actions": False,
              "provenance": "AI-authored synthetic references; not human-adjudicated or real customer traffic",
              "v2_check_status": "Consumed, inspected and now development data for V3",
              "v3_check_status": "Reserved before V3 training, now consumed; same-author synthetic cases limit independence",
              "original_benchmarks_reopened": False, "metrics": summary}
    write_json(output/"summary.json", report)
    with (output/"predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({**row, "tags": ";".join(row["tags"])} for row in rows)
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (suite, values) in zip(axes, summary.items()):
        names = list(values)
        axis.bar(names, [100*values[n]["label_agreement"] for n in names], color=["#84958a", "#7699ba", "#146146"])
        axis.set(ylim=(0, 110), ylabel="Provisional agreement (%)", title=suite.replace("_", " "), xticks=range(3), xticklabels=["Original", "Lexical", "Hybrid"])
        for i, name in enumerate(names):
            axis.text(i, 100*values[name]["label_agreement"]+2, f"{values[name]['correct']}/{values[name]['samples']}", ha="center")
    figure.suptitle("AI-authored / UNREVIEWED labels / synthetic evidence only")
    figure.tight_layout(); figure.savefig(output/"comparison.png", dpi=160); plt.close(figure)
    write_json(marker_path, {**marker, "status": "completed", "report": (output/"summary.json").relative_to(root).as_posix(),
                            "report_sha256": sha256_file(output/"summary.json")})
    return report
