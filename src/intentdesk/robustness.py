"""Synthetic development stress testing; never trains or reopens benchmark checks."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from intentdesk.common import sha256_file, write_json
from intentdesk.pilot import CATEGORIES, suggest
from intentdesk.predict import validate_text


LABELS = list(CATEGORIES)
DEFAULT_CASES = "data/robustness/card_support_v1.json"
SOURCES = ("robustness.py", "pilot.py", "scope.py", "policy.py", "predict.py", "neural.py")


def load_cases(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("provenance") != "assistant_authored_synthetic_development_cases"
            or payload.get("label_status") != "provisional_unreviewed"
            or not isinstance(payload.get("version"), str) or not payload["version"]
            or not isinstance(payload.get("cases"), list) or not payload["cases"]):
        raise ValueError("Use a non-empty, versioned synthetic development suite with provisional labels.")
    ids, texts = set(), set()
    for case in payload["cases"]:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]
                or case["id"] in ids or case.get("expected_label") not in LABELS):
            raise ValueError("Invalid or duplicate case identifier or expected category.")
        text = validate_text(case.get("text"))
        normalized = " ".join(text.casefold().split())
        tags = case.get("tags")
        if (normalized in texts or not isinstance(tags, list) or not tags
                or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
                or len(tags) != len(set(tags))):
            raise ValueError("Duplicate text or invalid phenomenon tags.")
        ids.add(case["id"]); texts.add(normalized)
    return payload


def export_template(root: Path, cases_path: Path | None = None, output: Path | None = None):
    cases_path = cases_path or root/DEFAULT_CASES
    payload = load_cases(cases_path)
    output = output or root/"local/robustness_labels.csv"
    if output.exists():
        raise ValueError("Label sheet already exists; choose another --output to preserve human work.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "dataset_sha256", "text", "tags", "reviewed_label", "reviewer", "note"])
        writer.writeheader()
        for case in payload["cases"]:
            writer.writerow({"case_id": case["id"], "dataset_sha256": sha256_file(cases_path),
                             "text": case["text"], "tags": ";".join(case["tags"]),
                             "reviewed_label": "", "reviewer": "", "note": ""})
    return {"label_sheet": str(output), "cases": len(payload["cases"]),
            "reviewed_labels": 0, "model_loaded": False,
            "instruction": "A person supplies labels before seeing predictions. Draft labels are deliberately omitted."}


def load_reviewed(path: Path | None, cases, dataset_hash):
    if path is None:
        return {}
    indexed = {case["id"]: case for case in cases}
    reviewed, seen = {}, set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "dataset_sha256", "text", "reviewed_label", "reviewer"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Review sheet is missing required columns.")
        for row in reader:
            identity = row["case_id"]
            if (identity not in indexed or identity in seen or row["dataset_sha256"] != dataset_hash
                    or row["text"] != indexed[identity]["text"]):
                raise ValueError("Review sheet has repeated, unknown, changed or stale cases.")
            seen.add(identity)
            label, reviewer = row["reviewed_label"].strip(), row["reviewer"].strip()
            if label:
                if label not in LABELS or not reviewer:
                    raise ValueError("Each reviewed category needs a valid label and reviewer code.")
                reviewed[identity] = label
            elif reviewer:
                raise ValueError("A reviewer code without a reviewed category is incomplete.")
    return reviewed


def metrics(rows, reference_key):
    rows = [row for row in rows if row.get(reference_key) is not None]
    matrix = np.zeros((len(LABELS), len(LABELS)+1), dtype=int)
    for row in rows:
        matrix[LABELS.index(row[reference_key]),
               LABELS.index(row["predicted_label"]) if row["predicted_label"] in LABELS else len(LABELS)] += 1
    per_class, f1s = {}, []
    for i, label in enumerate(LABELS):
        support, predicted, tp = int(matrix[i].sum()), int(matrix[:, i].sum()), int(matrix[i, i])
        precision = tp/predicted if predicted else 0.
        recall = tp/support if support else None
        f1 = 2*tp/(support+predicted) if support+predicted else 0.
        per_class[label] = {"support": support, "predicted_count": predicted, "precision": precision,
                            "recall": recall, "f1": f1, "zero_division": 0}
        if support:
            f1s.append(f1)
    supported = [row for row in rows if row[reference_key] not in ("oos", "uncertain")]
    outside = [row for row in rows if row[reference_key] == "oos"]
    unclear = [row for row in rows if row[reference_key] == "uncertain"]
    false_supported = sum(row["predicted_label"] in LABELS[:6] for row in outside)
    return {"samples": len(rows), "correct": sum(row[reference_key] == row["predicted_label"] for row in rows),
            "label_agreement": sum(row[reference_key] == row["predicted_label"] for row in rows)/len(rows) if rows else None,
            "macro_f1_present_reference_classes": float(np.mean(f1s)) if f1s else None,
            "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "matrix_rows": LABELS, "matrix_columns": LABELS+["error"],
            "supported_reference_count": len(supported),
            "supported_references_flagged_oos": sum(row["predicted_label"] == "oos" for row in supported),
            "outside_reference_count": len(outside), "outside_suggested_supported": false_supported,
            "outside_suggested_supported_rate": false_supported/len(outside) if outside else None,
            "unclear_reference_count": len(unclear),
            "unclear_suggested_specific_category": sum(row["predicted_label"] in LABELS[:6] for row in unclear),
            "needs_clarification_suggestions": sum(row["predicted_label"] == "uncertain" for row in rows),
            "runtime_errors": sum(row["predicted_label"] == "error" for row in rows)}


def summarize(rows):
    latencies = [row["latency_ms"] for row in rows if row["predicted_label"] != "error"]
    return {
        "provisional_reference": metrics(rows, "provisional_label"),
        "reviewer_declared_reference": metrics(rows, "reviewed_label"),
        "phenomena_provisional": {tag: metrics([row for row in rows if tag in row["tags"]], "provisional_label")
                                   for tag in sorted({tag for row in rows for tag in row["tags"]})},
        "latency": {"successful_requests": len(latencies), "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
                    "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
                    "scope": "One sequential local CPU pass after one warm-up; includes route and projection, excludes model startup; no production latency claim."},
        "unavailable_scope_count": sum(row["scope_status"] == "unavailable" for row in rows),
    }


def save_outputs(directory, report, rows):
    write_json(directory/"summary.json", report)
    with (directory/"predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "tags": ";".join(row["tags"])})
    metric = report["metrics"]["provisional_reference"]
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    matrix = np.array(metric["confusion_matrix"])
    axes[0].imshow(matrix, cmap="Blues", vmin=0)
    short = [CATEGORIES[label]["name"] for label in LABELS]
    axes[0].set(xticks=range(9), xticklabels=short+["Error"], yticks=range(8), yticklabels=short,
                xlabel="Suggested category", ylabel="Provisional reference", title="Synthetic development confusion matrix")
    plt.setp(axes[0].get_xticklabels(), rotation=55, ha="right", fontsize=8)
    for (i, j), value in np.ndenumerate(matrix):
        axes[0].text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black", fontsize=8)
    phenomena = report["metrics"]["phenomena_provisional"]
    tags = list(phenomena)
    axes[1].barh(tags, [100*phenomena[tag]["label_agreement"] for tag in tags], color="#146146")
    axes[1].set(xlim=(0, 100), xlabel="Agreement with provisional labels (%)", title="Overlapping phenomenon groups")
    fig.suptitle("Assistant-authored cases / UNREVIEWED references / no real-world accuracy claim", fontsize=11)
    fig.tight_layout(); fig.savefig(directory/"robustness.png", dpi=160); plt.close(fig)
    lines = ["# Synthetic development robustness check", "", f"Run: `{report['run_id']}`", "",
             "**Provisional, assistant-authored references. This is not independent test accuracy, human performance or evidence of customer savings.**", "",
             f"Cases: {metric['samples']}; agreement with provisional labels: {metric['correct']}/{metric['samples']} ({metric['label_agreement']:.1%}).",
             f"Reviewer-declared reference labels supplied: {report['metrics']['reviewer_declared_reference']['samples']}. Independence and reviewer identity are not verified.", "",
             "| Category | Reference count | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, values in metric["per_class"].items():
        recall = "n/a" if values["recall"] is None else f"{values['recall']:.3f}"
        lines.append(f"| {CATEGORIES[label]['name']} | {values['support']} | {values['precision']:.3f} | {recall} | {values['f1']:.3f} |")
    timing = report["metrics"]["latency"]
    lines += ["", f"Outside-desk examples suggested as one of the six supported categories: {metric['outside_suggested_supported']}/{metric['outside_reference_count']}.",
              f"Unclear/multiple-issue examples suggested as a specific category: {metric['unclear_suggested_specific_category']}/{metric['unclear_reference_count']}.",
              f"Runtime errors: {metric['runtime_errors']}. Unavailable scope: {report['metrics']['unavailable_scope_count']}.",
              f"Successful sequential requests: {timing['successful_requests']}; p50/p95 milliseconds: {timing['p50_ms']} / {timing['p95_ms']}.",
              "Model load and warm-up are recorded separately in JSON. Failed requests count as wrong; their elapsed time is retained in CSV and excluded from successful-request latency percentiles.", "",
              "![Synthetic development errors](robustness.png)", "", "## Cases needing inspection", ""]
    wrong = [row for row in rows if row["provisional_label"] != row["predicted_label"]]
    lines += [f"- `{row['case_id']}`: provisional `{row['provisional_label']}` → suggested `{row['predicted_label']}`; phenomena: {', '.join(row['tags'])}." for row in wrong]
    if not wrong:
        lines.append("No disagreement on this small synthetic suite. This does not establish generalization.")
    lines += ["", "Review reference categories and ambiguity before interpreting disagreements. Then collect fresh, representative, independently adjudicated examples. Do not tune against consumed final/scope checks, fabricate human sessions, or treat this development suite as untouched evaluation."]
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_suite(root: Path, cases_path: Path | None = None, labels_path: Path | None = None, router_factory=None):
    cases_path = cases_path or root/DEFAULT_CASES
    payload = load_cases(cases_path)
    dataset_hash = sha256_file(cases_path)
    reviewed = load_reviewed(labels_path, payload["cases"], dataset_hash)
    source_hashes = {name: sha256_file(root/"src/intentdesk"/name) for name in SOURCES}
    artifacts = {name: sha256_file(root/name) for name in ("artifacts/policy_latest.json", "artifacts/scope_latest.json", "uv.lock") if (root/name).is_file()}
    binding = {"dataset_sha256": dataset_hash, "source_sha256": source_hashes, "artifact_pointer_sha256": artifacts,
               "review_sheet_sha256": sha256_file(labels_path) if labels_path else None,
               "warm_up": "My debit card payment was rejected at a shop.", "order": "fixed file order; one pass"}
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if router_factory is None:
        from intentdesk.scope import ScopeRouter
        router_factory = ScopeRouter
    started = time.perf_counter(); router = router_factory(root); load_seconds = time.perf_counter()-started
    started = time.perf_counter(); router.route(binding["warm_up"]); warmup_seconds = time.perf_counter()-started
    rows = []
    for case in payload["cases"]:
        started = time.perf_counter()
        error, scope, top = "", "unavailable", ""
        try:
            analysis = router.route(case["text"])
            if analysis.get("action_executed") is not False or analysis.get("route_recommendation") != "review_required":
                raise ValueError("Review-only contract violated.")
            projected = suggest(analysis)["label"]
            scope = analysis.get("scope_assessment", {}).get("status", "unavailable")
            nearest = analysis.get("nearest_known_candidates", analysis.get("candidates", []))
            top = nearest[0]["intent"] if nearest else ""
        except Exception as failure:
            projected, error = "error", type(failure).__name__
        elapsed = (time.perf_counter()-started)*1000
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid monotonic elapsed time.")
        rows.append({"case_id": case["id"], "text": case["text"], "tags": case["tags"],
                     "provisional_label": case["expected_label"], "reviewed_label": reviewed.get(case["id"]),
                     "predicted_label": projected, "top_known_label": top, "scope_status": scope,
                     "latency_ms": elapsed, "error_type": error})
    directory = root/"reports/robustness"/(stamp+"_"+identity[:12])
    directory.mkdir(parents=True, exist_ok=False)
    report = {"run_id": directory.name, "binding_sha256": identity, "binding": binding,
              "suite_version": payload["version"], "provenance": payload["provenance"],
              "reference_status": payload["label_status"], "reviewer_identity_or_independence_verified": False,
              "policy_version": getattr(router, "metadata", {}).get("run_id"),
              "model_load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
              "automatic_actions": False, "business_value_established": False,
              "original_benchmark_reopened": False, "metrics": summarize(rows)}
    save_outputs(directory, report, rows)
    return {"report": str(directory/"report.md"), "cases": len(rows),
            "provisional_label_agreement": report["metrics"]["provisional_reference"]["label_agreement"],
            "reviewer_declared_labels": len(reviewed), "runtime_errors": report["metrics"]["provisional_reference"]["runtime_errors"],
            "independent_accuracy_claim": False, "business_value_established": False}
