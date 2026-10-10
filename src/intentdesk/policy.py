"""Frozen-model calibration, separately checked abstention, and local recommendations."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import psutil
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import beta
import torch
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.neural import (CONFIG, REVISION, configure, encode_batch, load_encoder,
                               make_head, validate_feature_cache)
from intentdesk.predict import validate_text
from intentdesk.train import classification_metrics, verify_development_data

PROTOCOL = {
    "version": 1, "candidates": ["B2_MPNet_linear", "N1_small"], "seed": 20261010,
    "temperature_bounds": [.05, 20.], "validation_selection_fraction": .2,
    "confidence_thresholds": [.0, .5, .6, .7, .8, .85, .9, .92, .94, .96, .98, .99, .995, 1.1],
    "cosine_thresholds": [0., .3, .4, .5, .6, .65, .7, .75, .8],
    "oos_prevalence": .2, "wrong_cost": 10., "review_cost": 1.,
    "minimum_coverage": .4, "maximum_selective_error": .05, "maximum_oos_acceptance": .05,
    "selection": "minimum weighted selection cost, then maximum coverage, then deterministic model/threshold order",
    "check_alpha": .05, "check_bounds": "three one-sided Clopper-Pearson bounds with Bonferroni alpha/3",
    "bootstrap_repetitions": 1000, "test_evaluated": False,
}


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def artifact_directory(root: Path, run_id: str) -> Path:
    artifacts = (root / "artifacts").resolve()
    directory = (artifacts / run_id).resolve()
    if directory == artifacts or not directory.is_relative_to(artifacts):
        raise ValueError("Invalid artifact path.")
    return directory


def split_validation(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed stratified 20/80 split; duplicate groups may not cross it."""
    if len({r["duplicate_group"] for r in rows}) != len(rows):
        raise ValueError("Validation duplicates need an explicit grouped-split audit first.")
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[row["intent"]].append(i)
    rng = np.random.default_rng(PROTOCOL["seed"])
    selection, check = [], []
    for _, indices in sorted(buckets.items()):
        indices = sorted(indices, key=lambda i: rows[i]["example_id"])
        indices = rng.permutation(indices)
        count = max(1, round(len(indices) * PROTOCOL["validation_selection_fraction"]))
        if count >= len(indices):
            raise ValueError("Too few validation rows for independent policy checking.")
        selection.extend(indices[:count].tolist())
        check.extend(indices[count:].tolist())
    return np.array(sorted(selection)), np.array(sorted(check))


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    if (logits.ndim != 2 or len(logits) != len(labels) or not len(labels)
            or not np.isfinite(logits).all() or (labels < 0).any() or (labels >= logits.shape[1]).any()):
        raise ValueError("Invalid calibration logits/labels.")
    bounds = np.log(PROTOCOL["temperature_bounds"])
    before = nll(logits, labels, 1.)
    optimized = minimize_scalar(lambda t: nll(logits, labels, float(np.exp(t))),
                                bounds=tuple(bounds), method="bounded", options={"xatol": 1e-7})
    if not optimized.success or not np.isfinite(optimized.fun):
        raise ValueError("Temperature fitting failed.")
    temperature = float(np.exp(optimized.x)) if optimized.fun < before else 1.
    return {"temperature": temperature, "fit_split": "calibrate", "nll_before": before,
            "nll_after": nll(logits, labels, temperature),
            "near_search_boundary": bool(min(abs(optimized.x-bounds)) < .02)}


def reliability(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict:
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    assignment = np.minimum((confidence * bins).astype(int), bins-1)
    result, ece = [], 0.
    for i in range(bins):
        mask = assignment == i
        if mask.any():
            average, accuracy = float(confidence[mask].mean()), float(correct[mask].mean())
            ece += mask.mean() * abs(average-accuracy)
            result.append({"bin": i, "count": int(mask.sum()), "confidence": average, "accuracy": accuracy})
    return {"equal_width_bins": bins, "ece": float(ece), "bins": result}


def binomial_upper(errors: int, total: int, alpha: float) -> float:
    return 1. if not total or errors == total else float(beta.ppf(1-alpha, errors+1, total-errors))


def binomial_lower(successes: int, total: int, alpha: float) -> float:
    return 0. if not total or not successes else float(beta.ppf(alpha, successes, total-successes+1))


def acceptance(probabilities, similarities, confidence_threshold, cosine_threshold):
    return (probabilities.max(1) >= confidence_threshold) & (similarities >= cosine_threshold)


def policy_metrics(probabilities, similarities, labels, confidence_threshold, cosine_threshold) -> dict:
    accepted = acceptance(probabilities, similarities, confidence_threshold, cosine_threshold)
    known = labels >= 0
    n_known, n_oos = int(known.sum()), int((~known).sum())
    if not n_known or not n_oos:
        raise ValueError("Routing evaluation needs both in-scope and OOS requests.")
    accepted_known = int((accepted & known).sum())
    wrong_known = int((accepted & known & (probabilities.argmax(1) != labels)).sum())
    accepted_oos = int((accepted & ~known).sum())
    prevalence = PROTOCOL["oos_prevalence"]
    coverage = (1-prevalence)*accepted_known/n_known + prevalence*accepted_oos/n_oos
    wrong_mass = (1-prevalence)*wrong_known/n_known + prevalence*accepted_oos/n_oos
    return {"in_scope_count": n_known, "oos_count": n_oos, "accepted_in_scope": accepted_known,
            "wrong_accepted_in_scope": wrong_known, "accepted_oos": accepted_oos,
            "weighted_coverage": coverage, "weighted_selective_error": wrong_mass/coverage if coverage else None,
            "oos_false_acceptance": accepted_oos/n_oos,
            "weighted_cost": PROTOCOL["wrong_cost"]*wrong_mass + PROTOCOL["review_cost"]*(1-coverage)}


def check_gate(metrics: dict) -> dict:
    alpha = PROTOCOL["check_alpha"] / 3
    q_upper = binomial_upper(metrics["wrong_accepted_in_scope"], metrics["accepted_in_scope"], alpha)
    oos_upper = binomial_upper(metrics["accepted_oos"], metrics["oos_count"], alpha)
    known_accept_lower = binomial_lower(metrics["accepted_in_scope"], metrics["in_scope_count"], alpha)
    coverage_lower = (1-PROTOCOL["oos_prevalence"])*known_accept_lower
    # Known-class error contributes at most q; OOS error mass at most prevalence*a.
    error_upper = min(1., q_upper + PROTOCOL["oos_prevalence"]*oos_upper/coverage_lower) if coverage_lower else 1.
    requirements = {"coverage_lower_bound": coverage_lower >= PROTOCOL["minimum_coverage"],
                    "selective_error_upper_bound": error_upper <= PROTOCOL["maximum_selective_error"],
                    "oos_acceptance_upper_bound": oos_upper <= PROTOCOL["maximum_oos_acceptance"],
                    "point_cost_below_all_review": metrics["weighted_cost"] < 1.}
    return {"passed": all(requirements.values()), "requirements": requirements,
            "simultaneous_confidence": 1-PROTOCOL["check_alpha"],
            "known_conditional_error_upper": q_upper, "oos_acceptance_upper": oos_upper,
            "weighted_coverage_lower": coverage_lower, "weighted_selective_error_upper": error_upper,
            "assumptions": "Fixed policy; independent unique requests within IS/OOS strata; fixed 20% OOS scenario. Not a production guarantee."}


def choose_policy(candidates: dict, labels: np.ndarray) -> tuple[dict, list[dict]]:
    trials, eligible = [], []
    for name, (probabilities, similarities) in candidates.items():
        for confidence in PROTOCOL["confidence_thresholds"]:
            for cosine in PROTOCOL["cosine_thresholds"]:
                metrics = policy_metrics(probabilities, similarities, labels, confidence, cosine)
                trial = {"model": name, "confidence_threshold": confidence, "cosine_threshold": cosine, **metrics}
                trials.append(trial)
                if (metrics["weighted_coverage"] >= PROTOCOL["minimum_coverage"]
                        and metrics["weighted_selective_error"] is not None
                        and metrics["weighted_selective_error"] <= PROTOCOL["maximum_selective_error"]
                        and metrics["oos_false_acceptance"] <= PROTOCOL["maximum_oos_acceptance"]
                        and metrics["weighted_cost"] < 1):
                    eligible.append(trial)
    if not eligible:
        return {"model": PROTOCOL["candidates"][0], "confidence_threshold": 1.1,
                "cosine_threshold": 0., "selection_passed": False}, trials
    selected = min(eligible, key=lambda t: (t["weighted_cost"], -t["weighted_coverage"],
                                          t["model"], t["confidence_threshold"], t["cosine_threshold"]))
    return {**selected, "selection_passed": True}, trials


class FrozenModels:
    def __init__(self, root: Path, parent_run: str, hashes: dict | None = None):
        configure()
        self.directory = artifact_directory(root, parent_run)
        self.metadata = read_json(self.directory / "metadata.json")
        if self.metadata["status"] != "completed" or self.metadata["revision"] != REVISION or self.metadata["config"] != CONFIG:
            raise ValueError("Frozen parent model identity/configuration changed.")
        self.classes = self.metadata["classes"]
        if len(self.classes) != 150 or len(set(self.classes)) != 150:
            raise ValueError("Invalid parent classes.")
        self.hashes = {name: sha256_file(self.directory / name) for name in ("linear.joblib", "N1_small.pt")}
        if hashes is not None and self.hashes != hashes:
            raise ValueError("Frozen model checksum mismatch.")
        expected_linear = self.metadata.get("linear_sha256")
        if expected_linear and expected_linear != self.hashes["linear.joblib"]:
            raise ValueError("Linear model checksum mismatch.")
        checkpoint = self.metadata["checkpoints"]["N1_small"]
        if checkpoint["sha256"] != self.hashes["N1_small.pt"]:
            raise ValueError("Small neural model checksum mismatch.")
        self.linear = joblib.load(self.directory / "linear.joblib")
        if list(self.linear.classes_) != self.classes or self.linear.n_features_in_ != 768:
            raise ValueError("Linear class/feature contract mismatch.")
        self.head = make_head(checkpoint["widths"]).eval()
        self.head.load_state_dict(torch.load(self.directory / "N1_small.pt", map_location="cpu", weights_only=True))

    def logits(self, name: str, features: np.ndarray) -> np.ndarray:
        if name == "B2_MPNet_linear":
            with threadpool_limits(limits=2):
                return self.linear.decision_function(features).astype(np.float64)
        if name != "N1_small":
            raise ValueError("Unknown frozen candidate.")
        with torch.inference_mode():
            return self.head(torch.from_numpy(features)).numpy().astype(np.float64)


def read_policy_data(root: Path):
    fit, tune, audit = verify_development_data(root)
    rows = {"fit": fit, "tune": tune}
    classes = {r["intent"] for r in fit}
    for split in ("calibrate", "validation"):
        path = root / "data" / "prepared" / (split + ".jsonl")
        if sha256_file(path) != audit["prepared_sha256"][path.name]:
            raise ValueError(f"Changed {split} partition.")
        rows[split] = read_records(path)
        if len(rows[split]) != audit["counts"][split] or not rows[split]:
            raise ValueError("Policy partition count mismatch.")
        for r in rows[split]:
            allowed = r["source_split"] == "train" and not r["is_oos"] if split == "calibrate" else r["source_split"] in ("val", "oos_val")
            if (not allowed or r["working_split"] != split or not r["text"].strip()
                    or r["is_oos"] != (r["intent"] == "oos")
                    or (not r["is_oos"] and r["intent"] not in classes)):
                raise ValueError("Policy partition schema/provenance mismatch.")
    seen = set()
    for split, records in rows.items():
        groups = {r["duplicate_group"] for r in records}
        if len(groups) != len(records) or seen & groups:
            raise ValueError("Policy partitions contain duplicate leakage.")
        seen |= groups
    return rows, audit


def policy_embeddings(root: Path, rows: dict, audit: dict):
    tokenizer, encoder, _ = load_encoder(root)
    directory = root / "data" / "embeddings" / "policy"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    identity = {"revision": REVISION, "encoder_config": CONFIG,
                "encoder_manifest_sha256": sha256_file(root / "models" / "mpnet" / "manifest.json"),
                "input_sha256": {s: audit["prepared_sha256"][s+".jsonl"] for s in ("calibrate", "validation")}}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if any(manifest[k] != v for k, v in identity.items()):
            raise ValueError("Policy embedding identity changed.")
        return {s: validate_feature_cache(directory/(s+".npy"), len(rows[s]), manifest["sha256"][s])
                for s in ("calibrate", "validation")}, manifest
    started, arrays, truncated = time.perf_counter(), {}, {}
    for split in ("calibrate", "validation"):
        features = np.empty((len(rows[split]), CONFIG["dimension"]), dtype=np.float32)
        for start in range(0, len(features), CONFIG["embedding_batch"]):
            stop = min(start+CONFIG["embedding_batch"], len(features))
            features[start:stop] = encode_batch(tokenizer, encoder, [r["text"] for r in rows[split][start:stop]])
            if stop % 256 == 0 or stop == len(features):
                print(f"Policy embeddings {split}: {stop:,}/{len(features):,}", flush=True)
        lengths = tokenizer([r["text"] for r in rows[split]], truncation=False)["input_ids"]
        truncated[split] = sum(len(ids) > CONFIG["max_length"] for ids in lengths)
        if truncated[split]:
            raise ValueError("Policy data exceeds the serving token cap; define an exclusion policy before evaluation.")
        np.save(directory/(split+".npy"), features, allow_pickle=False)
        arrays[split] = features
    manifest = {**identity, "sha256": {s: sha256_file(directory/(s+".npy")) for s in arrays},
                "truncated": truncated, "encoding_seconds": time.perf_counter()-started,
                "process_peak_working_set_bytes": psutil.Process().memory_info().peak_wset,
                "test_evaluated": False}
    write_json(manifest_path, manifest)
    return arrays, manifest


def make_prototypes(features, fit_rows, classes):
    labels = np.array([r["intent"] for r in fit_rows])
    prototypes = np.stack([features[labels == c].mean(0, dtype=np.float64) for c in classes])
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if not np.isfinite(prototypes).all() or (norms <= 0).any():
        raise ValueError("Empty intent prototype.")
    return (prototypes / norms).astype(np.float32)


def similarities(features, probabilities, prototypes):
    return np.clip(np.sum(features * prototypes[probabilities.argmax(1)], axis=1), -1., 1.)


def bootstrap_cost(probabilities, support, labels, selected):
    accepted = acceptance(probabilities, support, selected["confidence_threshold"], selected["cosine_threshold"])
    wrong = accepted & (probabilities.argmax(1) != labels)
    costs = np.where(accepted, np.where(wrong, PROTOCOL["wrong_cost"], 0.), PROTOCOL["review_cost"])
    known, unknown = costs[labels >= 0], costs[labels < 0]
    rng, values = np.random.default_rng(PROTOCOL["seed"]), []
    prevalence = PROTOCOL["oos_prevalence"]
    for _ in range(PROTOCOL["bootstrap_repetitions"]):
        values.append((1-prevalence) * known[rng.integers(len(known), size=len(known))].mean()
                      + prevalence * unknown[rng.integers(len(unknown), size=len(unknown))].mean())
    return {"weighted_cost_percentile_95": np.percentile(values, [2.5, 97.5]).tolist(),
            "paired_difference_vs_all_review_95": (np.percentile(values, [2.5, 97.5])-1).tolist(),
            "resampling": "IS and OOS separately, fixed 20% mixture, fixed checked policy",
            "limitation": "Conditional bootstrap misses unseen OOS errors when zero are observed; use the exact OOS upper bound too."}


def calibration_summary(logits, labels, classes, temperature):
    result = {}
    for name, t in (("before", 1.), ("after", temperature)):
        probabilities = softmax(logits/t, axis=1)
        result[name] = {**classification_metrics([classes[i] for i in labels],
                                                 np.array(classes)[probabilities.argmax(1)], probabilities, classes),
                        "reliability": reliability(probabilities, labels)}
    return result


def build_policy(root: Path) -> dict:
    """One independent check; reruns return the saved result rather than reselection."""
    rows, audit = read_policy_data(root)
    marker_path = root / "artifacts" / "policy_experiment.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker["status"] == "completed":
            directory = artifact_directory(root, marker["run_id"])
            pointer = read_json(root/"artifacts/policy_latest.json")
            if pointer["run_id"] != marker["run_id"] or sha256_file(directory/"metadata.json") != pointer["metadata_sha256"]:
                raise ValueError("Completed policy metadata/pointer changed.")
            router_metadata = read_json(directory/"metadata.json")
            return {"run_id": marker["run_id"], "gate": router_metadata["gate"],
                    "report": str(root/"reports/runs"/marker["run_id"]/"summary.md"), "reused_frozen_result": True}
        if marker.get("check_exposed"):
            raise ValueError("Independent check was already exposed. Preserve this experiment; do not select another policy against it.")
        parent_run, run_id = marker["parent_run"], marker["run_id"]
    else:
        parent_run = read_json(root/"artifacts/neural_latest.json")["run_id"]
        run_id = "policy_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    models = FrozenModels(root, parent_run)
    cached = read_json(root/"data/embeddings/manifest.json")
    if (cached["encoder_manifest_sha256"] != models.metadata["encoder_manifest_sha256"]
            or audit["manifest_sha256"] != models.metadata["manifest_sha256"]):
        raise ValueError("Frozen parent data/encoder mismatch.")
    arrays = {}
    for split in ("fit", "tune"):
        if cached["input_sha256"][split] != audit["prepared_sha256"][split+".jsonl"]:
            raise ValueError("Frozen parent embedding inputs changed.")
        arrays[split] = validate_feature_cache(root/"data/embeddings"/(split+".npy"), len(rows[split]), cached["sha256"][split])
    # Legacy B2 artifacts predate linear hashes: reproduce the recorded tune result,
    # then seal bytes before any calibration/validation outcome is computed.
    tune_probabilities = softmax(models.logits("B2_MPNet_linear", arrays["tune"]), axis=1)
    original = read_json(root/"reports/runs"/parent_run/"metrics.json")["B2_MPNet_linear"]
    reproduced = classification_metrics([r["intent"] for r in rows["tune"]],
                                       np.array(models.classes)[tune_probabilities.argmax(1)], tune_probabilities, models.classes)
    if any(abs(reproduced[k]-original[k]) > 1e-6 for k in ("accuracy", "macro_f1", "log_loss")):
        raise ValueError("Frozen linear model did not reproduce its recorded tune result.")
    selection, check = split_validation(rows["validation"])
    binding = {"protocol": PROTOCOL, "parent_run": parent_run, "model_hashes": models.hashes,
               "parent_metadata_sha256": sha256_file(models.directory/"metadata.json"),
               "embedding_manifest_sha256": sha256_file(root/"data/embeddings/manifest.json"),
               "input_hashes": {s: audit["prepared_sha256"][s+".jsonl"] for s in rows},
               "source_hashes": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
               "uv_lock_sha256": sha256_file(root/"uv.lock")}
    if marker_path.exists() and read_json(marker_path)["binding_sha256"] != digest_json(binding):
        raise ValueError("Protocol, inputs or code changed during the frozen experiment.")
    directory, reports = artifact_directory(root, run_id), root/"reports/runs"/run_id
    directory.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    marker = {"run_id": run_id, "parent_run": parent_run, "status": "started", "check_exposed": False,
              "binding_sha256": digest_json(binding)}
    write_json(marker_path, marker)
    write_json(reports/"protocol.json", binding)
    write_json(reports/"validation_split.json", {"selection": [rows["validation"][i]["example_id"] for i in selection],
                                                "check": [rows["validation"][i]["example_id"] for i in check]})
    print("Frozen protocol saved. Encoding calibration/validation; final test remains unopened.", flush=True)
    encoded, feature_metadata = policy_embeddings(root, rows, audit)
    prototypes = make_prototypes(arrays["fit"], rows["fit"], models.classes)
    np.save(directory/"prototypes.npy", prototypes, allow_pickle=False)
    label_index = {c: i for i, c in enumerate(models.classes)}
    y_cal = np.array([label_index[r["intent"]] for r in rows["calibrate"]])
    y_val = np.array([label_index.get(r["intent"], -1) for r in rows["validation"]])
    temperatures, logits, probabilities, supports, selection_inputs = {}, {}, {}, {}, {}
    calibration_results = {}
    for name in PROTOCOL["candidates"]:
        cal_logits = models.logits(name, encoded["calibrate"])
        temperatures[name] = fit_temperature(cal_logits, y_cal)
        calibration_results[name] = {"temperature_fit": temperatures[name],
                                     "calibration_fit_metrics": calibration_summary(cal_logits, y_cal, models.classes, temperatures[name]["temperature"])}
        # Encode/forward may run for all rows; only selection outcomes guide choices.
        logits[name] = models.logits(name, encoded["validation"])
        probabilities[name] = softmax(logits[name]/temperatures[name]["temperature"], axis=1)
        supports[name] = similarities(encoded["validation"], probabilities[name], prototypes)
        selection_inputs[name] = (probabilities[name][selection], supports[name][selection])
    selected, trials = choose_policy(selection_inputs, y_val[selection])
    write_json(reports/"selection_trials.json", trials)
    write_json(directory/"selection_freeze.json", {"selected": selected, "temperatures": temperatures,
                                                  "binding_sha256": digest_json(binding)})
    marker.update(status="selection_frozen", check_exposed=True)
    write_json(marker_path, marker)  # No reselection after this point, even after a crash.
    name = selected["model"]
    checked = policy_metrics(probabilities[name][check], supports[name][check], y_val[check],
                             selected["confidence_threshold"], selected["cosine_threshold"])
    gate = check_gate(checked)
    gate["passed"] = gate["passed"] and selected["selection_passed"]
    confidence_interval = bootstrap_cost(probabilities[name][check], supports[name][check], y_val[check], selected)
    known_check = check[y_val[check] >= 0]
    for candidate in PROTOCOL["candidates"]:
        calibration_results[candidate]["independent_check_in_scope"] = calibration_summary(
            logits[candidate][known_check], y_val[known_check], models.classes, temperatures[candidate]["temperature"])
    write_json(reports/"calibration.json", calibration_results)
    write_json(reports/"policy_check.json", {"metrics": checked, "gate": gate, "cost_uncertainty": confidence_interval})
    metadata = {"run_id": run_id, "parent_run": parent_run, "status": "completed", "protocol": PROTOCOL,
                "binding": binding, "model_hashes": models.hashes, "classes": models.classes,
                "encoder_config": CONFIG, "revision": REVISION,
                "encoder_manifest_sha256": models.metadata["encoder_manifest_sha256"],
                "prototypes_sha256": sha256_file(directory/"prototypes.npy"),
                "temperatures": temperatures, "selected": selected, "gate": gate, "check_metrics": checked,
                "selection_count": len(selection), "check_count": len(check),
                "test_evaluated": False, "production_approved": False,
                "legacy_linear_hash_sealed_before_validation": "linear_sha256" not in models.metadata,
                "feature_preparation": feature_metadata}
    write_json(directory/"metadata.json", metadata)
    plot_policy(reports, calibration_results, trials, selected, checked)
    table = "\n".join(f"| {n} | {v['temperature_fit']['temperature']:.3f} | {v['independent_check_in_scope']['before']['log_loss']:.4f} | {v['independent_check_in_scope']['after']['log_loss']:.4f} |"
                      for n, v in calibration_results.items())
    (reports/"summary.md").write_text(
        "# Calibration and independently checked review policy\n\nFinal test untouched. No production approval or external action.\n\n"
        f"Selection: {len(selection)} rows. Independent check: {len(check)} rows. Weights frozen; temperatures fitted on separate calibration rows only.\n\n"
        "| Candidate | Temperature | Check log loss before | After |\n| --- | ---: | ---: | ---: |\n"+table+
        f"\n\nSelected model: **{name}**, confidence ≥ {selected['confidence_threshold']}, predicted-intent cosine similarity ≥ {selected['cosine_threshold']}.\n\n"
        f"Independent-check weighted coverage: {checked['weighted_coverage']:.2%}; weighted selective error: {checked['weighted_selective_error']}; "
        f"OOS accepted: {checked['accepted_oos']}/{checked['oos_count']}; weighted cost {checked['weighted_cost']:.4f} versus all-review 1.0.\n\n"
        f"Conservative development gate passed: **{gate['passed']}**. See policy_check.json for simultaneous bounds and assumptions. "
        "Thresholds were frozen before checking. Do not adjust this policy using check errors. Temperatures do not establish the probability a request is in scope. "
        "Cosine support is a separate heuristic, not an OOS probability. All rates assume the stated 20% OOS scenario.\n\n"
        "![Calibration and routing](calibration_and_routing.png)\n\n"
        "Raw artifacts retain data/model/code hashes, every selection trial, frozen choices and split IDs. A failed gate makes all routing recommendations require review. "
        "A passed development gate remains subject to independent final testing and a representative real-workflow pilot.\n",
        encoding="utf-8")
    write_json(root/"artifacts/policy_latest.json", {"run_id": run_id, "metadata_sha256": sha256_file(directory/"metadata.json")})
    marker.update(status="completed")
    write_json(marker_path, marker)
    return {"run_id": run_id, "selected": selected, "independent_check": checked, "gate": gate,
            "report": str(reports/"summary.md")}


def plot_policy(reports, calibration, trials, selected, checked):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, values) in zip(axes[0], calibration.items()):
        ax.plot([0, 1], [0, 1], color="#999999", linestyle=":")
        for stage in ("before", "after"):
            bins = values["independent_check_in_scope"][stage]["reliability"]["bins"]
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", label=stage)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean class confidence", ylabel="Observed accuracy", title=name+" — independent IS check")
        ax.legend()
    for name in PROTOCOL["candidates"]:
        values = [t for t in trials if t["model"] == name]
        axes[1, 0].scatter([t["weighted_coverage"] for t in values], [t["weighted_cost"] for t in values], s=9, alpha=.4, label=name)
    axes[1, 0].axhline(1, color="gray", linestyle=":", label="All review")
    axes[1, 0].set(xlabel="Weighted coverage", ylabel="Weighted cost", title="Selection trials only — fixed 20% OOS")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(["All review", selected["model"]], [1., checked["weighted_cost"]], color=["#999999", "#177e89"])
    axes[1, 1].set(ylabel="Weighted cost", title="Frozen policy — independent check")
    fig.tight_layout()
    fig.savefig(reports/"calibration_and_routing.png", dpi=160)
    plt.close(fig)


class Router:
    """Load once, reuse for multiple requests; never execute external actions."""
    def __init__(self, root: Path):
        pointer = read_json(root/"artifacts/policy_latest.json")
        directory = artifact_directory(root, pointer["run_id"])
        if sha256_file(directory/"metadata.json") != pointer["metadata_sha256"]:
            raise ValueError("Policy metadata checksum mismatch.")
        self.metadata = read_json(directory/"metadata.json")
        if self.metadata["status"] != "completed" or self.metadata["encoder_config"] != CONFIG or self.metadata["revision"] != REVISION:
            raise ValueError("Incomplete or incompatible routing artifact.")
        self.models = FrozenModels(root, self.metadata["parent_run"], self.metadata["model_hashes"])
        self.tokenizer, self.encoder, _ = load_encoder(root)
        if sha256_file(root/"models/mpnet/manifest.json") != self.metadata["encoder_manifest_sha256"]:
            raise ValueError("Routing encoder checksum mismatch.")
        if sha256_file(directory/"prototypes.npy") != self.metadata["prototypes_sha256"]:
            raise ValueError("Prototype checksum mismatch.")
        self.prototypes = np.load(directory/"prototypes.npy", allow_pickle=False)
        if self.prototypes.shape != (150, 768) or not np.isfinite(self.prototypes).all():
            raise ValueError("Invalid routing prototypes.")

    def route(self, text: str) -> dict:
        text = validate_text(text)
        start = time.perf_counter()
        tokens = len(self.tokenizer(text, truncation=False)["input_ids"])
        result = {"policy_version": self.metadata["run_id"], "route_recommendation": "review_required",
                  "action_executed": False, "production_approved": False,
                  "development_gate_passed": self.metadata["gate"]["passed"], "input_tokens": tokens,
                  "score_type": "temperature_scaled_known_class_probability; not probability of being in scope"}
        if tokens > CONFIG["max_length"]:
            return {**result, "candidates": [], "reason": "Request exceeds the evaluated token limit; shorten or review the full request.",
                    "elapsed_ms": (time.perf_counter()-start)*1000}
        features = encode_batch(self.tokenizer, self.encoder, [text])
        selected = self.metadata["selected"]
        name = selected["model"]
        probabilities = softmax(self.models.logits(name, features)/self.metadata["temperatures"][name]["temperature"], axis=1)
        support = similarities(features, probabilities, self.prototypes)
        accepted = bool(acceptance(probabilities, support, selected["confidence_threshold"], selected["cosine_threshold"])[0])
        top = np.argsort(probabilities[0])[-3:][::-1]
        recommended = accepted and self.metadata["gate"]["passed"]
        return {**result, "model": name, "route_recommendation": "route_to_intent" if recommended else "review_required",
                "candidates": [{"intent": self.models.classes[i], "score": float(probabilities[0, i])} for i in top],
                "support_similarity": float(support[0]),
                "reason": "Meets the frozen development policy; recommendation only." if recommended else
                          ("Development gate failed; review all requests." if not self.metadata["gate"]["passed"] else "Confidence or semantic support is below the frozen threshold."),
                "elapsed_ms": (time.perf_counter()-start)*1000}
