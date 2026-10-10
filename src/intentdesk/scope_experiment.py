"""A predeclared, separately frozen OOS advisory experiment on local CPU."""

from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
from pathlib import Path
import urllib.request

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.data import REVISION, fingerprint
from intentdesk.neural import CONFIG, encode_batch, load_encoder, validate_feature_cache
from intentdesk.policy import (FrozenModels, binomial_upper, make_prototypes,
                               read_policy_data, similarities)
from intentdesk.scope import FEATURE_NAMES, scope_features

RUN = "scope_v1_20261010"
PROTOCOL = {
    "version": 1, "seed": 20261010, "reference_fraction": .8,
    "features": FEATURE_NAMES, "model": "StandardScaler + balanced logistic regression C=1",
    "fresh_oos_selection": 51, "fresh_oos_check": 100,
    "threshold_grid": [i/20 for i in range(1, 21)]+[1.1],
    "likely_oos_threshold": "5th percentile of known selection support scores; strict less-than",
    "supported_selection": "grid thresholds floored at likely-OOS threshold; minimum 80% known retention and maximum 5% OOS acceptance; minimum weighted error/review cost then lower threshold",
    "fallback_upper": 1.1, "oos_scenario_prevalence": .2, "false_oos_acceptance_cost": 10,
    "automatic_actions": False,
    "known_check": "reused calibrate partition; previously used for temperature fitting, NOT a fresh independent joint test",
    "fresh_check": "100 previously unused crowd-labeled OOS rows from pinned CLINC extension; no new human adjudication",
    "old_final_data": "never read or retuned in this experiment",
}
SOURCE_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}/data/data_oos_plus.json"
# SHA is pinned below after acquisition, before any model comparison.
SOURCE_SHA = "bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f"


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def group_order(group):
    return hashlib.sha256(("intentdesk-scope-v1:"+group).encode()).hexdigest()


def split_new_oos(pairs, existing_groups):
    seen, fresh, excluded = set(), [], []
    for index, pair in enumerate(pairs):
        if (not isinstance(pair, list) or len(pair) != 2 or pair[1] != "oos"
                or not isinstance(pair[0], str) or not pair[0].strip()):
            raise ValueError("Invalid additional OOS source row.")
        group = fingerprint(pair[0])
        if group in existing_groups or group in seen:
            excluded.append({"index": index, "group": group, "reason": "existing or duplicate fingerprint"})
        else:
            fresh.append({"example_id": f"oos_plus:{index:05}", "text": pair[0],
                          "duplicate_group": group, "intent": "oos"})
        seen.add(group)
    fresh.sort(key=lambda r: group_order(r["duplicate_group"]))
    cut = PROTOCOL["fresh_oos_selection"]
    if len(fresh) != cut+PROTOCOL["fresh_oos_check"]:
        raise ValueError("Unexpected fresh OOS count; declare a new protocol before training.")
    return fresh[:cut], fresh[cut:], excluded


def prepare_scope(root: Path):
    directory = root/"data/scope_v1"
    directory.mkdir(exist_ok=True)
    protocol_path = directory/"protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != PROTOCOL:
        raise ValueError("Scope protocol changed; preserve the existing experiment.")
    write_json(protocol_path, PROTOCOL)
    source = root/"data/raw/clinc_oos_plus.json"
    if not source.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            payload = response.read(12_000_000)
        if len(payload) >= 12_000_000:
            raise ValueError("Additional source exceeds the declared size budget.")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
    if sha256_file(source) != SOURCE_SHA:
        raise ValueError("Additional OOS source checksum mismatch.")
    audit = read_json(root/"data/prepared/audit.json")
    manifest = root/"data/prepared/manifest.csv"
    if sha256_file(manifest) != audit["manifest_sha256"]:
        raise ValueError("Original manifest changed.")
    with manifest.open(encoding="utf-8", newline="") as stream:
        original = list(csv.DictReader(stream))
    selection, check, excluded = split_new_oos(read_json(source)["oos_train"],
                                               {r["duplicate_group"] for r in original})
    reserved = root/"data/prepared/oos_development.jsonl"
    if sha256_file(reserved) != audit["prepared_sha256"][reserved.name]:
        raise ValueError("Reserved OOS partition changed.")
    training = read_records(reserved)
    known_groups = {r["duplicate_group"] for r in original if r["is_oos"] == "False"}
    groups = [r["duplicate_group"] for r in training]
    if (len(training) != 100 or len(set(groups)) != len(groups) or known_groups & set(groups)
            or any(r["intent"] != "oos" or r["source_split"] != "oos_train" for r in training)):
        raise ValueError("Reserved OOS allocation failed provenance/overlap audit.")
    files = {}
    import json
    for name, rows in (("train_oos", training), ("selection_oos", selection), ("check_oos", check)):
        content = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows)
        path = directory/(name+".jsonl")
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Prepared scope membership changed.")
        path.write_text(content, encoding="utf-8")
        files[path.name] = sha256_file(path)
    result = {"protocol": PROTOCOL, "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA,
              "original_manifest_sha256": sha256_file(manifest), "sha256": files,
              "counts": {"train_oos": len(training), "selection_oos": len(selection), "check_oos": len(check)},
              "excluded": excluded, "cross_partition_fingerprint_overlap": 0,
              "labels": "Original crowd-authored OOS labels; no independent new semantic adjudication",
              "limits": "Fingerprint exclusion cannot prove absence of paraphrase or pretrained-encoder overlap."}
    write_json(directory/"audit.json", result)
    write_json(root/"reports/scope_data_audit.json", result)
    return result


def select_thresholds(known_scores, oos_scores):
    if (not len(known_scores) or not len(oos_scores)
            or not np.isfinite(known_scores).all() or not np.isfinite(oos_scores).all()):
        raise ValueError("Threshold selection needs finite known/OOS scores.")
    lower = float(np.quantile(known_scores, .05, method="lower"))
    trials = []
    for candidate in PROTOCOL["threshold_grid"]:
        threshold = max(lower, candidate)
        retention = float(np.mean(known_scores >= threshold))
        false_accept = float(np.mean(oos_scores >= threshold))
        # Fixed illustrative 20% OOS mixture; uncertainty goes to human review.
        coverage = .8*retention+.2*false_accept
        cost = 10*.2*false_accept+1-coverage
        trials.append({"upper": threshold, "known_retention": retention,
                       "oos_false_acceptance": false_accept, "cost": cost})
    eligible = [t for t in trials if t["known_retention"] >= .8 and t["oos_false_acceptance"] <= .05]
    upper = min(eligible, key=lambda t: (t["cost"], t["upper"]))["upper"] if eligible else 1.1
    return {"lower": lower, "upper": max(lower, upper), "selection_passed": bool(eligible)}, trials


def scope_metrics(known_scores, oos_scores, thresholds):
    lower, upper = thresholds["lower"], thresholds["upper"]
    accepted = int(np.sum(oos_scores >= upper))
    return {"known_count": len(known_scores), "fresh_oos_count": len(oos_scores),
            "known_supported_candidates": int(np.sum(known_scores >= upper)),
            "known_incorrectly_flagged_oos": int(np.sum(known_scores < lower)),
            "fresh_oos_supported_candidates": accepted,
            "fresh_oos_flagged_likely_oos": int(np.sum(oos_scores < lower)),
            "fresh_oos_uncertain": int(np.sum((oos_scores >= lower) & (oos_scores < upper))),
            "oos_acceptance_upper_95": binomial_upper(accepted, len(oos_scores), .05),
            "auroc_diagnostic_reused_known": float(roc_auc_score(
                np.r_[np.ones(len(known_scores)), np.zeros(len(oos_scores))], np.r_[known_scores, oos_scores]))}


def verified_inputs(root):
    audit = read_json(root/"data/scope_v1/audit.json")
    if audit["protocol"] != PROTOCOL or read_json(root/"data/scope_v1/protocol.json") != PROTOCOL:
        raise ValueError("Scope data protocol changed.")
    for name, expected in audit["sha256"].items():
        if sha256_file(root/"data/scope_v1"/name) != expected:
            raise ValueError("Scope partition checksum mismatch.")
    return audit


def original_state(root):
    rows, audit = read_policy_data(root)
    metadata = read_json(root/"artifacts/policy_20261010T043806079244Z/metadata.json")
    models = FrozenModels(root, metadata["parent_run"], metadata["model_hashes"])
    cache = read_json(root/"data/embeddings/manifest.json")
    if any(cache["input_sha256"][s] != audit["prepared_sha256"][s+".jsonl"] for s in ("fit", "tune")):
        raise ValueError("Scope development embedding inputs changed.")
    arrays = {s: validate_feature_cache(root/"data/embeddings"/(s+".npy"), len(rows[s]), cache["sha256"][s])
              for s in ("fit", "tune")}
    cache2 = read_json(root/"data/embeddings/policy/manifest.json")
    if cache2["input_sha256"]["calibrate"] != audit["prepared_sha256"]["calibrate.jsonl"]:
        raise ValueError("Scope known-check embedding inputs changed.")
    arrays["calibrate"] = validate_feature_cache(root/"data/embeddings/policy/calibrate.npy",
                                                len(rows["calibrate"]), cache2["sha256"]["calibrate"])
    return rows, arrays, metadata, models


def encode_rows(root, records):
    tokenizer, encoder, _ = load_encoder(root)
    lengths = tokenizer([r["text"] for r in records], truncation=False)["input_ids"]
    if any(len(ids) > CONFIG["max_length"] for ids in lengths):
        raise ValueError("New scope rows exceed the serving token limit; preserve and report before redefining protocol.")
    batches = []
    for start in range(0, len(records), CONFIG["embedding_batch"]):
        batches.append(encode_batch(tokenizer, encoder, [r["text"] for r in records[start:start+CONFIG["embedding_batch"]]]))
    return np.concatenate(batches)


def frozen_binding(root, metadata):
    paths = ["src/intentdesk/scope.py", "src/intentdesk/scope_experiment.py", "uv.lock",
             "data/scope_v1/audit.json", "data/prepared/audit.json", "models/mpnet/manifest.json",
             f"artifacts/{RUN}/detector.joblib", f"artifacts/{RUN}/reference.npy"]
    paths += [f"artifacts/{metadata['parent_policy']}/metadata.json",
              "data/embeddings/manifest.json", "data/embeddings/policy/manifest.json"]
    for split in ("fit", "tune", "calibrate"):
        paths.append(f"data/prepared/{split}.jsonl")
    paths.extend(f"data/scope_v1/{name}.jsonl" for name in ("train_oos", "selection_oos", "check_oos"))
    return digest({"protocol": PROTOCOL, "metadata": metadata,
                   "files": {p: sha256_file(root/p) for p in paths}})


def train_scope(root):
    verified_inputs(root)
    directory = root/"artifacts"/RUN
    if (directory/"metadata.json").exists():
        metadata = read_json(directory/"metadata.json")
        marker = read_json(root/"artifacts/scope_v1_experiment.json")
        if frozen_binding(root, {**metadata, "status": "trained_frozen"}) != marker["binding_sha256"]:
            raise ValueError("Frozen scope experiment changed; do not retrain against an exposed check.")
        return {"status": "reused_frozen_training", "run_id": RUN}
    if (root/"artifacts/scope_v1_experiment.json").exists():
        raise ValueError("Experiment marker exists without training metadata; preserve for diagnosis.")
    rows, arrays, parent, models = original_state(root)
    buckets = defaultdict(list)
    for i, row in enumerate(rows["fit"]):
        buckets[row["intent"]].append(i)
    reference_ids, train_ids = [], []
    for indices in buckets.values():
        indices.sort(key=lambda i: group_order(rows["fit"][i]["duplicate_group"]))
        cut = int(len(indices)*PROTOCOL["reference_fraction"])
        reference_ids.extend(indices[:cut]); train_ids.extend(indices[cut:])
    if set(reference_ids) & set(train_ids):
        raise ValueError("Self-neighbor leakage in scope training.")
    reference = np.asarray(arrays["fit"][reference_ids])
    prototypes = make_prototypes(arrays["fit"], rows["fit"], models.classes)
    oos_train = read_records(root/"data/scope_v1/train_oos.jsonl")
    oos_select = read_records(root/"data/scope_v1/selection_oos.jsonl")
    additional = encode_rows(root, oos_train+oos_select)
    temperature = parent["temperatures"]["B2_MPNet_linear"]["temperature"]

    def transform(embedding):
        p = softmax(models.logits("B2_MPNet_linear", embedding)/temperature, axis=1)
        return scope_features(embedding, reference, prototypes, p)

    positive = transform(arrays["fit"][train_ids])
    negative = transform(additional[:len(oos_train)])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced",
                                    max_iter=2000, random_state=PROTOCOL["seed"]))
    with threadpool_limits(limits=2):
        classifier.fit(np.vstack((positive, negative)), np.r_[np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)])
        known_scores = classifier.predict_proba(transform(arrays["tune"]))[:, 1]
        oos_scores = classifier.predict_proba(transform(additional[len(oos_train):]))[:, 1]
    thresholds, trials = select_thresholds(known_scores, oos_scores)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, directory/"detector.joblib")
    np.save(directory/"reference.npy", reference, allow_pickle=False)
    metadata = {"status": "trained_frozen", "run_id": RUN, "parent_policy": parent["run_id"],
                "parent_policy_metadata_sha256": sha256_file(root/"artifacts"/parent["run_id"]/"metadata.json"),
                "features": FEATURE_NAMES, "protocol": PROTOCOL, "thresholds": thresholds,
                "reference_rows": len(reference), "known_training_rows": len(positive), "oos_training_rows": len(negative),
                "artifact_sha256": {name: sha256_file(directory/name) for name in ("detector.joblib", "reference.npy")},
                "automatic_actions": False}
    write_json(directory/"metadata.json", metadata)
    write_json(directory/"selection.json", {"thresholds": thresholds, "trials": trials,
                                            "metrics": scope_metrics(known_scores, oos_scores, thresholds)})
    # This marker binds code, data, model and selected thresholds BEFORE check inference.
    write_json(root/"artifacts/scope_v1_experiment.json", {"status": "trained_frozen", "check_exposed": False,
               "binding_sha256": frozen_binding(root, metadata), "frozen_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_id": RUN, "thresholds": thresholds, "known_training_rows": len(positive), "oos_training_rows": len(negative)}


def evaluate_scope(root):
    verified_inputs(root)
    directory = root/"artifacts"/RUN
    metadata = read_json(directory/"metadata.json")
    marker_path = root/"artifacts/scope_v1_experiment.json"
    marker = read_json(marker_path)
    # metadata status is updated only after evaluation, so bind the trained form.
    bound_metadata = {**metadata, "status": "trained_frozen"}
    if frozen_binding(root, bound_metadata) != marker["binding_sha256"]:
        raise ValueError("Frozen scope binding changed; preserve the exposed check and old result.")
    report_path = root/"reports/scope_evaluation.json"
    if marker["status"] == "completed":
        if sha256_file(report_path) != marker["report_sha256"]:
            raise ValueError("Completed scope report checksum mismatch.")
        return {**read_json(report_path), "reused_frozen_result": True}
    marker.update(check_exposed=True, status="evaluating")
    write_json(marker_path, marker)
    rows, arrays, parent, models = original_state(root)
    reference = np.load(directory/"reference.npy", allow_pickle=False)
    classifier = joblib.load(directory/"detector.joblib")
    check = read_records(root/"data/scope_v1/check_oos.jsonl")
    oos = encode_rows(root, check)
    prototypes = make_prototypes(arrays["fit"], rows["fit"], models.classes)
    temperature = parent["temperatures"]["B2_MPNet_linear"]["temperature"]
    embeddings = np.vstack((arrays["calibrate"], oos))
    p = softmax(models.logits("B2_MPNet_linear", embeddings)/temperature, axis=1)
    features = scope_features(embeddings, reference, prototypes, p)
    with threadpool_limits(limits=2):
        scores = classifier.predict_proba(features)[:, 1]
    n = len(arrays["calibrate"])
    metrics = scope_metrics(scores[:n], scores[n:], metadata["thresholds"])
    original = parent["selected"]
    old_accept = (p.max(1) >= original["confidence_threshold"]) & (similarities(embeddings, p, prototypes) >= original["cosine_threshold"])
    baseline = {"known_supported_candidates": int(old_accept[:n].sum()), "fresh_oos_supported_candidates": int(old_accept[n:].sum()),
                "scope": "Original frozen thresholds applied descriptively to the same check; not reselected"}
    report = {"run_id": RUN, "protocol": PROTOCOL, "binding_sha256": marker["binding_sha256"],
              "thresholds": metadata["thresholds"], "check": metrics, "old_thresholds_comparison": baseline,
              "selection": read_json(directory/"selection.json"), "automatic_actions": False,
              "new_independent_joint_test": False, "old_final_reused": False,
              "limits": ["Only 100 fresh OOS examples; known check data reused from temperature calibration.",
                         "Public crowd-authored labels, not a new independently adjudicated production suite.",
                         "Support score is not calibrated and must not be displayed as probability of being supported.",
                         "No automatic routing authorization; semantic/pretraining overlap cannot be ruled out."]}
    write_json(report_path, report)
    metadata["status"] = "completed"
    write_json(directory/"metadata.json", metadata)
    marker.update(status="completed", report_sha256=sha256_file(report_path))
    write_json(marker_path, marker)
    write_json(root/"artifacts/scope_latest.json", {"run_id": RUN, "metadata_sha256": sha256_file(directory/"metadata.json")})
    return report
