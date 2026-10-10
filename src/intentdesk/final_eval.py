"""Once-frozen CLINC evaluation. Results cannot select a replacement model/policy."""

from datetime import datetime, timezone
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import softmax
import torch

from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.neural import CONFIG, encode_batch, make_head
from intentdesk.policy import Router, artifact_directory, digest_json, policy_metrics, reliability, similarities
from intentdesk.train import classification_metrics

PROTOCOL = {"version": 1, "primary": "B2_MPNet_linear_temperature_scaled",
            "policy_run": "policy_20261010T043806079244Z",
            "neural_run": "neural_20261009T204150600532Z", "baseline_run": "20261009T200903346981Z",
            "compare": ["B1_lexical", "B2_MPNet_linear", "N1_small", "N2_large"],
            "selection_after_test": "forbidden; comparisons are descriptive only",
            "active_behavior": "review all; failed development gate cannot be overridden by this test",
            "known_test_count": 4500, "oos_test_count": 1000, "bootstrap_seed": 20261010,
            "accuracy_interval": "2000 bootstrap resamples of formatting duplicate groups; conditional on this benchmark",
            "final_test_evaluated": True}


def seal_experiment(root: Path, binding: dict) -> dict:
    path = root/"artifacts/final_experiment.json"
    identity = digest_json(binding)
    if path.exists():
        marker = read_json(path)
        if marker["binding_sha256"] != identity or marker["binding"] != binding:
            raise ValueError("Final test already bound to another configuration. Do not tune or replace it using exposed results.")
        return marker
    marker = {"status": "frozen_before_exposure", "test_exposed": False, "binding": binding,
              "binding_sha256": identity, "frozen_utc": datetime.now(timezone.utc).isoformat()}
    write_json(path, marker)
    return marker


def grouped_accuracy_interval(correct: np.ndarray, groups: list[str]) -> list[float]:
    totals = {}
    for value, group in zip(correct, groups, strict=True):
        successes, count = totals.get(group, (0, 0))
        totals[group] = (successes+int(value), count+1)
    counts = np.array(list(totals.values()))
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    samples = counts[rng.integers(0, len(counts), size=(2000, len(counts)))]
    values = samples[:, :, 0].sum(1)/samples[:, :, 1].sum(1)
    return np.quantile(values, [.025,.975]).tolist()


def measure_known(truth, probabilities, classes):
    predicted = np.asarray(classes)[np.asarray(probabilities).argmax(1)]
    return classification_metrics(truth, predicted, probabilities, classes)


def evaluate_final(root: Path) -> dict:
    if read_json(root/"artifacts/policy_latest.json")["run_id"] != PROTOCOL["policy_run"]:
        raise ValueError("Final protocol requires the previously frozen policy.")
    router = Router(root)
    if router.metadata["parent_run"] != PROTOCOL["neural_run"] or router.metadata["gate"]["passed"]:
        raise ValueError("Unexpected parent or active policy; review the experiment boundary.")
    baseline = artifact_directory(root, PROTOCOL["baseline_run"])/"baseline.joblib"
    parent = artifact_directory(root, PROTOCOL["neural_run"])
    audit = read_json(root/"data/prepared/audit.json")
    test_path = root/"data/prepared/locked_test.jsonl"
    if sha256_file(test_path) != audit["prepared_sha256"][test_path.name]:
        raise ValueError("Final test checksum mismatch.")
    hashes = {"policy_metadata": sha256_file(artifact_directory(root, PROTOCOL["policy_run"])/"metadata.json"),
              "baseline": sha256_file(baseline), "large_head": sha256_file(parent/"N2_large.pt"),
              "test": sha256_file(test_path), "encoder_manifest": sha256_file(root/"models/mpnet/manifest.json"),
              **router.metadata["model_hashes"]}
    if hashes["large_head"] != router.models.metadata["checkpoints"]["N2_large"]["sha256"]:
        raise ValueError("Large model changed before final evaluation.")
    binding = {"protocol": PROTOCOL, "hashes": hashes,
               "source_sha256": {name: sha256_file(root/"src/intentdesk"/name)
                                  for name in ("data.py", "train.py", "predict.py", "neural.py", "policy.py", "final_eval.py")},
               "uv_lock_sha256": sha256_file(root/"uv.lock")}
    marker = seal_experiment(root, binding)
    report_path = root/"reports/final_evaluation.json"
    if report_path.exists():
        existing = read_json(report_path)
        if existing["binding_sha256"] != marker["binding_sha256"]:
            raise ValueError("Final report binding changed.")
        return {**existing, "reused_frozen_result": True}
    # Only an identical frozen configuration may resume after an interrupted evaluation.
    marker.update(status="evaluating_frozen_configuration", test_exposed=True)
    write_json(root/"artifacts/final_experiment.json", marker)
    rows = read_records(test_path)
    if (len(rows) != 5500 or sum(r["is_oos"] for r in rows) != PROTOCOL["oos_test_count"]
            or any(r["working_split"] != "locked_test" or r["source_split"] not in ("test", "oos_test")
                   or r["is_oos"] != (r["intent"] == "oos") for r in rows)):
        raise ValueError("Final partition schema/count mismatch.")
    lengths = router.tokenizer([r["text"] for r in rows], truncation=False)["input_ids"]
    if any(len(ids) > CONFIG["max_length"] for ids in lengths):
        raise ValueError("Final data exceeds the serving token limit; preserve frozen protocol and report this limitation.")
    start = time.perf_counter()
    features = np.empty((len(rows), 768), dtype=np.float32)
    for offset in range(0, len(rows), CONFIG["embedding_batch"]):
        end = min(offset+CONFIG["embedding_batch"], len(rows))
        features[offset:end] = encode_batch(router.tokenizer, router.encoder, [r["text"] for r in rows[offset:end]])
        if end % 512 == 0 or end == len(rows):
            print(f"Frozen final evaluation embeddings: {end:,}/{len(rows):,}", flush=True)
    encoding_seconds = time.perf_counter()-start
    classes = router.models.classes
    lookup = {c: i for i, c in enumerate(classes)}
    labels = np.array([-1 if r["is_oos"] else lookup[r["intent"]] for r in rows])
    known = labels >= 0
    truth = np.array([r["intent"] for r in rows])[known]
    lexical = joblib.load(baseline)
    if list(lexical.classes_) != classes:
        raise ValueError("Baseline classes changed.")
    large = make_head(router.models.metadata["checkpoints"]["N2_large"]["widths"]).eval()
    large.load_state_dict(torch.load(parent/"N2_large.pt", map_location="cpu", weights_only=True))
    with torch.inference_mode():
        large_logits = large(torch.from_numpy(features[known])).numpy()
    scores = {"B1_lexical": lexical.predict_proba([r["text"] for r in rows if not r["is_oos"]]),
              "B2_MPNet_linear": softmax(router.models.logits("B2_MPNet_linear", features[known]), axis=1),
              "N1_small": softmax(router.models.logits("N1_small", features[known]), axis=1),
              "N2_large": softmax(large_logits, axis=1)}
    comparison = {name: measure_known(truth, values, classes) for name, values in scores.items()}
    all_logits = router.models.logits("B2_MPNet_linear", features)
    calibrated = softmax(all_logits/router.metadata["temperatures"]["B2_MPNet_linear"]["temperature"], axis=1)
    primary = measure_known(truth, calibrated[known], classes)
    primary["reliability"] = reliability(calibrated[known], labels[known])
    correct = calibrated[known].argmax(1) == labels[known]
    primary["group_bootstrap_accuracy_95_percent"] = grouped_accuracy_interval(correct, [r["duplicate_group"] for r in rows if not r["is_oos"]])
    selected = router.metadata["selected"]
    proposed = policy_metrics(calibrated, similarities(features, calibrated, router.prototypes), labels,
                              selected["confidence_threshold"], selected["cosine_threshold"])
    result = {"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "binding_sha256": marker["binding_sha256"], "protocol": PROTOCOL,
              "counts": {"known": int(known.sum()), "oos": int((~known).sum()), "over_token_limit": 0},
              "primary_calibrated_model": primary, "secondary_fixed_model_comparison": comparison,
              "proposed_thresholds_offline_only": proposed,
              "active_policy": {"review_required": len(rows), "automatic_coverage": 0., "actions_executed": 0,
                                "development_gate_passed": False, "test_used_to_enable_routing": False},
              "encoding_seconds": encoding_seconds,
              "limitations": ["Crowdsourced English benchmark, not real support traffic.",
                              "Final results must not select hyperparameters, temperatures, thresholds or replacement models.",
                              "Grouped bootstrap reflects this benchmark, not production distribution shifts.",
                              "Known-class accuracy does not validate unsupported-request detection."]}
    write_json(report_path, result)
    marker.update(status="completed", report_sha256=sha256_file(report_path))
    write_json(root/"artifacts/final_experiment.json", marker)
    return result
