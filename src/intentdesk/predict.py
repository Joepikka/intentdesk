"""Return research suggestions; routing stays disabled until evaluated."""

from pathlib import Path
import time
from uuid import uuid4

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, sha256_file


def validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty request.")
    if len(text) > 4000:
        raise ValueError("Keep requests to 4,000 characters or fewer.")
    return text.strip()


def load_baseline(root: Path):
    artifacts = (root / "artifacts").resolve()
    pointer = artifacts / "latest.json"
    if not pointer.exists():
        raise ValueError("No trained model. Run prepare, then train, from the project directory.")
    run = (artifacts / read_json(pointer)["run_id"]).resolve()
    if not run.is_relative_to(artifacts):
        raise ValueError("Invalid local model path.")
    metadata = read_json(run / "metadata.json")
    model_path = run / "baseline.joblib"
    if sha256_file(model_path) != metadata["run"]["model_sha256"]:
        raise ValueError("Model checksum mismatch; do not use this artifact.")
    return joblib.load(model_path), metadata


def predict(root: Path, text: str) -> dict:
    text = validate_text(text)
    model, metadata = load_baseline(root)
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        probabilities = model.predict_proba([text])[0]
    top = np.argsort(probabilities)[-3:][::-1]
    return {
        "request_id": str(uuid4()), "model_version": metadata["run"]["run_id"],
        "score_type": "uncalibrated_multiclass_probability", "calibrated": False,
        "candidates": [
            {"intent": str(model.classes_[i]), "domain": metadata["intent_domains"][model.classes_[i]],
             "score": float(probabilities[i])}
            for i in top
        ],
        "route_decision": "review_required",
        "reason": "Research baseline: calibration and out-of-scope routing have not been evaluated.",
        "elapsed_ms": (time.perf_counter() - start) * 1000,
    }

