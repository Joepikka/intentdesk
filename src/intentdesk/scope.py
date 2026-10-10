"""Advisory scope detection on frozen features; never authorizes an action."""

from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, sha256_file
from intentdesk.neural import CONFIG, encode_batch
from intentdesk.policy import Router, artifact_directory
from intentdesk.predict import validate_text

FEATURE_NAMES = ["nearest_cosine", "top5_mean_cosine", "maximum_prototype_cosine",
                 "known_class_max", "known_class_margin", "normalized_entropy"]


def scope_features(embeddings, reference, prototypes, probabilities):
    """Queries must be disjoint from reference; bound RAM with 128-row blocks."""
    x, ref, proto, p = (np.asarray(v, dtype=np.float32) for v in
                        (embeddings, reference, prototypes, probabilities))
    if (x.ndim != 2 or ref.ndim != 2 or proto.ndim != 2 or p.ndim != 2
            or not len(x) or x.shape[1] != 768 or ref.shape[1] != 768 or proto.shape != (150, 768)
            or p.shape != (len(x), 150) or len(ref) < 5
            or any(not np.isfinite(v).all() for v in (x, ref, proto, p))
            or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5)
            or any(not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4) for v in (x, ref, proto))):
        raise ValueError("Invalid scope features or reference contract.")
    result = np.empty((len(x), len(FEATURE_NAMES)), dtype=np.float64)
    with threadpool_limits(limits=2):
        for start in range(0, len(x), 128):
            stop = min(start+128, len(x))
            neighbors = x[start:stop] @ ref.T
            best = np.partition(neighbors, -5, axis=1)[:, -5:]
            ordered = np.sort(p[start:stop], axis=1)
            result[start:stop] = np.column_stack((
                best.max(1), best.mean(1), (x[start:stop] @ proto.T).max(1),
                ordered[:, -1], ordered[:, -1]-ordered[:, -2],
                -(p[start:stop]*np.log(np.maximum(p[start:stop], 1e-12))).sum(1)/np.log(150)))
    return result


def assessment(score, lower, upper):
    if not np.isfinite(score) or not 0 <= score <= 1 or not 0 <= lower <= upper <= 1.1:
        raise ValueError("Invalid advisory scope score/thresholds.")
    if score < lower:
        return "likely_out_of_scope"
    return "supported_candidate" if score >= upper else "uncertain"


class ScopeDetector:
    def __init__(self, root: Path, parent_policy: str):
        pointer = read_json(root/"artifacts/scope_latest.json")
        directory = artifact_directory(root, pointer["run_id"])
        if sha256_file(directory/"metadata.json") != pointer["metadata_sha256"]:
            raise ValueError("Scope metadata checksum mismatch.")
        self.metadata = read_json(directory/"metadata.json")
        if (self.metadata["status"] != "completed" or self.metadata["parent_policy"] != parent_policy
                or self.metadata["features"] != FEATURE_NAMES or self.metadata["automatic_actions"]):
            raise ValueError("Scope model contract mismatch.")
        if sha256_file(root/"artifacts"/parent_policy/"metadata.json") != self.metadata["parent_policy_metadata_sha256"]:
            raise ValueError("Scope parent policy changed.")
        for name, digest in self.metadata["artifact_sha256"].items():
            if name not in ("detector.joblib", "reference.npy") or sha256_file(directory/name) != digest:
                raise ValueError("Scope artifact checksum mismatch.")
        if set(self.metadata["artifact_sha256"]) != {"detector.joblib", "reference.npy"}:
            raise ValueError("Incomplete scope artifacts.")
        self.reference = np.load(directory/"reference.npy", allow_pickle=False)
        if (self.reference.shape != (self.metadata["reference_rows"], 768)
                or not np.isfinite(self.reference).all()
                or not np.allclose(np.linalg.norm(self.reference, axis=1), 1, atol=1e-4)):
            raise ValueError("Invalid scope reference cache.")
        self.model = joblib.load(directory/"detector.joblib")
        if self.model.n_features_in_ != 6 or list(self.model.classes_) != [0, 1]:
            raise ValueError("Invalid scope classifier features/classes.")

    def predict(self, embedding, prototypes, probabilities):
        values = scope_features(embedding, self.reference, prototypes, probabilities)
        with threadpool_limits(limits=2):
            score = float(self.model.predict_proba(values)[0, 1])
        thresholds = self.metadata["thresholds"]
        return {"status": assessment(score, thresholds["lower"], thresholds["upper"]),
                "support_score": score, "score_type": "uncalibrated advisory score; not probability of support",
                "version": self.metadata["run_id"], "automatic_actions": False}


class ScopeRouter(Router):
    """Keep the frozen classifier/policy intact; add a separately versioned advisory."""
    def __init__(self, root: Path):
        super().__init__(root)
        self.scope = None
        self.scope_error = None
        if (root/"artifacts/scope_latest.json").exists():
            try:
                self.scope = ScopeDetector(root, self.metadata["run_id"])
            except (ValueError, OSError, KeyError, TypeError):
                # Invalid optional advisory artifacts must never weaken review-all.
                self.scope_error = "Scope assessment unavailable; review this request manually."

    def route(self, text: str) -> dict:
        text = validate_text(text)
        if self.scope is None:
            result = super().route(text)
            result.update(route_recommendation="review_required", action_executed=False)
            result["scope_assessment"] = {"status": "unavailable", "automatic_actions": False}
            result["reason"] = self.scope_error or "Human review is required. Scope assessment is not installed."
            return result
        started = time.perf_counter()
        tokens = len(self.tokenizer(text, truncation=False)["input_ids"])
        base = {"policy_version": self.metadata["run_id"], "route_recommendation": "review_required",
                "action_executed": False, "production_approved": False,
                "development_gate_passed": self.metadata["gate"]["passed"], "input_tokens": tokens,
                "score_type": "temperature_scaled_known_class_probability; not probability of being in scope"}
        if tokens > CONFIG["max_length"]:
            return {**base, "candidates": [], "scope_assessment": {"status": "unavailable"},
                    "reason": "Request exceeds the evaluated token limit; shorten or review the full request.",
                    "elapsed_ms": (time.perf_counter()-started)*1000}
        embedding = encode_batch(self.tokenizer, self.encoder, [text])
        name = "B2_MPNet_linear"
        p = softmax(self.models.logits(name, embedding)/self.metadata["temperatures"][name]["temperature"], axis=1)
        scope = self.scope.predict(embedding, self.prototypes, p)
        nearest = [{"intent": self.models.classes[i], "score": float(p[0, i])}
                   for i in np.argsort(p[0])[-3:][::-1]]
        reasons = {
            "likely_out_of_scope": "Likely outside the supported categories. Review as unsupported (oos), or correct the assessment.",
            "uncertain": "Scope is uncertain. The closest known categories may not apply; human review is required.",
            "supported_candidate": "A supported category looks plausible. Confirm the intent; human review is still required.",
        }
        return {**base, "model": name, "scope_assessment": scope,
                "candidates": [] if scope["status"] == "likely_out_of_scope" else nearest,
                "nearest_known_candidates": nearest, "reason": reasons[scope["status"]],
                "elapsed_ms": (time.perf_counter()-started)*1000}
