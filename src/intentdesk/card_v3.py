"""Hybrid card candidate: frozen semantic embeddings plus typo/negation features.

V2 remains a recorded lexical comparison. Its consumed check is development
evidence for this candidate, never a fresh holdout again.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import warnings

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from intentdesk.card_v2 import make_model, training_rows
from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.data import fingerprint
from intentdesk.neural import CONFIG as ENCODER_CONFIG, REVISION, encode_batch, load_encoder, validate_feature_cache
from intentdesk.pilot import CATEGORIES
from intentdesk.predict import validate_text
from intentdesk.robustness import load_cases


CHECK_CASES = "data/robustness/card_support_check_v3.json"
CONFIG = {"version": "card_support_v3", "semantic_weight": 3.0, "lexical_weight": .5,
          "C": 4.0, "max_iter": 1000, "threads": 2, "seed": 20261012,
          "minimum_supported_score": .40, "minimum_supported_margin": .10,
          "selection": "Fixed hybrid candidate after V2 check exposure; V3 check reserved before training",
          "score_type": "uncalibrated eight-class model score; not verified correctness probability"}
SOURCES = ("card_v3.py", "card_v2.py", "card_v2_data.py", "neural.py")


def combined(lexical, embeddings):
    return hstack([lexical*CONFIG["lexical_weight"], csr_matrix(embeddings*CONFIG["semantic_weight"])], format="csr")


def train_card(root: Path):
    rows, fit_hash = training_rows(root)
    protected = {fingerprint(case["text"]) for case in load_cases(root/CHECK_CASES)["cases"]}
    if any(fingerprint(row["text"]) in protected for row in rows):
        raise ValueError("Training overlaps the reserved V3 check.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root/"artifacts"/("card_v3_"+stamp); directory.mkdir(parents=True, exist_ok=False)
    binding = {"config": CONFIG, "fit_sha256": fit_hash,
               "source_sha256": {s: sha256_file(root/"src/intentdesk"/s) for s in SOURCES},
               "uv_lock_sha256": sha256_file(root/"uv.lock"), "encoder_revision": REVISION,
               "reserved_check_sha256": sha256_file(root/CHECK_CASES),
               "encoder_manifest_sha256": sha256_file(root/"models/mpnet/manifest.json")}
    write_json(directory/"training_protocol.json", binding)
    manifest = read_json(root/"data/embeddings/manifest.json")
    if manifest["input_sha256"]["fit"] != fit_hash or manifest["revision"] != REVISION:
        raise ValueError("Original fit embedding identity changed.")
    official = read_records(root/"data/prepared/fit.jsonl")
    original = validate_feature_cache(root/"data/embeddings/fit.npy", len(official), manifest["sha256"]["fit"])
    lookup = {row["example_id"]: i for i, row in enumerate(official)}
    features = np.empty((len(rows), 768), dtype=np.float32)
    extra = []
    for i, row in enumerate(rows):
        if row["origin"] == "clinc_fit":
            features[i] = original[lookup[row["family"]]]
        else:
            extra.append(i)
    texts = [rows[i]["text"] for i in extra]
    cache_identity = hashlib.sha256(json.dumps({"texts": texts, "encoder": binding["encoder_manifest_sha256"]}, sort_keys=True).encode()).hexdigest()
    cache = root/"local/card_v3_cache"/cache_identity
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer, encoder, _ = load_encoder(root)
    started = time.perf_counter()
    for start in range(0, len(extra), 8):
        stop = min(start+8, len(extra))
        path = cache/f"batch_{start}.npy"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            digest = read_json(receipt)["sha256"]
            batch = validate_feature_cache(path, stop-start, digest)
        else:
            if any(len(tokenizer(text, truncation=False)["input_ids"]) > ENCODER_CONFIG["max_length"] for text in texts[start:stop]):
                raise ValueError("Synthetic training text exceeds the encoder token budget.")
            batch = encode_batch(tokenizer, encoder, texts[start:stop])
            np.save(path, batch, allow_pickle=False)
            write_json(receipt, {"sha256": sha256_file(path)})
        features[extra[start:stop]] = batch
        if start % 256 == 0:
            print(f"Hybrid training features: {stop}/{len(extra)} extra texts; original fit embeddings reused.", flush=True)
    lexical = make_model().named_steps["features"]
    labels = [row["label"] for row in rows]
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        sparse = lexical.fit_transform([row["text"] for row in rows])
        classifier = LogisticRegression(C=4., class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=20261012)
        classifier.fit(combined(sparse, features), labels)
    joblib.dump({"lexical": lexical, "classifier": classifier}, directory/"model.joblib")
    metadata = {"run_id": directory.name, "status": "completed", "binding": binding,
                "training_rows": len(rows), "class_counts": dict(Counter(labels)),
                "origin_counts": dict(Counter(row["origin"] for row in rows)),
                "new_embeddings": len(extra), "features": classifier.n_features_in_,
                "encoder_frozen": True, "encoder_trainable_parameters": 0,
                "training_seconds": time.perf_counter()-started,
                "model_sha256": sha256_file(directory/"model.joblib"),
                "automatic_actions": False, "human_reviewed_synthetic_labels": 0,
                "check_exposed": False, "original_benchmarks_reopened": False}
    write_json(directory/"metadata.json", metadata)
    write_json(root/"artifacts/card_v3_latest.json", {"run_id": directory.name,
                "metadata_sha256": sha256_file(directory/"metadata.json")})
    return metadata


class CardRouter:
    def __init__(self, root: Path, shared_encoder=None):
        pointer = read_json(root/"artifacts/card_v3_latest.json")
        if not re.fullmatch(r"card_v3_[0-9TZ]+", pointer["run_id"]):
            raise ValueError("Invalid hybrid card artifact path.")
        directory = root/"artifacts"/pointer["run_id"]
        if sha256_file(directory/"metadata.json") != pointer["metadata_sha256"]:
            raise ValueError("Hybrid metadata checksum changed.")
        self.metadata = read_json(directory/"metadata.json")
        binding = self.metadata["binding"]
        if (self.metadata["status"] != "completed" or self.metadata["automatic_actions"]
                or binding["config"] != CONFIG or binding["uv_lock_sha256"] != sha256_file(root/"uv.lock")
                or binding["encoder_manifest_sha256"] != sha256_file(root/"models/mpnet/manifest.json")
                or any(sha256_file(root/"src/intentdesk"/s) != digest for s, digest in binding["source_sha256"].items())
                or sha256_file(directory/"model.joblib") != self.metadata["model_sha256"]):
            raise ValueError("Hybrid card artifact/source identity changed.")
        model = joblib.load(directory/"model.joblib")
        self.lexical, self.classifier = model["lexical"], model["classifier"]
        if set(self.classifier.classes_) != set(CATEGORIES):
            raise ValueError("Hybrid card classes changed.")
        if shared_encoder is None:
            self.tokenizer, self.encoder, _ = load_encoder(root)
        else:
            self.tokenizer, self.encoder = shared_encoder

    def route(self, text):
        text = validate_text(text)
        started = time.perf_counter()
        tokens = len(self.tokenizer(text, truncation=False)["input_ids"])
        candidates, label = [], "uncertain"
        if tokens <= ENCODER_CONFIG["max_length"]:
            embedding = encode_batch(self.tokenizer, self.encoder, [text])
            with threadpool_limits(limits=2):
                scores = self.classifier.predict_proba(combined(self.lexical.transform([text]), embedding))[0]
            if not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1):
                raise ValueError("Invalid hybrid card scores.")
            order = np.argsort(scores)[::-1]
            candidates = [{"intent": str(self.classifier.classes_[i]), "score": float(scores[i])} for i in order[:3]]
            label = candidates[0]["intent"]
            if label not in ("oos", "uncertain") and (scores[order[0]] < CONFIG["minimum_supported_score"]
                    or scores[order[0]]-scores[order[1]] < CONFIG["minimum_supported_margin"]):
                label = "uncertain"
        reason = ("Please describe one issue clearly, or choose which issue to handle first." if label == "uncertain"
                  else "This looks outside the six card-support categories; confirm or correct it." if label == "oos"
                  else "A card category looks plausible. A person must confirm or correct it.")
        if tokens > ENCODER_CONFIG["max_length"]:
            reason = "The full request exceeds the encoder token limit. Shorten it or review it manually; no truncated prefix was classified."
        return {"label": label, "name": CATEGORIES[label]["name"], "reason": reason,
                "candidates": candidates, "model_version": self.metadata["run_id"], "input_tokens": tokens,
                "score_type": CONFIG["score_type"], "route_recommendation": "review_required",
                "action_executed": False, "production_approved": False,
                "synthetic_training_labels_human_reviewed": False,
                "elapsed_ms": (time.perf_counter()-started)*1000}
