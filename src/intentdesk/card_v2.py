"""Separate, lightweight eight-category card workflow; original models stay frozen."""

from collections import Counter
from datetime import datetime, timezone
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from intentdesk.card_v2_data import SEED, noisy, synthetic_rows
from intentdesk.common import read_json, read_records, sha256_file, write_json
from intentdesk.data import fingerprint
from intentdesk.pilot import CATEGORIES
from intentdesk.predict import validate_text
from intentdesk.robustness import DEFAULT_CASES, load_cases


CHECK_CASES = "data/robustness/card_support_check_v2.json"
CONFIG = {"version": "card_support_v2", "seed": SEED, "word_ngrams": [1, 3],
          "char_ngrams": [3, 5], "max_features_per_branch": 25000, "min_df": 2,
          "word_weight": 1.4, "C": 4.0, "max_iter": 800, "threads": 2,
          "minimum_supported_score": .40, "minimum_supported_margin": .10,
          "selection": "Fixed before training/check exposure; no check-set tuning",
          "score_type": "uncalibrated eight-class model score; not verified correctness probability"}
SOURCES = ("card_v2.py", "card_v2_data.py")


def negation_tokens(text):
    """Bounded heuristic scope for lexical features; not a semantic parser.

    Character features still see the original text. This never rewrites the
    displayed request, removes a clause or makes a category decision itself.
    """
    text = text.casefold().replace("’", "'")
    text = re.sub(r"\b(can't|cannot)\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    tokens, remaining = [], 0
    for word in re.findall(r"[a-z0-9]+|[.,;:!?]", text):
        if word in {"but", "however", "yet"} or word in ".,;:!?":
            remaining = 0
        if word in {"not", "never", "without", "no"}:
            remaining = 5
            tokens.append(word)
        elif re.fullmatch(r"[a-z0-9]+", word):
            tokens.append("neg_"+word if remaining else word)
            remaining = max(0, remaining-1)
    return " ".join(tokens)


def make_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=negation_tokens, ngram_range=(1, 3),
                                max_features=25000, min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                max_features=25000, min_df=2, sublinear_tf=True)),
    ], transformer_weights={"word": CONFIG["word_weight"], "char": 1.0})
    return Pipeline([("features", features), ("classifier", LogisticRegression(
        C=CONFIG["C"], class_weight="balanced", max_iter=CONFIG["max_iter"],
        random_state=SEED, solver="lbfgs"))])


def training_rows(root):
    """Read only audited official fit rows; tune/calibration/test files stay closed."""
    audit = read_json(root/"data/prepared/audit.json")
    path = root/"data/prepared/fit.jsonl"
    if sha256_file(path) != audit["prepared_sha256"]["fit.jsonl"]:
        raise ValueError("Original fit data checksum changed.")
    official = read_records(path)
    if not official or any(row["source_split"] != "train" or row["working_split"] != "fit"
                           or row["is_oos"] for row in official):
        raise ValueError("Only audited official known-intent fit rows may train card v2.")
    rows = []
    for i, row in enumerate(official):
        label = row["intent"] if row["intent"] in CATEGORIES else "oos"
        item = {"text": row["text"], "label": label, "family": row["example_id"], "origin": "clinc_fit"}
        rows.append(item)
        if label != "oos":
            for j in range(2):
                rows.append({**item, "text": noisy(row["text"], SEED+i*3+j), "origin": "clinc_fit_typo"})
    rows.extend(synthetic_rows())
    # Mechanical checks only; these references never become training examples.
    protected = {fingerprint(case["text"]) for path in (root/DEFAULT_CASES, root/CHECK_CASES)
                 for case in load_cases(path)["cases"]}
    if any(fingerprint(row["text"]) in protected for row in rows):
        raise ValueError("Training/check exact-fingerprint overlap; preserve the check and revise training.")
    # Avoid giving unchanged typo variants or repeated synthetic wrappers extra weight.
    unique = {}
    for row in rows:
        key = fingerprint(row["text"])
        if key in unique and unique[key]["label"] != row["label"]:
            raise ValueError("Conflicting training labels.")
        unique.setdefault(key, row)
    return list(unique.values()), sha256_file(path)


def train_card(root: Path):
    rows, fit_hash = training_rows(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root/"artifacts"/("card_v2_"+stamp)
    directory.mkdir(parents=True, exist_ok=False)
    binding = {"config": CONFIG, "fit_sha256": fit_hash,
               "source_sha256": {name: sha256_file(root/"src/intentdesk"/name) for name in SOURCES},
               "uv_lock_sha256": sha256_file(root/"uv.lock"), "sklearn_version": sklearn.__version__,
               "reserved_check_sha256": sha256_file(root/CHECK_CASES),
               "development_suite_sha256": sha256_file(root/DEFAULT_CASES)}
    write_json(directory/"training_protocol.json", binding)
    model = make_model()
    started = time.perf_counter()
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit([row["text"] for row in rows], [row["label"] for row in rows])
    if set(model.classes_) != set(CATEGORIES):
        raise ValueError("Card model must contain all eight workflow categories.")
    joblib.dump(model, directory/"model.joblib")
    metadata = {"run_id": directory.name, "status": "completed", "binding": binding,
                "classes": list(model.classes_), "training_rows": len(rows),
                "class_counts": dict(Counter(row["label"] for row in rows)),
                "origin_counts": dict(Counter(row["origin"] for row in rows)),
                "training_seconds": time.perf_counter()-started,
                "features": int(model.named_steps["classifier"].n_features_in_),
                "model_sha256": sha256_file(directory/"model.joblib"),
                "human_reviewed_synthetic_labels": 0, "automatic_actions": False,
                "check_exposed": False, "original_benchmarks_reopened": False}
    write_json(directory/"metadata.json", metadata)
    write_json(root/"artifacts/card_v2_latest.json", {"run_id": directory.name,
               "metadata_sha256": sha256_file(directory/"metadata.json")})
    return metadata


class CardRouter:
    def __init__(self, root: Path):
        pointer = read_json(root/"artifacts/card_v2_latest.json")
        name = pointer["run_id"]
        if not re.fullmatch(r"card_v2_[0-9TZ]+", name):
            raise ValueError("Invalid card model directory.")
        directory = root/"artifacts"/name
        if sha256_file(directory/"metadata.json") != pointer["metadata_sha256"]:
            raise ValueError("Card metadata checksum mismatch.")
        self.metadata = read_json(directory/"metadata.json")
        if (self.metadata["status"] != "completed" or self.metadata["automatic_actions"]
                or self.metadata["binding"]["config"] != CONFIG
                or self.metadata["binding"]["uv_lock_sha256"] != sha256_file(root/"uv.lock")
                or any(sha256_file(root/"src/intentdesk"/source) != digest
                       for source, digest in self.metadata["binding"]["source_sha256"].items())
                or sha256_file(directory/"model.joblib") != self.metadata["model_sha256"]):
            raise ValueError("Card artifact/config/source contract changed; preserve and retrain separately.")
        self.model = joblib.load(directory/"model.joblib")
        if set(self.model.classes_) != set(CATEGORIES):
            raise ValueError("Card model class contract mismatch.")

    def route(self, text):
        text = validate_text(text)
        started = time.perf_counter()
        if len(text.split()) > 180:
            return self.result("uncertain", [], "This request is too long for the card v2 demonstration. Please shorten it or review it manually.", started)
        with threadpool_limits(limits=2):
            scores = self.model.predict_proba([text])[0]
        if not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1):
            raise ValueError("Invalid card model scores.")
        order = np.argsort(scores)[::-1]
        candidates = [{"intent": str(self.model.classes_[i]), "score": float(scores[i])} for i in order[:3]]
        label = candidates[0]["intent"]
        if label not in ("oos", "uncertain") and (scores[order[0]] < CONFIG["minimum_supported_score"]
                or scores[order[0]]-scores[order[1]] < CONFIG["minimum_supported_margin"]):
            label = "uncertain"
        reason = ("Please describe one issue clearly, or choose which issue to handle first." if label == "uncertain"
                  else "This looks outside the six card-support categories; confirm or correct it." if label == "oos"
                  else "A card category looks plausible. A person must confirm or correct it.")
        return self.result(label, candidates, reason, started)

    def result(self, label, candidates, reason, started):
        return {"label": label, "name": CATEGORIES[label]["name"], "reason": reason,
                "candidates": candidates, "model_version": self.metadata["run_id"],
                "score_type": CONFIG["score_type"], "route_recommendation": "review_required",
                "action_executed": False, "production_approved": False,
                "synthetic_training_labels_human_reviewed": False,
                "elapsed_ms": (time.perf_counter()-started)*1000}
