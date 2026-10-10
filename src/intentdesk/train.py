"""Train one fixed lexical baseline. Evaluate development data only."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from intentdesk import __version__
from intentdesk.common import read_json, read_records, sha256_file, write_json

CONFIG = {
    "word_ngram_range": [1, 2], "word_max_features": 15000,
    "char_ngram_range": [3, 5], "char_max_features": 20000,
    "min_df": 2, "sublinear_tf": True, "C": 4.0,
    "solver": "lbfgs", "max_iter": 500, "tol": 0.0001,
    "seed": 20261010, "blas_threads": 2,
}


def make_pipeline() -> Pipeline:
    """Feature transforms learn only inside fit(), on the fit partition."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=tuple(CONFIG["word_ngram_range"]), max_features=CONFIG["word_max_features"],
            min_df=CONFIG["min_df"], sublinear_tf=CONFIG["sublinear_tf"],
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=tuple(CONFIG["char_ngram_range"]),
            max_features=CONFIG["char_max_features"], min_df=CONFIG["min_df"],
            sublinear_tf=CONFIG["sublinear_tf"],
        )),
    ])
    classifier = LogisticRegression(
        C=CONFIG["C"], solver=CONFIG["solver"], max_iter=CONFIG["max_iter"],
        tol=CONFIG["tol"], random_state=CONFIG["seed"],
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def classification_metrics(y: list[str], predicted, probabilities, classes) -> dict:
    # sklearn clips at the input dtype's epsilon. Use the same precision for
    # float32 neural scores and float64 sklearn scores before computing losses.
    probabilities = np.asarray(probabilities, dtype=np.float64)
    totals = probabilities.sum(axis=1, keepdims=True)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(totals, 1.0, rtol=0, atol=1e-6)):
        raise ValueError("Expected finite probability rows summing to one.")
    # Remove float32 softmax roundoff before float64 log-loss validation.
    probabilities = probabilities / totals
    classes = list(classes)
    index = {label: i for i, label in enumerate(classes)}
    targets = np.zeros_like(probabilities)
    targets[np.arange(len(y)), [index[label] for label in y]] = 1
    return {
        "samples": len(y), "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "brier_convention": "mean over rows of sum over classes of squared probability error; range 0 to 2",
    }


def verify_development_data(root: Path) -> tuple[list[dict], list[dict], dict]:
    prepared = root / "data" / "prepared"
    audit = read_json(prepared / "audit.json")
    for name in ["fit.jsonl", "tune.jsonl"]:
        if sha256_file(prepared / name) != audit["prepared_sha256"][name]:
            raise ValueError(f"Prepared {name} changed; run prepare again.")
    if sha256_file(prepared / "manifest.csv") != audit["manifest_sha256"]:
        raise ValueError("Split manifest changed; run prepare again.")
    fit = read_records(prepared / "fit.jsonl")
    tune = read_records(prepared / "tune.jsonl")
    if not fit or not tune:
        raise ValueError("Fit and tune partitions must be non-empty.")
    for rows, split in [(fit, "fit"), (tune, "tune")]:
        if any(r["working_split"] != split or r["source_split"] != "train" or r["is_oos"] for r in rows):
            raise ValueError(f"Only in-scope official training rows may enter {split}.")
    if {r["duplicate_group"] for r in fit} & {r["duplicate_group"] for r in tune}:
        raise ValueError("Fit/tune duplicate leakage detected.")
    if len({r["intent"] for r in fit}) != 150 or {r["intent"] for r in fit} != {r["intent"] for r in tune}:
        raise ValueError("Both development partitions must contain all 150 intents.")
    return fit, tune, audit


def plot_results(report_dir: Path, metrics: dict, fit: list[dict], errors: list[dict]) -> None:
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(names))
    ax.bar(positions - 0.18, [metrics[n]["accuracy"] for n in names], width=0.36, label="Accuracy")
    ax.bar(positions + 0.18, [metrics[n]["macro_f1"] for n in names], width=0.36, label="Macro-F1")
    ax.set(xticks=positions, xticklabels=names, ylim=(0, 1), ylabel="Score", title="Development comparison — tune partition only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "baseline_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist([len(r["text"].split()) for r in fit], bins=25, color="#177e89")
    ax.set(xlabel="Whitespace-delimited tokens", ylabel="Fit examples", title="Request lengths — fit data only")
    fig.tight_layout()
    fig.savefig(report_dir / "request_lengths.png", dpi=160)
    plt.close(fig)

    pairs = Counter((r["true_intent"], r["predicted_intent"]) for r in errors).most_common(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    if pairs:
        ax.barh([f"{a} → {b}" for (a, b), _ in pairs][::-1], [count for _, count in pairs][::-1], color="#b64f47")
        ax.xaxis.get_major_locator().set_params(integer=True)
    else:
        ax.text(0.5, 0.5, "No errors on this development partition", ha="center")
    ax.set(xlabel="Number of mistakes", title="Most frequent confusion pairs — tune data only")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_pairs.png", dpi=160)
    plt.close(fig)


def train(root: Path) -> dict:
    fit, tune, audit = verify_development_data(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = root / "artifacts" / run_id
    report_dir = root / "reports" / "runs" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "intentdesk": __version__,
    }
    source_hashes = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    metadata = {
        "run_id": run_id, "status": "started", "config": CONFIG,
        "source_hashes": source_hashes, "environment": environment,
        "manifest_sha256": audit["manifest_sha256"], "dataset_sha256": audit["dataset_sha256"],
        "evaluation_split": "tune", "test_evaluated": False,
        "probabilities_calibrated": False, "automatic_routing_enabled": False,
    }
    write_json(report_dir / "run.json", metadata)
    x_fit, y_fit = [r["text"] for r in fit], [r["intent"] for r in fit]
    x_tune, y_tune = [r["text"] for r in tune], [r["intent"] for r in tune]
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(fit), 1)), y_fit)
    dummy_input = np.zeros((len(tune), 1))
    b0 = classification_metrics(y_tune, dummy.predict(dummy_input), dummy.predict_proba(dummy_input), dummy.classes_)
    model = make_pipeline()
    print(f"Training B1 on {len(fit):,} fit requests; checking {len(tune):,} tune requests...", flush=True)
    started = time.perf_counter()
    with threadpool_limits(limits=CONFIG["blas_threads"]), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, y_fit)
        fit_seconds = time.perf_counter() - started
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            metadata.update(status="failed", reason="Logistic regression did not converge.")
            write_json(report_dir / "run.json", metadata)
            raise ValueError("Training did not converge; failed run recorded and no new model published.")
        probabilities = model.predict_proba(x_tune)
        predicted = model.classes_[probabilities.argmax(axis=1)]
        b1 = classification_metrics(y_tune, predicted, probabilities, model.classes_)
        model_path = artifact_dir / "baseline.joblib"
        joblib.dump(model, model_path, compress=3)
        reloaded = joblib.load(model_path)
        if not np.allclose(reloaded.predict_proba(x_tune[:20]), probabilities[:20], atol=1e-12, rtol=1e-12):
            raise ValueError("Saved model failed reload parity; model not published.")
        for text in x_tune[:5]:
            reloaded.predict_proba([text])
        timings = []
        for text in x_tune[:50]:
            start = time.perf_counter()
            reloaded.predict_proba([text])
            timings.append((time.perf_counter() - start) * 1000)
    metrics = {"B0_majority": b0, "B1_word_char_tfidf": b1}
    errors = [
        {"example_id": row["example_id"], "text": row["text"], "true_intent": row["intent"],
         "predicted_intent": str(predicted[i]), "raw_score": float(probabilities[i].max()),
         "reviewed_by_human": False}
        for i, row in enumerate(tune) if row["intent"] != predicted[i]
    ]
    write_json(report_dir / "metrics.json", metrics)
    write_json(report_dir / "per_intent.json", classification_report(
        y_tune, predicted, labels=model.classes_, output_dict=True, zero_division=0,
    ))
    write_json(report_dir / "development_errors.json", errors)
    plot_results(report_dir, metrics, fit, errors)
    metadata.update(
        status="completed", fit_samples=len(fit), tune_samples=len(tune), fit_seconds=fit_seconds,
        feature_count=sum(len(v.vocabulary_) for _, v in model.named_steps["features"].transformer_list),
        iterations=int(model.named_steps["classifier"].n_iter_.max()),
        model_sha256=sha256_file(model_path), model_bytes=model_path.stat().st_size,
        warm_latency={"unit": "milliseconds", "samples": len(timings), "batch_size": 1,
                      "median": float(np.median(timings)), "p95": float(np.percentile(timings, 95)),
                      "scope": "full TF-IDF plus classifier, sequential local requests, two BLAS threads; not a load test"},
        reload_max_absolute_error=float(np.max(np.abs(reloaded.predict_proba(x_tune[:20]) - probabilities[:20]))),
    )
    bundle = {"run": metadata, "intent_domains": {r["intent"]: r["domain"] for r in fit}}
    write_json(artifact_dir / "metadata.json", bundle)
    write_json(report_dir / "run.json", metadata)
    report = f"""# First lexical baseline — development result

Run: `{run_id}`. **Final test not evaluated.** These are tune-set results, not production results.

| Model | Accuracy | Macro-F1 | Log loss | Multiclass Brier |
| --- | ---: | ---: | ---: | ---: |
| Majority | {b0['accuracy']:.4f} | {b0['macro_f1']:.4f} | {b0['log_loss']:.4f} | {b0['multiclass_brier']:.4f} |
| Word + char TF-IDF / logistic regression | {b1['accuracy']:.4f} | {b1['macro_f1']:.4f} | {b1['log_loss']:.4f} | {b1['multiclass_brier']:.4f} |

Fit: {len(fit):,}; tune: {len(tune):,}; labels: 150. Training: {fit_seconds:.1f} s.
One fixed configuration; no hyperparameter search or champion selection performed.
Raw classifier scores are uncalibrated. Automatic routing is disabled.
Neither embeddings nor a neural network nor generation have been implemented yet.

![Baseline comparison](baseline_comparison.png)
![Fit request lengths](request_lengths.png)
![Development confusion pairs](confusion_pairs.png)

Review `development_errors.json`: {len(errors)} errors, **not yet manually reviewed**.
See `run.json` for data/source hashes, environment, model size, reload parity and measured latency.
The unused calibration, official validation and final-test partitions remain separate.
Public, crowdsourced, single-intent English data cannot establish performance on real customer traffic.
"""
    (report_dir / "summary.md").write_text(report, encoding="utf-8")
    write_json(root / "artifacts" / "latest.json", {"run_id": run_id})
    write_json(root / "reports" / "latest.json", {"run_id": run_id})
    return {"run_id": run_id, "metrics": metrics, "report": str(report_dir / "summary.md"), "fit_seconds": fit_seconds}

