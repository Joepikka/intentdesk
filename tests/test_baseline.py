import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from intentdesk.common import sha256_file, write_json
from intentdesk.predict import load_baseline, predict, validate_text
from intentdesk.train import classification_metrics, make_pipeline


@pytest.fixture
def baseline(tmp_path):
    texts = [
        "track package delivery", "package delivery status", "delivery track package", "where package delivery",
        "reset password login", "password reset account", "login forgot password", "account password reset",
        "bank account balance", "check balance account", "balance bank check", "account balance bank",
    ]
    labels = ["delivery"] * 4 + ["password"] * 4 + ["balance"] * 4
    model = make_pipeline()
    with threadpool_limits(limits=2):
        model.fit(texts, labels)
    run = tmp_path / "artifacts" / "unit-test"
    run.mkdir(parents=True)
    joblib.dump(model, run / "baseline.joblib")
    write_json(run / "metadata.json", {
        "run": {"run_id": "unit-test", "model_sha256": sha256_file(run / "baseline.joblib")},
        "intent_domains": {label: "demo" for label in set(labels)},
    })
    write_json(tmp_path / "artifacts" / "latest.json", {"run_id": "unit-test"})
    return tmp_path, model


def test_inference_does_not_fit_vocabulary_on_new_requests(baseline):
    _, model = baseline
    features = model.named_steps["features"].transformer_list
    before = {name: dict(vectorizer.vocabulary_) for name, vectorizer in features}
    with threadpool_limits(limits=2):
        model.predict_proba(["neverobservedunicornword"])
    assert before == {name: dict(vectorizer.vocabulary_) for name, vectorizer in features}
    assert "neverobservedunicornword" not in dict(features)["word"].vocabulary_


def test_artifact_roundtrip_and_routing_is_disabled(baseline):
    root, original = baseline
    loaded, _ = load_baseline(root)
    with threadpool_limits(limits=2):
        assert np.allclose(original.predict_proba(["reset password"]), loaded.predict_proba(["reset password"]))
    response = predict(root, "reset password")
    assert response["candidates"][0]["intent"] == "password"
    assert response["route_decision"] == "review_required"
    assert response["calibrated"] is False
    assert response["score_type"] == "uncalibrated_multiclass_probability"


def test_changed_artifact_is_rejected(baseline):
    root, _ = baseline
    (root / "artifacts" / "unit-test" / "baseline.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_baseline(root)


def test_pointer_cannot_load_outside_artifact_directory(tmp_path):
    write_json(tmp_path / "artifacts" / "latest.json", {"run_id": "../../outside"})
    with pytest.raises(ValueError, match="path"):
        load_baseline(tmp_path)


def test_missing_model_has_an_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="No trained model"):
        load_baseline(tmp_path)


@pytest.mark.parametrize("text", ["", "   ", "a" * 4001, None])
def test_invalid_input_is_rejected(text):
    with pytest.raises(ValueError):
        validate_text(text)


def test_metric_convention_is_explicit():
    p = np.array([[0.8, 0.2], [0.1, 0.9]])
    result = classification_metrics(["a", "b"], ["a", "b"], p, ["a", "b"])
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 1
    assert result["multiclass_brier"] == pytest.approx(0.05)


def test_neural_float32_scores_use_same_log_loss_precision_as_sklearn():
    probabilities = np.array([[1e-10, 1.0]], dtype=np.float32)
    result = classification_metrics(["a"], ["b"], probabilities, ["a", "b"])
    assert result["log_loss"] == pytest.approx(-np.log(float(probabilities[0, 0])), abs=1e-6)


@pytest.mark.parametrize("probabilities", [np.array([[np.nan, 1.0]]), np.array([[.2, .2]])])
def test_metrics_reject_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="probability"):
        classification_metrics(["a"], ["b"], probabilities, ["a", "b"])

