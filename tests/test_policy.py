import numpy as np
import pytest

pytest.importorskip("torch")
from scipy.special import softmax
from intentdesk.common import write_json
from intentdesk.neural import CONFIG, REVISION
from intentdesk.policy import (FrozenModels, PROTOCOL, Router, binomial_upper, check_gate,
                               choose_policy, fit_temperature, make_prototypes,
                               policy_metrics, reliability, split_validation)


def test_temperature_reduces_overconfident_loss_without_changing_labels():
    logits = np.array([[8., 0.]] * 50 + [[0., 8.]] * 50)
    labels = np.array([0]*40 + [1]*10 + [1]*40 + [0]*10)
    result = fit_temperature(logits, labels)
    assert result["temperature"] > 1
    assert result["nll_after"] < result["nll_before"]
    np.testing.assert_array_equal(logits.argmax(1), (logits/result["temperature"]).argmax(1))


def test_temperature_rejects_oos_or_invalid_calibration_labels():
    with pytest.raises(ValueError, match="Invalid calibration"):
        fit_temperature(np.ones((2, 3)), np.array([0, -1]))


def test_validation_selection_and_check_are_deterministic_disjoint_and_stratified():
    rows = [{"example_id": f"{c}:{i}", "intent": c, "duplicate_group": f"{c}-{i}"}
            for c in ("a", "b", "oos") for i in range(100)]
    select, check = split_validation(rows)
    second_select, second_check = split_validation(rows)
    np.testing.assert_array_equal(select, second_select)
    np.testing.assert_array_equal(check, second_check)
    assert set(select).isdisjoint(check)
    assert len(select) == 60 and len(check) == 240
    assert sum(rows[i]["intent"] == "oos" for i in check) == 80
    rows[1]["duplicate_group"] = rows[0]["duplicate_group"]
    with pytest.raises(ValueError, match="duplicates"):
        split_validation(rows)


def test_semantic_support_can_reject_a_high_confidence_unsupported_request():
    probabilities = np.array([[.99, .01], [.01, .99], [.999, .001], [.9, .1]])
    support = np.array([.85, .85, .2, .1])
    labels = np.array([0, 1, -1, -1])
    selected, _ = choose_policy({"B2_MPNet_linear": (probabilities, support)}, labels)
    assert selected["selection_passed"]
    assert selected["accepted_oos"] == 0
    assert selected["accepted_in_scope"] == 2
    assert selected["cosine_threshold"] > .2


def test_review_all_fallback_when_no_policy_meets_selection_requirements():
    p = np.array([[.5, .5], [.5, .5]])
    selected, _ = choose_policy({"B2_MPNet_linear": (p, np.ones(2))}, np.array([0, -1]))
    assert not selected["selection_passed"]
    assert selected["confidence_threshold"] > 1


def test_weighted_metrics_use_stated_oos_mixture_instead_of_dataset_prevalence():
    probabilities = np.array([[.99, .01]]*9 + [[.01, .99]])
    labels = np.array([0]*9 + [-1])
    result = policy_metrics(probabilities, np.ones(10), labels, .9, 0)
    assert result["weighted_coverage"] == 1
    assert result["weighted_selective_error"] == pytest.approx(.2)
    assert result["weighted_cost"] == pytest.approx(2.)


def test_exact_uncertainty_does_not_treat_zero_observed_errors_as_zero_risk():
    assert binomial_upper(0, 80, .05/3) > .04
    good = {"in_scope_count": 2396, "oos_count": 80, "accepted_in_scope": 1800,
            "wrong_accepted_in_scope": 0, "accepted_oos": 0, "weighted_cost": .4}
    assert check_gate(good)["passed"]
    assert not check_gate({**good, "accepted_oos": 2})["passed"]
    assert not check_gate({**good, "accepted_in_scope": 0})["passed"]


def test_reliability_includes_perfect_confidence():
    report = reliability(np.array([[1., 0.], [0., 1.]]), np.array([0, 0]))
    assert report["bins"][0]["count"] == 2
    assert report["ece"] == .5


def test_prototypes_follow_declared_class_order_and_are_normalized():
    features = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    result = make_prototypes(features, [{"intent": "b"}, {"intent": "a"}], ["a", "b"])
    np.testing.assert_allclose(result, [[0., 1.], [1., 0.]])


def test_frozen_model_hash_tampering_is_rejected_before_deserialization(tmp_path, monkeypatch):
    directory = tmp_path/"artifacts"/"parent"
    directory.mkdir(parents=True)
    write_json(directory/"metadata.json", {"status": "completed", "revision": REVISION,
                                           "config": CONFIG, "classes": [str(i) for i in range(150)]})
    for name in ("linear.joblib", "N1_small.pt"):
        (directory/name).write_bytes(b"changed")
    monkeypatch.setattr("intentdesk.policy.joblib.load", lambda *a, **kw: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="checksum"):
        FrozenModels(tmp_path, "parent", {"linear.joblib": "old", "N1_small.pt": "old"})


def test_router_reviews_overlong_requests_without_truncating_and_classifying(monkeypatch):
    router = Router.__new__(Router)
    router.metadata = {"run_id": "test", "gate": {"passed": True}}
    router.tokenizer = lambda *args, **kwargs: {"input_ids": list(range(CONFIG["max_length"]+1))}
    monkeypatch.setattr("intentdesk.policy.encode_batch", lambda *a: pytest.fail("must not classify truncated request"))
    result = router.route("a long request")
    assert result["route_recommendation"] == "review_required"
    assert result["candidates"] == [] and result["action_executed"] is False


def test_failed_development_gate_forces_review_even_with_high_confidence(monkeypatch):
    from types import SimpleNamespace
    router = Router.__new__(Router)
    router.metadata = {"run_id": "test", "gate": {"passed": False},
                       "selected": {"model": "B2_MPNet_linear", "confidence_threshold": .5, "cosine_threshold": 0.},
                       "temperatures": {"B2_MPNet_linear": {"temperature": 1.}}}
    router.tokenizer = lambda *args, **kwargs: {"input_ids": [1, 2]}
    router.encoder = None
    router.models = SimpleNamespace(classes=["a", "b"], logits=lambda *a: np.array([[20., 0.]]))
    features = np.zeros((1, 768), dtype=np.float32)
    features[:, 0] = 1
    router.prototypes = np.repeat(features, 2, axis=0)
    monkeypatch.setattr("intentdesk.policy.encode_batch", lambda *a: features)
    result = router.route("a supported request")
    assert result["candidates"][0]["score"] > .99
    assert result["route_recommendation"] == "review_required"


def test_exposed_independent_check_cannot_be_used_for_reselection(tmp_path, monkeypatch):
    from intentdesk.policy import build_policy
    monkeypatch.setattr("intentdesk.policy.read_policy_data", lambda root: ({}, {}))
    write_json(tmp_path/"artifacts/policy_experiment.json", {"status": "selection_frozen", "check_exposed": True})
    with pytest.raises(ValueError, match="already exposed"):
        build_policy(tmp_path)
