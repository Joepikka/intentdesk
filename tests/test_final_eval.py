import numpy as np
import pytest

pytest.importorskip("torch")
from intentdesk.final_eval import PROTOCOL, grouped_accuracy_interval, measure_known, seal_experiment


def test_final_experiment_is_bound_before_exposure_and_cannot_be_reselected(tmp_path):
    marker = seal_experiment(tmp_path, {"primary": "fixed", "weights": "hash"})
    assert not marker["test_exposed"] and marker["status"] == "frozen_before_exposure"
    assert seal_experiment(tmp_path, {"primary":"fixed","weights":"hash"}) == marker
    with pytest.raises(ValueError, match="already bound"):
        seal_experiment(tmp_path, {"primary":"different","weights":"hash"})


def test_grouped_bootstrap_is_deterministic_and_keeps_duplicate_groups_together():
    correct = np.array([1,1,0,0])
    first = grouped_accuracy_interval(correct, ["a","a","b","b"])
    assert first == grouped_accuracy_interval(correct, ["a","a","b","b"])
    assert first == [0., 1.]
    assert grouped_accuracy_interval(np.ones(4), ["a","b","c","d"]) == [1.,1.]


def test_final_primary_and_review_behavior_are_explicit_before_measurement():
    assert PROTOCOL["primary"] == "B2_MPNet_linear_temperature_scaled"
    assert "forbidden" in PROTOCOL["selection_after_test"]
    assert "review all" in PROTOCOL["active_behavior"]


def test_final_metrics_use_class_order_and_actual_predictions():
    result = measure_known(["a", "b"], np.array([[.8,.2],[.7,.3]]), ["a","b"])
    assert result["accuracy"] == .5 and result["samples"] == 2
    assert result["log_loss"] > 0
