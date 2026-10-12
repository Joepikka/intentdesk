import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from intentdesk import card_v2
from intentdesk.card_v2 import CardRouter, CHECK_CASES, CONFIG, SOURCES, negation_tokens, training_rows
from intentdesk.card_v2_data import noisy, synthetic_rows
from intentdesk.common import sha256_file, write_json
from intentdesk.pilot import CATEGORIES
from intentdesk.robustness import DEFAULT_CASES, load_cases


def test_negation_scope_preserves_fraud_cues_and_resets_at_contrast():
    assert negation_tokens("Card is not lost, but damaged.") == "card is not neg_lost but damaged"
    assert "neg_authorize" in negation_tokens("I didn't authorize the purchase.")
    assert "neg_permission" in negation_tokens("Someone used it without permission.")
    assert "neg_find" in negation_tokens("I can’t find it.")
    assert negation_tokens("not one two three four five six").endswith("neg_five six")


def test_augmented_families_remain_traceable_and_all_labels_present():
    rows = synthetic_rows()
    assert {row["label"] for row in rows} == set(CATEGORIES)
    assert all(row["family"] and row["origin"] == "ai_synthetic_development" for row in rows)
    assert noisy("replacement payment", 21) == noisy("replacement payment", 21)
    assert noisy("replacement payment", 21) != "replacement payment"
    suite = load_cases(Path(__file__).parents[1]/CHECK_CASES)
    assert len(suite["cases"]) == 48 and suite["label_status"] == "provisional_unreviewed"


def fixture_data(root):
    prepared = root/"data/prepared"; prepared.mkdir(parents=True)
    rows = [{"text": "unique fixture physical bank card broke", "intent": "damaged_card", "example_id": "train:fixture",
             "source_split": "train", "working_split": "fit", "is_oos": False}]
    path = prepared/"fit.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    write_json(prepared/"audit.json", {"prepared_sha256": {"fit.jsonl": sha256_file(path)}})
    # Files deliberately contain invalid JSON: opening a protected data split would fail.
    for name in ("tune", "calibrate", "locked_test", "validation"):
        (prepared/(name+".jsonl")).write_text("DO NOT READ", encoding="utf-8")
    for name in (DEFAULT_CASES, CHECK_CASES):
        write_json(root/name, {"version": "fixture", "provenance": "assistant_authored_synthetic_development_cases",
            "label_status": "provisional_unreviewed", "cases": [{"id": "check", "text": "reserved fixture message",
            "expected_label": "uncertain", "tags": ["missing_context"]}]})
    return path


def test_training_uses_only_audited_fit_and_rejects_check_overlap(tmp_path, monkeypatch):
    path = fixture_data(tmp_path)
    monkeypatch.setattr(card_v2, "synthetic_rows", lambda: [])
    rows, _ = training_rows(tmp_path)
    assert rows and all(row["family"] == "train:fixture" for row in rows)
    monkeypatch.setattr(card_v2, "synthetic_rows", lambda: [{"text": "reserved fixture message", "label": "uncertain"}])
    with pytest.raises(ValueError, match="overlap"):
        training_rows(tmp_path)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        training_rows(tmp_path)


def test_fit_source_contract_blocks_holdout_rows(tmp_path):
    path = fixture_data(tmp_path)
    row = json.loads(path.read_text()); row["working_split"] = "calibrate"
    path.write_text(json.dumps(row), encoding="utf-8")
    write_json(path.parent/"audit.json", {"prepared_sha256": {"fit.jsonl": sha256_file(path)}})
    with pytest.raises(ValueError, match="Only audited"):
        training_rows(tmp_path)


@pytest.fixture
def trained(tmp_path, monkeypatch):
    source = tmp_path/"src/intentdesk"; source.mkdir(parents=True)
    actual = Path(card_v2.__file__).parent
    for name in SOURCES:
        (source/name).write_bytes((actual/name).read_bytes())
    (tmp_path/"uv.lock").write_text("fixture lock", encoding="utf-8")
    for name in (CHECK_CASES, DEFAULT_CASES):
        write_json(tmp_path/name, {"fixture": True})
    rows = [{"text": f"{label.replace('_', ' ')} fixture wording variation {i}", "label": label,
             "family": label, "origin": "fixture"} for label in CATEGORIES for i in range(4)]
    monkeypatch.setattr(card_v2, "training_rows", lambda root: (rows, "fixture"))
    metadata = card_v2.train_card(tmp_path)
    return tmp_path, metadata


def test_train_reload_and_tamper_rejection(trained):
    root, metadata = trained
    router = CardRouter(root)
    assert metadata["training_rows"] == 32 and metadata["check_exposed"] is False
    assert metadata["automatic_actions"] is False
    assert (root/"artifacts"/metadata["run_id"]/"training_protocol.json").exists()
    assert router.route("payment card help")["route_recommendation"] == "review_required"
    path = root/"artifacts"/metadata["run_id"]/"model.joblib"
    path.write_bytes(path.read_bytes()+b"tamper")
    with pytest.raises(ValueError, match="contract changed"):
        CardRouter(root)


@pytest.mark.parametrize("text", ["", " ", 42, "x"*4001])
def test_bad_request_is_rejected_before_predict(text):
    router = object.__new__(CardRouter)
    with pytest.raises(ValueError):
        router.route(text)


def fake_router(scores):
    router = object.__new__(CardRouter)
    router.metadata = {"run_id": "fixture"}
    router.model = SimpleNamespace(classes_=np.array(list(CATEGORIES)),
                                  predict_proba=lambda texts: np.array([scores]))
    return router


def test_low_margin_supported_prediction_asks_for_clarification():
    result = fake_router([.41, .40, .04, .03, .03, .03, .03, .03]).route("fixture card request")
    assert result["label"] == "uncertain" and result["action_executed"] is False
    assert result["candidates"][0]["intent"] == "report_lost_card"
    assert "uncalibrated" in result["score_type"]


def test_specific_and_outside_predictions_and_length_guard():
    scores = [.9]+[.1/7]*7
    router = fake_router(scores)
    assert router.route("fixture card missing")["label"] == "report_lost_card"
    assert router.route("word "*181)["label"] == "uncertain"
    scores = [.1/7]*8; scores[6] = .9
    assert fake_router(scores).route("fixture other service")["label"] == "oos"
