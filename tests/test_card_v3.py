import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from intentdesk import card_v3
from intentdesk.card_v3 import CHECK_CASES, CONFIG, SOURCES, CardRouter, combined
from intentdesk.common import sha256_file, write_json
from intentdesk.pilot import CATEGORIES
from intentdesk.robustness import load_cases


def test_combined_features_have_declared_weights_and_order():
    result = combined(csr_matrix([[1., 2.]]), np.array([[.1, .2]]))
    assert result.toarray().tolist()[0] == pytest.approx([.5, 1., .3, .6])
    suite = load_cases(Path(__file__).parents[1]/CHECK_CASES)
    assert len(suite["cases"]) == 32 and {c["expected_label"] for c in suite["cases"]} == set(CATEGORIES)


def tokenizer(text, **kwargs):
    return {"input_ids": list(range(len(text.split())+2))}


def embedding(tokenizer, encoder, texts):
    result = np.zeros((len(texts), 768), dtype=np.float32); result[:, 0] = 1
    return result


@pytest.fixture
def trained(tmp_path, monkeypatch):
    actual = Path(card_v3.__file__).parent
    for name in SOURCES:
        target = tmp_path/"src/intentdesk"/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((actual/name).read_bytes())
    (tmp_path/"uv.lock").write_text("fixture", encoding="utf-8")
    write_json(tmp_path/"models/mpnet/manifest.json", {"fixture": True})
    write_json(tmp_path/CHECK_CASES, {"version": "fixture", "provenance": "assistant_authored_synthetic_development_cases",
        "label_status": "provisional_unreviewed", "cases": [{"id": "reserved", "text": "unused fixture request",
        "expected_label": "uncertain", "tags": ["missing_context"]}]})
    write_json(tmp_path/"data/prepared/fixture.json", {})
    (tmp_path/"data/prepared/fit.jsonl").write_text(json.dumps({"example_id": "a"}), encoding="utf-8")
    cache = tmp_path/"data/embeddings/fit.npy"; cache.parent.mkdir(parents=True)
    np.save(cache, embedding(None, None, ["fixture"]), allow_pickle=False)
    write_json(tmp_path/"data/embeddings/manifest.json", {"input_sha256": {"fit": "fixture"},
               "revision": card_v3.REVISION, "sha256": {"fit": sha256_file(cache)}})
    rows = [{"text": label.replace("_", " ")+f" fixture variant {i}", "label": label,
             "family": "a", "origin": "clinc_fit" if i == 0 else "fixture"}
            for label in CATEGORIES for i in range(4)]
    monkeypatch.setattr(card_v3, "training_rows", lambda root: (rows, "fixture"))
    monkeypatch.setattr(card_v3, "load_encoder", lambda root: (tokenizer, None, {}))
    monkeypatch.setattr(card_v3, "encode_batch", embedding)
    metadata = card_v3.train_card(tmp_path)
    return tmp_path, metadata


def test_train_reuses_original_features_and_hash_checks_before_reload(trained):
    root, metadata = trained
    assert metadata["training_rows"] == 32 and metadata["new_embeddings"] == 24
    assert metadata["encoder_trainable_parameters"] == 0 and metadata["check_exposed"] is False
    router = CardRouter(root, shared_encoder=(tokenizer, None))
    assert router.route("fixture card request")["action_executed"] is False
    path = root/"artifacts"/metadata["run_id"]/"model.joblib"
    path.write_bytes(b"invalid pickle")
    with pytest.raises(ValueError, match="identity changed"):
        CardRouter(root, shared_encoder=(tokenizer, None))


def test_too_long_input_never_encodes_a_truncated_prefix(monkeypatch):
    router = object.__new__(CardRouter); router.metadata = {"run_id": "fixture"}; router.tokenizer = tokenizer
    def forbidden(*args):
        raise AssertionError("Encoder must not run for over-budget input")
    monkeypatch.setattr(card_v3, "encode_batch", forbidden)
    result = router.route("words "*129)
    assert result["label"] == "uncertain" and not result["candidates"]
    assert "no truncated prefix" in result["reason"]
