import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from intentdesk.neural import configure, fit_head, make_head, open_feature_progress, validate_feature_cache, predict_neural
from intentdesk.common import sha256_file


def test_large_head_parameter_count_and_backward():
    configure()
    model = make_head([768, 1024, 512, 150])
    assert sum(p.numel() for p in model.parameters()) == 1_389_206
    logits = model(torch.randn(4, 768))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3, 4]))
    loss.backward()
    assert logits.shape == (4, 150)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_learns_tiny_separable_problem_and_reload(tmp_path):
    configure()
    # Separate samples from three easily distinguishable clusters.
    labels = torch.arange(3).repeat(16)
    centers = torch.eye(3) * 4
    x_fit = centers[labels] + torch.randn(48, 3) * .01
    x_tune = centers[labels] + torch.randn(48, 3) * .01
    model, history, selected = fit_head(x_fit, labels, x_tune, labels, [3, 16, 3], max_epochs=100, patience=100)
    with torch.inference_mode():
        predicted = model(x_tune).argmax(1)
    assert (predicted == labels).float().mean() == 1
    assert history[-1]["fit_loss"] < history[0]["fit_loss"]
    assert selected == min(history, key=lambda h: h["tune_loss"])["epoch"]
    path = tmp_path / "head.pt"
    torch.save(model.state_dict(), path)
    restored = make_head([3, 16, 3]).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.inference_mode():
        torch.testing.assert_close(restored(x_tune), model(x_tune), rtol=0, atol=0)


def test_missing_cache_with_completed_progress_cannot_be_recreated_as_zeros(tmp_path):
    with pytest.raises(ValueError, match="missing file"):
        open_feature_progress(tmp_path/"missing.npy", 2, 2)
    assert not (tmp_path/"missing.npy").exists()


def test_modified_or_zero_embeddings_cannot_be_blessed_as_completed(tmp_path):
    import numpy as np
    path = tmp_path/"features.npy"
    features = np.zeros((2, 768), dtype=np.float32)
    features[:, 0] = 1
    np.save(path, features)
    digest = sha256_file(path)
    validate_feature_cache(path, 2, digest)
    features[:] = 0
    np.save(path, features)
    with pytest.raises(ValueError, match="checksum"):
        validate_feature_cache(path, 2, digest)
    with pytest.raises(ValueError, match="normalized"):
        validate_feature_cache(path, 2, sha256_file(path))


@pytest.mark.parametrize("text", [None, "", "   ", "a"*4001])
def test_neural_input_validation_happens_before_loading_models(tmp_path, text):
    with pytest.raises(ValueError):
        predict_neural(tmp_path, text)

