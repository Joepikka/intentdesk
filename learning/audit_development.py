"""Report a declared development-label sensitivity; never edit data or train weights."""

import numpy as np
from pathlib import Path
from scipy.special import softmax
import torch
from threadpoolctl import threadpool_limits

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.neural import make_head, validate_feature_cache
from intentdesk.policy import FrozenModels
from intentdesk.predict import load_baseline
from intentdesk.train import classification_metrics, verify_development_data


def main():
    root = Path(__file__).resolve().parents[1]
    _, tune, audit = verify_development_data(root)
    flags = read_json(root/"data/quality_flags.json")["flags"]
    flagged_ids = {r["example_id"] for r in flags}
    for flag in flags:
        matches = [r for r in tune if r["example_id"] == flag["example_id"]]
        if len(matches) != 1 or matches[0]["text"] != flag["text"] or matches[0]["intent"] != flag["recorded_intent"]:
            raise ValueError("Declared quality flag does not match the audited development row.")
    keep = np.array([r["example_id"] not in flagged_ids for r in tune])
    labels = np.array([r["intent"] for r in tune])
    run_id = read_json(root/"artifacts/neural_latest.json")["run_id"]
    models = FrozenModels(root, run_id)
    cache = read_json(root/"data/embeddings/manifest.json")
    features = validate_feature_cache(root/"data/embeddings/tune.npy", len(tune), cache["sha256"]["tune"])
    baseline, _ = load_baseline(root)
    with threadpool_limits(limits=2):
        b1 = baseline.predict_proba([r["text"] for r in tune])
    if list(baseline.classes_) != models.classes:
        raise ValueError("Class ordering changed.")
    probabilities = {"B1_word_char_tfidf": b1}
    for name in ("B2_MPNet_linear", "N1_small"):
        probabilities[name] = softmax(models.logits(name, features), axis=1)
    checkpoint = models.metadata["checkpoints"]["N2_large"]
    path = models.directory/"N2_large.pt"
    if sha256_file(path) != checkpoint["sha256"]:
        raise ValueError("Larger checkpoint changed.")
    head = make_head(checkpoint["widths"]).eval()
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        probabilities["N2_large"] = head(torch.from_numpy(features)).double().softmax(1).numpy()
    result = {"neural_run": run_id, "reviewer": "AI assistant", "human_reviewed": False,
              "data_modified": False, "test_evaluated": False,
              "tune_input_sha256": audit["prepared_sha256"]["tune.jsonl"],
              "excluded_only_in_supplementary_analysis": sorted(flagged_ids), "models": {}}
    for name, p in probabilities.items():
        predicted = np.array(models.classes)[p.argmax(1)]
        result["models"][name] = {
            "original": classification_metrics(labels.tolist(), predicted, p, models.classes),
            "supplementary_excluding_flagged_row": classification_metrics(labels[keep].tolist(), predicted[keep], p[keep], models.classes)}
    write_json(root/"reports/development_label_sensitivity.json", result)
    print("Original labels preserved. Supplementary development sensitivity saved.")
    for name, values in result["models"].items():
        print(f"{name}: original {values['original']['accuracy']:.4%}, supplementary {values['supplementary_excluding_flagged_row']['accuracy']:.4%}")


if __name__ == "__main__":
    main()
