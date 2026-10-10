"""Local frozen MPNet features and a supervised PyTorch head. Development only."""

from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import os
import platform
import time
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import psutil
import torch
from torch import nn
from huggingface_hub import snapshot_download
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.train import classification_metrics, verify_development_data
from intentdesk.predict import validate_text

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
CONFIG = {"seed": 20261010, "device": "cpu", "threads": 2,
          "embedding_batch": 8, "max_length": 128, "dimension": 768,
          "batch_size": 128, "learning_rate": 0.001, "weight_decay": 0.01,
          "dropout": 0.2, "max_epochs": 40, "patience": 6,
          "selection": "minimum tune cross-entropy; earliest tie",
          "encoder_frozen": True}


def configure() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)


def load_encoder(root: Path):
    configure()
    location = root / "models" / "mpnet"
    manifest_path = location / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
        if (manifest.get("revision") != REVISION or manifest.get("model_id") != MODEL_ID
                or not required.issubset(manifest.get("sha256", {}))):
            raise ValueError("Encoder identity or required-file manifest mismatch.")
        for name, digest in manifest["sha256"].items():
            if sha256_file(location / name) != digest:
                raise ValueError(f"Encoder file changed: {name}")
    else:
        print("Downloading pinned MPNet weights and tokenizer (local CPU execution)...", flush=True)
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=location,
                          allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                          "tokenizer_config.json", "special_tokens_map.json",
                                          "vocab.txt", "README.md", "LICENSE"])
        manifest = {"model_id": MODEL_ID, "revision": REVISION,
                    "sha256": {p.name: sha256_file(p) for p in location.iterdir() if p.is_file()}}
        write_json(manifest_path, manifest)
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(location, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True).eval()
    model.requires_grad_(False)
    if model.config.hidden_size != CONFIG["dimension"]:
        raise ValueError("Unexpected encoder dimension.")
    return tokenizer, model, manifest


def encode_batch(tokenizer, encoder, texts: list[str]) -> np.ndarray:
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=CONFIG["max_length"], return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        result = nn.functional.normalize(pooled, p=2, dim=1)
    return result.numpy()


def validate_feature_cache(path: Path, rows: int, expected_sha256: str) -> np.ndarray:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"Embedding cache missing or checksum mismatch: {path.name}")
    array = np.load(path, allow_pickle=False)
    if (array.shape != (rows, CONFIG["dimension"]) or array.dtype != np.float32
            or not np.isfinite(array).all()
            or not np.allclose(np.linalg.norm(array, axis=1), 1., atol=1e-4, rtol=0)):
        raise ValueError(f"Invalid normalized embedding array: {path.name}")
    return array


def open_feature_progress(path: Path, rows: int, offset: int):
    if not isinstance(offset, int) or offset < 0 or offset > rows:
        raise ValueError("Invalid embedding progress offset.")
    if offset and not path.exists():
        raise ValueError("Embedding progress references a missing file; refusing to create a false completed cache.")
    features = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                        dtype=np.float32, shape=(rows, CONFIG["dimension"]))
    if features.shape != (rows, CONFIG["dimension"]) or features.dtype != np.float32:
        raise ValueError("Invalid embedding cache shape or dtype.")
    return features


def embed(root: Path) -> dict:
    """Cache fit/tune only; completed batches survive interruption."""
    fit, tune, audit = verify_development_data(root)
    tokenizer, encoder, encoder_manifest = load_encoder(root)
    directory = root / "data" / "embeddings"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"model_id": MODEL_ID, "revision": REVISION, "config": CONFIG,
                "input_sha256": {name: audit["prepared_sha256"][name + ".jsonl"] for name in ("fit", "tune")},
                "encoder_manifest_sha256": sha256_file(root / "models" / "mpnet" / "manifest.json")}
    completed_manifest = directory / "manifest.json"
    if completed_manifest.exists():
        cached = read_json(completed_manifest)
        if any(cached.get(k) != value for k, value in identity.items()):
            raise ValueError("Completed embedding cache identity changed.")
        for name, rows in (("fit", fit), ("tune", tune)):
            validate_feature_cache(directory / (name + ".npy"), len(rows), cached["sha256"][name])
        return cached  # Never bless modified arrays by writing new checksums.
    progress_path = directory / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"identity": identity, "completed": {}}
    if progress["identity"] != identity:
        raise ValueError("Embedding cache belongs to a different model/data/configuration; preserve it elsewhere before rebuilding.")
    started = time.perf_counter()
    for name, rows in (("fit", fit), ("tune", tune)):
        path = directory / (name + ".npy")
        offset = progress["completed"].get(name, 0)
        features = open_feature_progress(path, len(rows), offset)
        if offset:
            digest = hashlib.sha256(features[:offset].tobytes()).hexdigest()
            if progress.get("completed_sha256", {}).get(name) != digest:
                raise ValueError("Partial embedding checkpoint checksum mismatch.")
        for start in range(offset, len(rows), CONFIG["embedding_batch"]):
            stop = min(start + CONFIG["embedding_batch"], len(rows))
            features[start:stop] = encode_batch(tokenizer, encoder, [r["text"] for r in rows[start:stop]])
            if stop == len(rows) or stop % 256 == 0:
                features.flush()
                progress["completed"][name] = stop
                progress.setdefault("completed_sha256", {})[name] = hashlib.sha256(features[:stop].tobytes()).hexdigest()
                write_json(progress_path, progress)
                print(f"Encoded {name}: {stop:,}/{len(rows):,}; elapsed {time.perf_counter()-started:.0f}s", flush=True)
        del features
    lengths = tokenizer([r["text"] for r in fit + tune], truncation=False, padding=False)["input_ids"]
    metadata = {**identity, "completed": progress["completed"], "test_evaluated": False,
                "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
                "encoder_trainable_parameters": 0, "elapsed_this_invocation_seconds": time.perf_counter()-started,
                "truncated_development_requests": sum(len(ids) > CONFIG["max_length"] for ids in lengths),
                "sha256": {name: sha256_file(directory / (name + ".npy")) for name in ("fit", "tune")},
                "process_peak_working_set_bytes": psutil.Process().memory_info().peak_wset,
                "encoder_files": encoder_manifest["sha256"]}
    write_json(directory / "manifest.json", metadata)
    return metadata


def make_head(widths: list[int]) -> nn.Sequential:
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.extend([nn.ReLU(), nn.Dropout(CONFIG["dropout"])])
    return nn.Sequential(*layers)


def fit_head(x_fit, y_fit, x_tune, y_tune, widths, *, max_epochs=None, patience=None):
    configure()
    model = make_head(widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    for epoch in range(1, (max_epochs or CONFIG["max_epochs"]) + 1):
        model.train()
        total_loss, total_correct, gradient_norm = 0., 0, 0.
        order = torch.randperm(len(x_fit), generator=generator)
        for batch in order.split(CONFIG["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_fit[batch])
            loss = criterion(logits, y_fit[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite neural loss; no checkpoint published.")
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_correct += (logits.argmax(1) == y_fit[batch]).sum().item()
        model.eval()
        with torch.inference_mode():
            logits = model(x_tune)
            tune_loss = criterion(logits, y_tune).item()
            tune_accuracy = (logits.argmax(1) == y_tune).float().mean().item()
        history.append({"epoch": epoch, "fit_loss": total_loss/len(x_fit), "fit_accuracy": total_correct/len(x_fit),
                        "tune_loss": tune_loss, "tune_accuracy": tune_accuracy,
                        "last_batch_gradient_norm_before_clip": gradient_norm})
        print(f"Head {widths}: epoch {epoch}, fit loss {total_loss/len(x_fit):.4f}, tune loss {tune_loss:.4f}, accuracy {tune_accuracy:.4f}", flush=True)
        if tune_loss < best:
            best, best_epoch, best_state, stale = tune_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= (patience or CONFIG["patience"]):
            break
    model.load_state_dict(best_state)
    return model.eval(), history, best_epoch


def train_neural(root: Path) -> dict:
    configure()
    fit, tune, audit = verify_development_data(root)
    directory = root / "data" / "embeddings"
    cached = read_json(directory / "manifest.json")
    if cached["revision"] != REVISION or cached["config"] != CONFIG:
        raise ValueError("Embedding model/configuration mismatch; run embed.")
    arrays = []
    for name, rows in (("fit", fit), ("tune", tune)):
        if cached["input_sha256"][name] != audit["prepared_sha256"][name + ".jsonl"]:
            raise ValueError("Embedding data mismatch.")
        path = directory / (name + ".npy")
        array = validate_feature_cache(path, len(rows), cached["sha256"][name])
        arrays.append(array)
    classes = sorted({r["intent"] for r in fit})
    labels = {name: i for i, name in enumerate(classes)}
    x_fit, x_tune = (torch.from_numpy(array) for array in arrays)
    y_fit, y_tune = (torch.tensor([labels[r["intent"]] for r in rows]) for rows in (fit, tune))
    run_id = "neural_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir, report_dir = root / "artifacts" / run_id, root / "reports" / "runs" / run_id
    artifact_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    metadata = {"run_id": run_id, "status": "started", "config": CONFIG, "classes": classes,
                "model_id": MODEL_ID, "revision": REVISION, "encoder_parameters": cached["encoder_parameters"],
                "encoder_trainable_parameters": 0, "evaluation_split": "tune", "test_evaluated": False,
                "automatic_routing_enabled": False, "probabilities_calibrated": False,
                "embedding_manifest_sha256": sha256_file(directory / "manifest.json"),
                "encoder_manifest_sha256": cached["encoder_manifest_sha256"],
                "manifest_sha256": audit["manifest_sha256"],
                "source_hashes": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
                "environment": {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
                                "transformers": version("transformers"), "numpy": np.__version__, "sklearn": version("scikit-learn")},
                "uv_lock_sha256": sha256_file(root / "uv.lock"),
                "fit_samples": len(fit), "tune_samples": len(tune)}
    write_json(report_dir / "run.json", metadata)
    started = time.perf_counter()
    metrics, histories, checkpoints = {}, {}, {}
    print("Training linear control on the same MPNet features...", flush=True)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        linear = LogisticRegression(C=4., max_iter=500, random_state=CONFIG["seed"])
        linear.fit(arrays[0], [r["intent"] for r in fit])
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            raise ValueError("Embedding linear control failed convergence.")
        probabilities = linear.predict_proba(arrays[1])
    metrics["B2_MPNet_linear"] = classification_metrics([r["intent"] for r in tune], linear.classes_[probabilities.argmax(1)], probabilities, linear.classes_)
    joblib.dump(linear, artifact_dir / "linear.joblib", compress=3)
    metadata["linear_sha256"] = sha256_file(artifact_dir / "linear.joblib")
    restored_linear = joblib.load(artifact_dir / "linear.joblib")
    if not np.allclose(restored_linear.predict_proba(arrays[1][:20]), probabilities[:20], atol=1e-7, rtol=0):
        raise ValueError("Linear checkpoint failed reload parity.")
    for name, widths in (("N1_small", [768, 128, 150]), ("N2_large", [768, 1024, 512, 150])):
        model, history, best_epoch = fit_head(x_fit, y_fit, x_tune, y_tune, widths)
        with torch.inference_mode():
            probabilities = model(x_tune).softmax(1).numpy()
        predicted = np.asarray(classes)[probabilities.argmax(1)]
        metrics[name] = classification_metrics([r["intent"] for r in tune], predicted, probabilities, classes)
        path = artifact_dir / (name + ".pt")
        torch.save(model.state_dict(), path)
        reloaded = make_head(widths).eval()
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        with torch.inference_mode():
            error = float((model(x_tune[:20])-reloaded(x_tune[:20])).abs().max())
        if error > 1e-6:
            raise ValueError("Neural checkpoint failed reload parity.")
        checkpoints[name] = {"widths": widths, "trainable_parameters": sum(p.numel() for p in model.parameters()),
                             "best_epoch": best_epoch, "epochs_run": len(history), "sha256": sha256_file(path),
                             "reload_max_logit_error": error}
        histories[name] = history
        write_json(report_dir / (name + "_errors.json"), [
            {"example_id": r["example_id"], "text": r["text"], "true_intent": r["intent"],
             "predicted_intent": str(predicted[i]), "reviewed_by_human": False}
            for i, r in enumerate(tune) if r["intent"] != predicted[i]])
    metadata.update(status="completed", checkpoints=checkpoints, fit_seconds=time.perf_counter()-started,
                    process_peak_working_set_bytes=psutil.Process().memory_info().peak_wset)
    write_json(artifact_dir / "metadata.json", metadata)
    write_json(report_dir / "run.json", metadata)
    write_json(report_dir / "metrics.json", metrics)
    write_json(report_dir / "history.json", histories)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        axes[0].plot([h["epoch"] for h in history], [h["fit_loss"] for h in history], label=name+" fit")
        axes[0].plot([h["epoch"] for h in history], [h["tune_loss"] for h in history], linestyle="--", label=name+" tune")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Learning curves (selected by tune loss)")
    axes[0].legend(fontsize=8)
    axes[1].bar(list(metrics), [m["macro_f1"] for m in metrics.values()], color=["#177e89", "#6779ab", "#b64f47"])
    axes[1].set(ylim=(0, 1), ylabel="Macro-F1", title="Same embeddings and development split")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "neural_comparison.png", dpi=160)
    plt.close(fig)
    table = "\n".join(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['log_loss']:.4f} |" for name, m in metrics.items())
    (report_dir / "summary.md").write_text(
        "# Local neural experiment\n\nDevelopment only; final test not evaluated. Frozen pretrained MPNet; only the heads were trained here.\n\n"
        "| Model | Accuracy | Macro-F1 | Log loss |\n| --- | ---: | ---: | ---: |\n"+table+
        "\n\nN1: 768 → 128 → 150. N2: 768 → 1024 → 512 → 150. Same optimizer and epoch budget; widths/depth change together. "
        "One seed, fixed configurations; this is not evidence of statistical superiority or production quality. "
        "Calibration, OOS handling and official validation are still pending. The larger head is available for inspection, not automatically promoted.\n\n"
        "![Learning curves and comparison](neural_comparison.png)\n\nSee run.json for hashes, sizes and timing; history.json for every epoch; *_errors.json for unreviewed mistakes.\n",
        encoding="utf-8")
    write_json(root / "artifacts" / "neural_latest.json", {"run_id": run_id})
    return {"run_id": run_id, "metrics": metrics, "checkpoints": checkpoints, "report": str(report_dir / "summary.md")}


def predict_neural(root: Path, text: str) -> dict:
    text = validate_text(text)
    run_id = read_json(root / "artifacts" / "neural_latest.json")["run_id"]
    directory = (root / "artifacts" / run_id).resolve()
    if not directory.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("Invalid artifact path.")
    metadata = read_json(directory / "metadata.json")
    if metadata["status"] != "completed" or metadata["config"] != CONFIG:
        raise ValueError("Neural artifact is incomplete or preprocessing/configuration changed.")
    checkpoint = metadata["checkpoints"]["N2_large"]
    path = directory / "N2_large.pt"
    if sha256_file(path) != checkpoint["sha256"] or metadata["revision"] != REVISION:
        raise ValueError("Neural checkpoint/revision mismatch.")
    tokenizer, encoder, _ = load_encoder(root)
    token_count = len(tokenizer(text, truncation=False)["input_ids"])
    if sha256_file(root / "models" / "mpnet" / "manifest.json") != metadata["encoder_manifest_sha256"]:
        raise ValueError("Encoder differs from training.")
    head = make_head(checkpoint["widths"]).eval()
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        probabilities = head(torch.from_numpy(encode_batch(tokenizer, encoder, [text]))).softmax(1)[0]
    indices = probabilities.argsort(descending=True)[:3].tolist()
    return {"model": "N2_large", "run_id": run_id,
            "calibrated": False, "input_tokens": token_count,
            "truncated": token_count > CONFIG["max_length"],
            "candidates": [{"intent": metadata["classes"][i], "raw_score": float(probabilities[i])} for i in indices],
            "review_required": True, "reason": "Calibration and out-of-scope handling not validated."}
