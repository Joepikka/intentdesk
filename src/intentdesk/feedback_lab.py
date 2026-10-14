"""Versioned feedback experiments; no modifications to IntentDesk's frozen models."""

from datetime import datetime, timezone
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import time

import joblib
import numpy as np

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.feedback_data import DIRECTORY, load_partition
from intentdesk import feedback_models as classical
from intentdesk import feedback_transformer as neural


SOURCES = ("feedback_data.py","feedback_models.py","feedback_transformer.py","feedback_lab.py")


@contextmanager
def training_lock(root):
    """OS releases the lock even if a terminal closes; never run duplicate jobs."""
    path = root/"local/feedback_training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("A feedback training/evaluation job is already running. Use one terminal.") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def binding(root):
    return {"prepared_manifest_sha256": sha256_file(root/DIRECTORY/"manifest.json"),
        "uv_lock_sha256": sha256_file(root/"uv.lock"),
        "source_sha256": {name: sha256_file(root/"src/intentdesk"/name) for name in SOURCES},
        "classical_config": classical.CONFIG, "transformer_config": neural.CONFIG}


def directory_for(root):
    pointer = read_json(root/"artifacts/feedback_lab_latest.json")
    name = pointer["run_id"]
    if not re.fullmatch(r"feedback_lab_[0-9TZ]+",name):
        raise ValueError("Invalid feedback experiment directory.")
    directory = root/"artifacts"/name
    return directory


def start_extension(root: Path):
    """Copy an unexposed run before continuing it under the longer protocol."""
    previous = directory_for(root)
    identity = binding(root)
    old = read_json(previous/"protocol.json")
    if old == identity:
        return {"run_id": previous.name, "reused_extension": True}
    if (previous/"final_experiment.json").exists():
        raise ValueError("The previous feedback final test is consumed; preserve it and obtain untouched evaluation before extending.")
    for key in ("prepared_manifest_sha256", "uv_lock_sha256", "classical_config"):
        if old[key] != identity[key]:
            raise ValueError("Continuation requires unchanged data, dependency lock and classical configuration.")
    if any(old["source_sha256"][name] != identity["source_sha256"][name]
           for name in ("feedback_data.py", "feedback_models.py")):
        raise ValueError("Continuation cannot reuse classical artifacts after their data/model code changes.")
    mutable = {"epochs", "minimum_epochs", "patience", "selection"}
    if {k:v for k,v in old["transformer_config"].items() if k not in mutable} != {
            k:v for k,v in identity["transformer_config"].items() if k not in mutable}:
        raise ValueError("Continuation requires unchanged tokenizer, architecture and optimizer configuration.")
    for name in neural.CONFIG["architectures"]:
        checkpoint = previous/(name+"_resume.pt")
        if sha256_file(checkpoint) != read_json(checkpoint.with_suffix(".json"))["sha256"]:
            raise ValueError("Continuation checkpoint checksum changed.")
    name = "feedback_lab_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = root/"artifacts"/name
    target.mkdir(parents=True, exist_ok=False)
    skipped = {"protocol.json", "metadata.json", "final_experiment.json", "continuation.json"}
    skipped.update(n+suffix for n in neural.CONFIG["architectures"] for suffix in (".pt", "_development.json"))
    copied = {}
    for source in previous.iterdir():
        if source.is_file() and source.name not in skipped and not source.name.endswith(".tmp"):
            shutil.copy2(source, target/source.name)
            copied[source.name] = sha256_file(target/source.name)
    write_json(target/"protocol.json", identity)
    write_json(target/"continuation.json", {"previous_run": previous.name,
        "previous_protocol": old, "previous_protocol_sha256": sha256_file(previous/"protocol.json"),
        "copied_sha256": copied, "previous_final_test_exposed": False,
        "description": "Continue existing task checkpoints; prior run remains preserved. No final-test selection."})
    write_json(root/"artifacts/feedback_lab_latest.json", {"run_id": name})
    return {"run_id": name, "previous_run": previous.name, "reused_extension": False}


def train(root: Path):
    identity = binding(root)
    pointer = root/"artifacts/feedback_lab_latest.json"
    if pointer.exists():
        directory = directory_for(root)
        protocol = read_json(directory/"protocol.json")
        if protocol != identity:
            raise ValueError("Feedback source/data/config changed; preserve the current experiment before a new version.")
        if (directory/"metadata.json").exists():
            return checked_metadata(root)
    else:
        name = "feedback_lab_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = root/"artifacts"/name; directory.mkdir(parents=True,exist_ok=False)
        write_json(directory/"protocol.json",identity)
        write_json(pointer,{"run_id":name})
    fit, development = load_partition(root,"fit"), load_partition(root,"development")
    manifest = read_json(root/DIRECTORY/"manifest.json")
    if sha256_file(root/DIRECTORY/"audit.sqlite3") != manifest["audit_sha256"]:
        raise ValueError("Feedback full-fit audit database checksum changed.")
    available = {int(k): v for k,v in manifest["available_by_partition_and_rating"]["fit"].items()}
    started = time.perf_counter()
    metrics, regression, unsupervised, features = classical.train_classical(directory,fit,development,available)
    metrics["streaming_sgd"] = classical.train_streaming(root,directory,development,available)
    vocab_path = directory/"vocabulary.json"
    vocab_receipt = directory/"vocabulary_receipt.json"
    if vocab_path.exists():
        if sha256_file(vocab_path) != read_json(vocab_receipt)["sha256"]:
            raise ValueError("Transformer vocabulary checkpoint changed.")
        vocab = read_json(vocab_path)
    else:
        vocab = neural.vocabulary([r["text"] for r in fit])
        write_json(vocab_path,vocab)
        write_json(vocab_receipt,{"sha256":sha256_file(vocab_path)})
    for name in neural.CONFIG["architectures"]:
        receipt = directory/(name+"_development.json")
        if receipt.exists():
            cached = read_json(receipt)
            if sha256_file(directory/(name+".pt")) != cached["training"]["sha256"]:
                raise ValueError("Completed scratch model checksum changed.")
            metrics[name] = cached; continue
        model, training = neural.fit_candidate(directory,name,[r["text"] for r in fit],[r["rating"] for r in fit],
            [r["text"] for r in development],[r["rating"] for r in development],vocab)
        tokens, _ = neural.tokenize([r["text"] for r in development],vocab)
        p = neural.probabilities(model,tokens)
        metrics[name] = {**classical.classification([r["rating"] for r in development],p.argmax(1)+1),
                         "representation":"scratch_transformer","training":training,"fit_rows":len(fit)}
        write_json(receipt,metrics[name])
    selected = sorted(metrics,key=lambda name: (-metrics[name]["macro_f1"],name))[0]
    metadata = {"run_id": directory.name,"status":"trained","binding":identity,
        "data":manifest,"development_classification":metrics,"development_regression":regression,
        "unsupervised":unsupervised,"features":features,"selected_classifier":selected,
        "selected_by":"development macro-F1 only","test_exposed":False,"automatic_actions":False,
        "task":"Public product-review rating; not support intent or verified customer benefit",
        "continuation": read_json(directory/"continuation.json") if (directory/"continuation.json").exists() else None,
        "elapsed_this_invocation_seconds":time.perf_counter()-started,
        "artifact_sha256":{p.name:sha256_file(p) for p in directory.iterdir() if p.is_file() and p.name not in {"metadata.json"}}}
    write_json(directory/"metadata.json",metadata)
    write_json(pointer,{"run_id":directory.name,"metadata_sha256":sha256_file(directory/"metadata.json")})
    return metadata


def checked_metadata(root):
    directory = directory_for(root); pointer = read_json(root/"artifacts/feedback_lab_latest.json")
    if sha256_file(directory/"metadata.json") != pointer.get("metadata_sha256"):
        raise ValueError("Feedback metadata checksum changed or training is incomplete.")
    metadata = read_json(directory/"metadata.json")
    if metadata["binding"] != binding(root):
        raise ValueError("Feedback experiment source/data/config changed.")
    for name,digest in metadata["artifact_sha256"].items():
        if name != Path(name).name or sha256_file(directory/name) != digest:
            raise ValueError("Feedback model artifact checksum changed.")
    return metadata
