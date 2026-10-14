"""Pinned local MiniLM, bounded text windows and resumable feature extraction."""
import os
from pathlib import Path
import time

from huggingface_hub import snapshot_download
import numpy as np
import psutil
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.feedback_v3 import clean_markup


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CONFIG = {"dimension": 384, "length": 128, "batch_size": 8, "threads": 2,
          "views": ["prefix", "ends"], "device": "cpu"}


def configure():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)


def load_encoder(root):
    configure()
    directory = root/"models/feedback_minilm"
    manifest_path = directory/"manifest.json"
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION or not required.issubset(manifest["sha256"]):
            raise ValueError("MiniLM identity/manifest changed.")
        for name, digest in manifest["sha256"].items():
            if name != Path(name).name or sha256_file(directory/name) != digest:
                raise ValueError("Pinned MiniLM file changed.")
    else:
        print("Downloading pinned MiniLM weights/tokenizer for local CPU use...", flush=True)
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=directory,
            allow_patterns=list(required | {"vocab.txt", "special_tokens_map.json", "README.md", "LICENSE"}))
        if not all((directory/n).is_file() for n in required):
            raise ValueError("Incomplete pinned MiniLM download.")
        manifest = {"model_id": MODEL_ID, "revision": REVISION,
            "sha256": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}}
        write_json(manifest_path, manifest)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                        use_safetensors=True).eval()
    encoder.requires_grad_(False)
    if encoder.config.hidden_size != CONFIG["dimension"] or encoder.config.num_hidden_layers != 6:
        raise ValueError("Unexpected MiniLM architecture.")
    return tokenizer, encoder, manifest


def windows(tokenizer, texts):
    ids = tokenizer([clean_markup(t) for t in texts], add_special_tokens=False,
                    truncation=False, verbose=False)["input_ids"]
    budget = CONFIG["length"]-tokenizer.num_special_tokens_to_add(pair=False)
    first = [tokenizer.build_inputs_with_special_tokens(row[:budget]) for row in ids]
    last = [tokenizer.build_inputs_with_special_tokens(row[-budget:]) for row in ids]
    long = np.array([len(row)>budget for row in ids], dtype=bool)
    return first, last, long


def pooled(encoder, inputs):
    hidden = encoder(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1), p=2, dim=1)


def embed_batch(tokenizer, encoder, texts):
    first, last, long = windows(tokenizer, texts)
    with torch.inference_mode():
        prefix = pooled(encoder, tokenizer.pad({"input_ids": first}, padding=True, return_tensors="pt"))
        ends = prefix.clone()
        selected = np.flatnonzero(long)
        if len(selected):
            tail = pooled(encoder, tokenizer.pad({"input_ids": [last[i] for i in selected]}, padding=True, return_tensors="pt"))
            ends[selected] = nn.functional.normalize(prefix[selected]+tail, p=2, dim=1)
    return {"prefix": prefix.numpy(), "ends": ends.numpy()}, int(long.sum())


def embed_rows(directory, name, rows, tokenizer, encoder, encoder_manifest):
    receipt_path = directory/(name+"_features.json")
    identity = {"rows_sha256": sha256_file(directory/(name+".jsonl")),
                "encoder": encoder_manifest, "config": CONFIG}
    receipt = {"identity": identity, "offset": 0, "long_reviews": 0}
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["identity"] != identity:
            raise ValueError("Feature source/config changed.")
        for view, digest in receipt["sha256"].items():
            if sha256_file(directory/(name+"_"+view+".npy")) != digest:
                raise ValueError("Feature checkpoint changed.")
    offset = receipt["offset"]
    if not isinstance(offset, int) or not 0<=offset<=len(rows):
        raise ValueError("Invalid feature progress offset.")
    arrays = {}
    for view in CONFIG["views"]:
        path = directory/(name+"_"+view+".npy")
        if offset and not path.exists():
            raise ValueError("Missing feature checkpoint; refuse false completion.")
        arrays[view] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
            dtype=np.float32, shape=(len(rows), CONFIG["dimension"]))
        if arrays[view].shape != (len(rows), CONFIG["dimension"]) or arrays[view].dtype != np.float32:
            raise ValueError("Unexpected feature shape or dtype.")
    started = time.perf_counter()
    for start in range(offset, len(rows), CONFIG["batch_size"]):
        batch = rows[start:start+CONFIG["batch_size"]]
        output, long = embed_batch(tokenizer, encoder, [r["text"] for r in batch])
        for view in arrays:
            arrays[view][start:start+len(batch)] = output[view]
        receipt["long_reviews"] += long
        receipt["offset"] = start+len(batch)
        if start//CONFIG["batch_size"]%100==0 or receipt["offset"]==len(rows):
            for array in arrays.values():
                array.flush()
            receipt["sha256"] = {v: sha256_file(directory/(name+"_"+v+".npy")) for v in arrays}
            receipt["elapsed_latest_invocation_seconds"] = time.perf_counter()-started
            receipt["process_rss_bytes"] = psutil.Process().memory_info().rss
            write_json(receipt_path, receipt)
            print(f"MiniLM {name}: {receipt['offset']:,}/{len(rows):,}; prefix + first/last-window mean", flush=True)
    for array in arrays.values():
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-4):
            raise ValueError("Invalid normalized MiniLM features.")
    return arrays, receipt
