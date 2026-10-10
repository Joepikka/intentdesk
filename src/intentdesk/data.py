"""Pinned data acquisition, validation, and leakage-aware development splits."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from urllib.request import Request, urlopen

from intentdesk.common import read_json, sha256_file, write_json

REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
BASE_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}/"
DATA_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
EXPECTED_COUNTS = {
    "train": 15000, "val": 3000, "test": 4500,
    "oos_train": 100, "oos_val": 100, "oos_test": 1000,
}
SEED = 20261010


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fingerprint(text: str) -> str:
    """Conservative grouping: case/spacing/punctuation variants stay together."""
    key = re.sub(r"[^\w\s]", "", normalized_text(text))
    key = " ".join(key.split())
    # Keep punctuation-only requests distinct instead of collapsing to empty.
    return hashlib.sha256((key or normalized_text(text)).encode()).hexdigest()


def build_records(payload: dict, domains: dict) -> list[dict]:
    """Validate schema and labels without printing held-out examples."""
    if not isinstance(payload, dict) or set(payload) != set(EXPECTED_COUNTS):
        raise ValueError("Dataset must contain exactly the six official split keys.")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("Domain mapping must be a non-empty object.")
    intent_domains = {}
    for domain, intents in domains.items():
        if not isinstance(domain, str) or not isinstance(intents, list):
            raise ValueError("Invalid domain mapping.")
        for intent in intents:
            if not isinstance(intent, str) or intent in intent_domains:
                raise ValueError("Duplicate or invalid intent in domain mapping.")
            intent_domains[intent] = domain
    records = []
    for source in EXPECTED_COUNTS:
        if not isinstance(payload[source], list):
            raise ValueError(f"Split {source} must be a list.")
        for index, pair in enumerate(payload[source]):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid record at {source}:{index}.")
            text, intent = pair
            if not isinstance(text, str) or not text.strip() or not isinstance(intent, str):
                raise ValueError(f"Blank text or invalid types at {source}:{index}.")
            is_oos = source.startswith("oos_")
            if (is_oos and intent != "oos") or (not is_oos and intent not in intent_domains):
                raise ValueError(f"Unexpected label at {source}:{index}.")
            records.append({
                "example_id": f"{source}:{index:05d}", "text": text,
                "intent": intent, "domain": "out_of_scope" if is_oos else intent_domains[intent],
                "is_oos": is_oos, "source_split": source, "working_split": "pending",
                "text_hash": hashlib.sha256(normalized_text(text).encode()).hexdigest(),
                "duplicate_group": fingerprint(text), "exclusion_reason": None,
            })
    return records


def validate_official_contract(records: list[dict]) -> None:
    counts = Counter(r["source_split"] for r in records)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected official split counts: {dict(counts)}")
    for source, per_class in [("train", 100), ("val", 20), ("test", 30)]:
        labels = Counter(r["intent"] for r in records if r["source_split"] == source)
        if len(labels) != 150 or set(labels.values()) != {per_class}:
            raise ValueError(f"Split {source} violates the 150-intent contract.")


def assign_partitions(records: list[dict], seed: int = SEED) -> dict:
    """Protect test first, validation second, then deduplicate/split training.

    Mechanical fingerprints inspect overlap only. Held-out contents never
    influence model vocabulary, weights, examples, or hyperparameters.
    """
    test_groups = {r["duplicate_group"] for r in records if r["source_split"] in {"test", "oos_test"}}
    val_groups = {r["duplicate_group"] for r in records if r["source_split"] in {"val", "oos_val"}}
    train_groups = defaultdict(list)
    for row in records:
        source = row["source_split"]
        if source in {"test", "oos_test"}:
            row["working_split"] = "locked_test"
        elif source in {"val", "oos_val"}:
            if row["duplicate_group"] in test_groups:
                row["working_split"] = "excluded"
                row["exclusion_reason"] = "validation_overlap_with_locked_test"
            else:
                row["working_split"] = "validation"
        elif source == "oos_train":
            row["working_split"] = "oos_development"
        else:
            train_groups[row["duplicate_group"]].append(row)

    candidates = defaultdict(list)
    for group, members in sorted(train_groups.items()):
        reason = None
        if group in test_groups or group in val_groups:
            reason = "training_overlap_with_protected_holdout"
        elif len({r["intent"] for r in members}) > 1:
            reason = "conflicting_training_labels"
        if reason:
            for row in members:
                row["working_split"] = "excluded"
                row["exclusion_reason"] = reason
            continue
        # First source row is a reproducible representative of this group.
        members.sort(key=lambda r: r["example_id"])
        candidates[members[0]["intent"]].append(members[0])
        for row in members[1:]:
            row["working_split"] = "excluded"
            row["exclusion_reason"] = "duplicate_training_variant"

    rng = random.Random(seed)
    for label, members in sorted(candidates.items()):
        members.sort(key=lambda r: r["example_id"])
        if len(members) < 10:
            raise ValueError(f"Too few independent training records for {label}.")
        rng.shuffle(members)
        held = max(1, round(len(members) * 0.1))
        for index, row in enumerate(members):
            row["working_split"] = "calibrate" if index < held else "tune" if index < held * 2 else "fit"

    active_groups = defaultdict(set)
    for row in records:
        if row["working_split"] in {"fit", "tune", "calibrate", "validation", "locked_test"}:
            active_groups[row["duplicate_group"]].add(row["working_split"])
    if any(len(splits) > 1 for splits in active_groups.values()):
        raise ValueError("A duplicate group crosses active partitions.")
    return {
        "seed": seed,
        "counts": dict(sorted(Counter(r["working_split"] for r in records).items())),
        "exclusions": dict(sorted(Counter(r["exclusion_reason"] for r in records if r["exclusion_reason"]).items())),
        "group_overlap_between_active_partitions": 0,
        "policy": "protect test; exclude val overlap; protect val; deduplicate train; seeded 80/10/10 per intent",
        "fingerprint": "Unicode NFKC, casefold, collapse whitespace, remove punctuation; SHA256",
        "limitations": [
            "Fingerprint grouping detects formatting variants, not all semantic near-duplicates.",
            "It may conservatively group requests whose punctuation changes meaning.",
            "OOS training examples are unused; no automatic routing policy is trained at this milestone.",
        ],
    }


def download_sources(root: Path) -> dict:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    existing = raw / "provenance.json"
    provenance = read_json(existing) if existing.exists() else {
        "repository": "https://github.com/clinc/oos-eval", "revision": REVISION,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(), "files": {},
    }
    if provenance.get("revision") != REVISION:
        raise ValueError("Existing raw provenance has a different revision.")
    for relative in ["data/data_full.json", "data/domains.json", "LICENSE", "README.md"]:
        path = raw / Path(relative).name
        url = BASE_URL + relative
        if not path.exists():
            request = Request(url, headers={"User-Agent": "IntentDesk-learning-project/0.1"})
            with urlopen(request, timeout=60) as response:
                content = response.read()
            if relative.endswith("data_full.json") and hashlib.sha256(content).hexdigest() != DATA_SHA256:
                raise ValueError("Downloaded dataset checksum does not match the pinned source.")
            path.write_bytes(content)
        digest = sha256_file(path)
        previous = provenance["files"].get(path.name)
        if previous and previous["sha256"] != digest:
            raise ValueError(f"Cached source changed: {path.name}")
        provenance["files"][path.name] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
    if sha256_file(raw / "data_full.json") != DATA_SHA256:
        raise ValueError("Pinned dataset checksum mismatch.")
    write_json(existing, provenance)
    # Preserve the actual upstream legal text; this is separate from project code.
    license_copy = root / "data" / "CLINC150_LICENSE.txt"
    license_copy.write_bytes((raw / "LICENSE").read_bytes())
    return provenance


def prepare(root: Path) -> dict:
    provenance = download_sources(root)
    raw = root / "data" / "raw"
    domains = read_json(raw / "domains.json")
    records = build_records(read_json(raw / "data_full.json"), domains)
    validate_official_contract(records)
    audit = assign_partitions(records)
    audit["official_counts"] = EXPECTED_COUNTS
    audit["dataset_revision"] = REVISION
    audit["dataset_sha256"] = DATA_SHA256
    audit["domain_mapping_sha256"] = sha256_file(raw / "domains.json")
    audit["source_provenance"] = provenance
    audit["test_evaluated"] = False
    output = root / "data" / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    for split in ["fit", "tune", "calibrate", "validation", "oos_development", "locked_test", "excluded"]:
        selected = [r for r in records if r["working_split"] == split]
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8",
        )
    manifest_path = output / "manifest.csv"
    fields = [k for k in records[0] if k != "text"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in records)
    audit["manifest_sha256"] = sha256_file(manifest_path)
    audit["prepared_sha256"] = {p.name: sha256_file(p) for p in sorted(output.glob("*.jsonl"))}
    write_json(output / "audit.json", audit)
    write_json(root / "reports" / "data_audit.json", audit)
    return audit

