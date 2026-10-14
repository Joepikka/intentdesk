"""Pinned large review data, product-group splits and disk-backed duplicate audit."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
import unicodedata
from urllib.request import Request, urlopen

from intentdesk.common import read_json, read_records, sha256_file, write_json


REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
BASE = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/{REVISION}/raw/review_categories/"
FILES = {
    "Appliances.jsonl": {"bytes": 929451412, "sha256": "d9dd03aa9be0c8e6fea66d8ce99a7e5b359e272d6323186c1aa80f6d36af1dab"},
    "Amazon_Fashion.jsonl": {"bytes": 1051324731, "sha256": "e597b3ce6cbb2f9994a32e0eee6bb04fdae50725f2d49c89f6baef1a7b986c57"},
}
DIRECTORY = "data/feedback_lab"
PROTOCOL = {"version": "feedback_lab_v1", "seed": 20261013,
    "split": "SHA256(seed,parent_asin) modulo 100: fit<70, development<85, test otherwise",
    "dedup": "NFKC/casefold/alphanumeric fingerprints; exclude conflicting rating or cross-partition duplicates",
    "fit_per_rating": 20000, "development_limit": 10000, "test_limit": 20000,
    "minimum_chars": 20, "maximum_chars": 8000,
    "rating_shortcuts": "Remove explicit numeric/word star ratings and X out of 5 / X/5 declarations before fingerprinting and modeling",
    "selection": "Fit only on fit; choose classifier by development macro-F1; regression separate; freeze all artifacts before test",
    "features": "review title and body only; no rating, user ID, product ID or helpful votes in model features",
    "limitations": "Product-disjoint is not user-disjoint or temporal evaluation; heuristic rating cleanup cannot remove every possible shortcut; review labels are subjective, not support intents",
}


def clean_rating_declarations(text):
    text = re.sub(r"\b(?:[1-5]|one|two|three|four|five)\s*(?:[- ]?stars?\b|(?:out\s+of\s+|/\s*)5\b)", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def text_key(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(re.sub(r"[^\w\s]", "", normalized).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def product_split(product):
    if not isinstance(product, str) or not product:
        raise ValueError("A product group is required.")
    bucket = int(hashlib.sha256(f"{PROTOCOL['seed']}:{product}".encode()).hexdigest()[:8], 16) % 100
    return "fit" if bucket < 70 else "development" if bucket < 85 else "test"


def download(root: Path):
    raw = root/DIRECTORY/"raw"; raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 8_000_000_000:
        raise ValueError("Feedback data needs at least 8 GB free for sources/audit/model work.")
    for name, spec in FILES.items():
        path = raw/name
        if path.exists():
            if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
                raise ValueError("Existing review source checksum mismatch; preserve it before retrying.")
            continue
        partial = path.with_suffix(".jsonl.part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec["bytes"]:
            raise ValueError("Review partial download exceeds pinned size.")
        if offset < spec["bytes"]:
            headers = {"User-Agent": "IntentDesk-research/0.1", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(BASE+name+"?download=true", headers=headers), timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Review server did not honor the resume range.")
                reported = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(4*1024*1024):
                        stream.write(block); offset += len(block)
                        if offset > spec["bytes"]:
                            raise ValueError("Review source exceeded pinned size.")
                        if time.monotonic()-reported > 10:
                            print(f"{name}: {offset/1e6:.0f}/{spec['bytes']/1e6:.0f} MB", flush=True); reported = time.monotonic()
        if partial.stat().st_size != spec["bytes"] or sha256_file(partial) != spec["sha256"]:
            raise ValueError("Review download failed pinned size/SHA256 checks; partial retained.")
        partial.replace(path)
    write_json(raw/"provenance.json", {"repository": "McAuley-Lab/Amazon-Reviews-2023", "revision": REVISION,
        "files": FILES, "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://amazon-reviews-2023.github.io/", "purpose": "local research and education",
        "terms": "Dataset card does not supply an explicit blanket data license. No claim of MIT/commercial rights; original review rights retained; raw text not published."})
    return raw


def ingest(paths, database: Path):
    """Do not retain user identifiers or image URLs in the derived audit database."""
    masks = {"fit": 1, "development": 2, "test": 4}
    if database.exists():
        raise ValueError("Audit database already exists; preserve it before a new preparation.")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("CREATE TABLE reviews(hash TEXT PRIMARY KEY,text TEXT NOT NULL, rating INTEGER NOT NULL, product TEXT NOT NULL, split TEXT NOT NULL, ratings INTEGER NOT NULL, splits INTEGER NOT NULL, copies INTEGER NOT NULL) WITHOUT ROWID")
    counts = Counter(); batch = []
    sql = "INSERT INTO reviews VALUES(?,?,?,?,?,?,?,1) ON CONFLICT(hash) DO UPDATE SET ratings=ratings|excluded.ratings,splits=splits|excluded.splits,copies=copies+1"
    try:
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    counts["source_rows"] += 1
                    row = json.loads(line)
                    rating, product = row.get("rating"), row.get("parent_asin")
                    title, body = row.get("title"), row.get("text")
                    if (not isinstance(rating, (float, int)) or isinstance(rating, bool) or rating not in (1,2,3,4,5)
                            or not isinstance(product, str) or not product
                            or not isinstance(title, str) or not isinstance(body, str)):
                        counts["invalid_rows"] += 1; continue
                    text = clean_rating_declarations((title.strip()+". "+body.strip()).strip())
                    if not PROTOCOL["minimum_chars"] <= len(text) <= PROTOCOL["maximum_chars"]:
                        counts["length_exclusions"] += 1; continue
                    split = product_split(product)
                    batch.append((text_key(text), text, int(rating), product, split, 1 << (int(rating)-1), masks[split]))
                    if len(batch) >= 2000:
                        connection.executemany(sql, batch); connection.commit(); batch.clear()
                    if counts["source_rows"] % 100000 == 0:
                        print(f"Audited {counts['source_rows']:,} review rows on disk", flush=True)
        if batch:
            connection.executemany(sql, batch); connection.commit()
        connection.execute("CREATE INDEX eligible_split ON reviews(split,rating,hash) WHERE ratings IN(1,2,4,8,16) AND splits IN(1,2,4)")
        connection.commit()
        counts["distinct_fingerprints"] = connection.execute("SELECT count(*) FROM reviews").fetchone()[0]
        counts["conflicting_rating_groups"] = connection.execute("SELECT count(*) FROM reviews WHERE ratings NOT IN(1,2,4,8,16)").fetchone()[0]
        counts["cross_partition_duplicate_groups"] = connection.execute("SELECT count(*) FROM reviews WHERE splits NOT IN(1,2,4)").fetchone()[0]
        counts["duplicate_extra_copies"] = connection.execute("SELECT coalesce(sum(copies-1),0) FROM reviews").fetchone()[0]
        return dict(counts)
    finally:
        connection.close()


def eligible_rows(root, split, batch_size=2048, after_key=""):
    if split not in {"fit", "development", "test"}:
        raise ValueError("Unknown feedback partition.")
    with sqlite3.connect(root/DIRECTORY/"audit.sqlite3") as db:
        cursor = db.execute("SELECT hash,text,rating,product FROM reviews WHERE split=? AND hash>? AND ratings IN(1,2,4,8,16) AND splits IN(1,2,4) ORDER BY hash", (split, after_key))
        while batch := cursor.fetchmany(batch_size):
            yield [{"id": r[0], "text": r[1], "rating": r[2], "product": r[3]} for r in batch]


def select_rows(db, split, limit, rating=None):
    query = "SELECT hash,text,rating,product FROM reviews WHERE split=? AND ratings IN(1,2,4,8,16) AND splits IN(1,2,4)"
    args = [split]
    if rating is not None:
        query += " AND rating=?"; args.append(rating)
    query += " ORDER BY hash LIMIT ?"; args.append(limit)
    return [{"id": r[0], "text": r[1], "rating": r[2], "product": r[3]} for r in db.execute(query, args)]


def prepare(root: Path):
    directory = root/DIRECTORY; directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/"manifest.json"
    if manifest.exists():
        previous = read_json(manifest)
        if previous["protocol"] != PROTOCOL or previous["source_revision"] != REVISION:
            raise ValueError("Feedback preparation protocol changed; preserve previous experiment.")
        for name, digest in previous["prepared_sha256"].items():
            if sha256_file(directory/name) != digest:
                raise ValueError("Feedback prepared data changed.")
        return previous
    raw = download(root)
    counts = ingest([raw/name for name in FILES], directory/"audit.sqlite3")
    with sqlite3.connect(directory/"audit.sqlite3") as db:
        available = {split: dict(db.execute("SELECT rating,count(*) FROM reviews WHERE split=? AND ratings IN(1,2,4,8,16) AND splits IN(1,2,4) GROUP BY rating", (split,))) for split in ("fit", "development", "test")}
        selected = {"fit": sum((select_rows(db, "fit", PROTOCOL["fit_per_rating"], rating) for rating in range(1,6)), []),
                    "development": select_rows(db, "development", PROTOCOL["development_limit"]),
                    "test": select_rows(db, "test", PROTOCOL["test_limit"])}
    if len(selected["fit"]) != 5*PROTOCOL["fit_per_rating"]:
        raise ValueError("Insufficient class support for the declared balanced training sample.")
    keys = {name: {r["id"] for r in rows} for name, rows in selected.items()}
    products = {name: {r["product"] for r in rows} for name, rows in selected.items()}
    for a, b in (("fit", "development"), ("fit", "test"), ("development", "test")):
        if keys[a] & keys[b] or products[a] & products[b]:
            raise ValueError("Feedback partition overlap.")
    for split, rows in selected.items():
        rows.sort(key=lambda row: row["id"])
        (directory/(split+".jsonl")).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    result = {"protocol": PROTOCOL, "source_revision": REVISION, "source_files": FILES,
        "raw_bytes": sum(spec["bytes"] for spec in FILES.values()), "audit": counts,
        "available_by_partition_and_rating": available,
        "selected_counts": {name: len(rows) for name, rows in selected.items()},
        "selected_rating_counts": {name: dict(Counter(r["rating"] for r in rows)) for name, rows in selected.items()},
        "product_overlap": 0, "fingerprint_overlap": 0, "test_exposed": False,
        "prepared_sha256": {name+".jsonl": sha256_file(directory/(name+".jsonl")) for name in selected},
        "audit_sha256": sha256_file(directory/"audit.sqlite3")}
    write_json(manifest, result)
    return result


def load_partition(root, name):
    if name not in {"fit", "development", "test"}:
        raise ValueError("Unknown feedback partition.")
    manifest = read_json(root/DIRECTORY/"manifest.json")
    path = root/DIRECTORY/(name+".jsonl")
    if sha256_file(path) != manifest["prepared_sha256"][path.name]:
        raise ValueError("Feedback partition checksum changed.")
    rows = read_records(path)
    if not rows or any(r["rating"] not in range(1,6) or product_split(r["product"]) != name or text_key(r["text"]) != r["id"] for r in rows):
        raise ValueError("Feedback partition identity/label contract failed.")
    return rows
