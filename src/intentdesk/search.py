"""Bounded-memory SQLite FTS5 retrieval over the separate MS MARCO corpus."""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tarfile
import time
from urllib.request import urlopen

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from intentdesk.common import read_json, sha256_file, write_json
from intentdesk.retrieval_data import prepare_retrieval, SOURCE_REVISION, URL

INDEX_CONFIG = {"version": 1, "tokenizer": "porter unicode61", "ranker": "SQLite FTS5 default BM25",
                "query": "OR of first 12 unique Unicode alphanumeric non-stop terms; fallback first 12 terms",
                "stop_words_sha256": hashlib.sha256("\n".join(sorted(ENGLISH_STOP_WORDS)).encode()).hexdigest(),
                "cache_kib": 65536, "batch_rows": 5000}
EVAL_CONFIG = {"version": 1, "sample_queries": 100, "selection": "lowest SHA256(intentdesk-search-v1:qid)",
               "cutoff": 10, "timeout_seconds": 10., "bootstrap_seed": 20261010,
               "use": "exploratory official development benchmark, not a final test"}
BASE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"


def match_query(text: str) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > 1000:
        raise ValueError("Search requires 1–1,000 characters.")
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)))
    selected = [t for t in terms if t not in ENGLISH_STOP_WORDS] or terms
    # Quoted single tokens cannot inject FTS operators, prefixes, columns or SQL.
    return " OR ".join('"'+t+'"' for t in selected[:12])


def build_index(root: Path) -> dict:
    corpus = prepare_retrieval(root)
    directory = root/"data/retrieval/msmarco-v1"
    target, partial = directory/"search.sqlite3", directory/"search.building.sqlite3"
    manifest_path = directory/"search_manifest.json"
    identity = {"config": INDEX_CONFIG, "collection_sha256": corpus["collection"]["sha256"],
                "collection_bytes": corpus["collection"]["bytes"], "expected_passages": corpus["collection"]["passages"]}
    if target.exists():
        saved = read_json(manifest_path)
        if saved["identity"] != identity or sha256_file(target) != saved["index_sha256"]:
            raise ValueError("Existing retrieval index changed; preserve it for investigation.")
        return {**saved, "reused": True}
    if shutil.disk_usage(directory).free < 12*1024**3:
        raise ValueError("Allow 12 GiB free for the local index and SQLite merge workspace.")
    start = time.perf_counter()
    with closing(sqlite3.connect(partial)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute(f"PRAGMA cache_size=-{INDEX_CONFIG['cache_kib']}")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE IF NOT EXISTS build_state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
        row = db.execute("SELECT value FROM build_state WHERE id=1").fetchone()
        if row:
            state = json.loads(row[0])
            if state["identity"] != identity:
                raise ValueError("Partial index belongs to a different corpus/configuration.")
        else:
            db.execute("CREATE VIRTUAL TABLE passages USING fts5(text, tokenize='porter unicode61')")
            state = {"identity": identity, "rows": 0, "byte_offset": 0, "build_seconds": 0.}
            db.execute("INSERT INTO build_state VALUES(1,?)", (json.dumps(state),))
            db.commit()
        print(f"Indexing from passage {state['rows']:,}; 64 MiB SQLite cache; disk-backed merges.", flush=True)
        with (directory/"collection.tsv").open("rb") as source:
            source.seek(state["byte_offset"])
            last_report = time.monotonic()
            previous_seconds = state["build_seconds"]
            while True:
                batch = []
                for _ in range(INDEX_CONFIG["batch_rows"]):
                    line = source.readline()
                    if not line:
                        break
                    identifier, text = line.rstrip(b"\r\n").split(b"\t", 1)
                    expected = state["rows"]+len(batch)
                    if int(identifier) != expected or not text:
                        raise ValueError("Corpus ID order or content changed while indexing.")
                    batch.append((expected+1, text.decode("utf-8")))
                if not batch:
                    break
                with db:
                    db.executemany("INSERT INTO passages(rowid,text) VALUES(?,?)", batch)
                    state.update(rows=state["rows"]+len(batch), byte_offset=source.tell(),
                                 build_seconds=previous_seconds+time.perf_counter()-start)
                    db.execute("UPDATE build_state SET value=? WHERE id=1", (json.dumps(state),))
                if time.monotonic()-last_report >= 15:
                    print(f"Indexed {state['rows']:,}/{identity['expected_passages']:,} passages", flush=True)
                    last_report = time.monotonic()
        if state["rows"] != identity["expected_passages"] or state["byte_offset"] != identity["collection_bytes"]:
            raise ValueError("Index is incomplete.")
        print("Checking the completed full-text index...", flush=True)
        db.execute("INSERT INTO passages(passages) VALUES('integrity-check')")
        db.commit()
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite index integrity check failed.")
        if db.execute("SELECT count(*) FROM passages").fetchone()[0] != identity["expected_passages"]:
            raise ValueError("Indexed row count mismatch.")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    partial.replace(target)
    result = {"status": "completed", "identity": identity, "passages": state["rows"],
              "index_bytes": target.stat().st_size, "index_sha256": sha256_file(target),
              "sqlite_version": sqlite3.sqlite_version, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "build_seconds": previous_seconds+time.perf_counter()-start, "integrity_check": "passed",
              "terms": corpus["terms"], "used_for_classifier_training": False,
              "retrieval_quality_evaluated": False}
    write_json(manifest_path, result)
    write_json(root/"reports/retrieval_index.json", result)
    return result


class Searcher:
    def __init__(self, root: Path, verify_hash: bool = False):
        self.path = root/"data/retrieval/msmarco-v1/search.sqlite3"
        self.manifest = read_json(self.path.with_name("search_manifest.json"))
        if (self.manifest["status"] != "completed" or self.manifest["identity"]["config"] != INDEX_CONFIG
                or not self.path.is_file() or self.path.stat().st_size != self.manifest["index_bytes"]):
            raise ValueError("Missing, incomplete or incompatible search index.")
        if verify_hash and sha256_file(self.path) != self.manifest["index_sha256"]:
            raise ValueError("Search index checksum mismatch.")

    def search(self, text: str, limit: int = 5, timeout_seconds: float = 2.) -> dict:
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("Choose 1–10 search results.")
        query = match_query(text)
        start = time.perf_counter()
        if not query:
            return {"results": [], "elapsed_ms": 0., "reason": "No searchable words in this query."}
        with closing(sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro", uri=True)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA cache_size=-32768")
            expired = False
            def check_budget():
                nonlocal expired
                expired = time.perf_counter()-start > timeout_seconds
                return int(expired)
            db.set_progress_handler(check_budget, 1000)
            try:
                rows = db.execute("SELECT rowid,text,rank FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT ?",
                                  (query, limit)).fetchall()
            except sqlite3.OperationalError as error:
                if expired or "interrupted" in str(error):
                    raise TimeoutError("Search took too long. Try more specific terms.") from None
                raise
        return {"results": [{"passage_id": row[0]-1, "text": row[1], "bm25_score": -row[2]} for row in rows],
                "elapsed_ms": (time.perf_counter()-start)*1000, "source": "MS MARCO v1 passages",
                "score_type": "lexical BM25 ranking score, not confidence or factual verification",
                "terms": self.manifest["terms"], "answer_generated": False}


def download_benchmark(root: Path) -> dict:
    directory = root/"data/retrieval/msmarco-v1/benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/"manifest.json"
    if manifest.exists():
        result = read_json(manifest)
        if result["source_revision"] != SOURCE_REVISION or any(
                sha256_file(directory/name) != info["sha256"] for name, info in result["files"].items()):
            raise ValueError("Retrieval benchmark files changed.")
        return result
    registry = (directory.parent/"upstream_Datasets.md").read_text(encoding="utf-8")
    files = {}
    for name in ("queries.tar.gz", "qrels.dev.tsv"):
        url = BASE_URL+name
        if url not in registry:
            raise ValueError("Benchmark URL missing from the pinned official registry.")
        target = directory/name
        temporary = target.with_suffix(target.suffix+".part")
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024*1024):
                total += len(chunk)
                if total > 100*1024**2:
                    raise ValueError("Unexpectedly large benchmark download.")
                output.write(chunk)
            if total != int(response.headers.get("Content-Length", -1)):
                raise ValueError("Benchmark download size mismatch.")
            etag = response.headers.get("ETag")
        temporary.replace(target)
        files[name] = {"url": url, "bytes": total, "sha256": sha256_file(target), "etag": etag}
    with tarfile.open(directory/"queries.tar.gz", "r:gz") as archive:
        members = [m for m in archive if m.name in ("queries.dev.tsv", "./queries.dev.tsv") and m.isfile()]
        if len(members) != 1 or members[0].size > 30*1024**2:
            raise ValueError("Unexpected official development query archive.")
        source = archive.extractfile(members[0])
        with source, (directory/"queries.dev.tsv").open("wb") as output:
            shutil.copyfileobj(source, output)
    files["queries.dev.tsv"] = {"member_of": "queries.tar.gz", "bytes": (directory/"queries.dev.tsv").stat().st_size,
                                "sha256": sha256_file(directory/"queries.dev.tsv")}
    result = {"source_revision": SOURCE_REVISION, "files": files,
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "split": "official development; no leaderboard test evaluation"}
    write_json(manifest, result)
    return result


def ranking_metrics(results: list[int], relevant: set[int], cutoff: int = 10) -> dict:
    if not relevant or len(results) != len(set(results)):
        raise ValueError("Need relevance judgments and unique ranked IDs.")
    hits = [i+1 for i, identifier in enumerate(results[:cutoff]) if identifier in relevant]
    return {"reciprocal_rank": 1./hits[0] if hits else 0., "recall": len(hits)/len(relevant),
            "hit": float(bool(hits))}


def evaluate_search(root: Path) -> dict:
    directory = root/"data/retrieval/msmarco-v1/benchmark"
    benchmark = download_benchmark(root)
    searcher = Searcher(root, verify_hash=True)
    judgments = {}
    for line in (directory/"qrels.dev.tsv").read_text(encoding="utf-8").splitlines():
        qid, iteration, pid, label = line.split()
        if iteration != "0" or int(label) != 1 or not 0 <= int(pid) < searcher.manifest["passages"]:
            raise ValueError("Unexpected MS MARCO relevance record.")
        judgments.setdefault(qid, set()).add(int(pid))
    query_ids = sorted(judgments, key=lambda q: hashlib.sha256(("intentdesk-search-v1:"+q).encode()).hexdigest())[:EVAL_CONFIG["sample_queries"]]
    query_id_set = set(query_ids)
    queries = {}
    with (directory/"queries.dev.tsv").open(encoding="utf-8") as source:
        for line in source:
            qid, text = line.rstrip("\n").split("\t", 1)
            if qid in query_id_set:
                if qid in queries:
                    raise ValueError("Duplicate benchmark query ID.")
                queries[qid] = text
    if set(queries) != query_id_set or len(query_ids) != EVAL_CONFIG["sample_queries"]:
        raise ValueError("Missing selected development queries.")
    protocol = {"config": EVAL_CONFIG, "index_config": INDEX_CONFIG, "query_ids": query_ids,
                "benchmark": benchmark, "index_sha256": searcher.manifest["index_sha256"]}
    report = root/"reports/retrieval_evaluation.json"
    freeze = directory/"evaluation_protocol.json"
    if freeze.exists() and read_json(freeze) != protocol:
        raise ValueError("Retrieval protocol changed. Preserve the original experiment before designing a new comparison.")
    if report.exists():
        existing = read_json(report)
        if existing["protocol"] != protocol:
            raise ValueError("Existing retrieval result belongs to a different protocol.")
        return {**existing, "reused": True}
    write_json(freeze, protocol)
    rows, timeouts, times = [], 0, []
    for i, qid in enumerate(query_ids):
        start = time.perf_counter()
        try:
            found = searcher.search(queries[qid], limit=10, timeout_seconds=EVAL_CONFIG["timeout_seconds"])
            pids = [r["passage_id"] for r in found["results"]]
            timeout = False
        except TimeoutError:
            pids, timeout = [], True
            timeouts += 1
        elapsed = (time.perf_counter()-start)*1000
        times.append(elapsed)
        rows.append({"query_id": qid, "ranked_passage_ids": pids, "timed_out": timeout,
                     "elapsed_ms": elapsed, **ranking_metrics(pids, judgments[qid])})
        if (i+1) % 10 == 0:
            print(f"Evaluated {i+1}/{len(query_ids)} fixed development queries", flush=True)
    rng = np.random.default_rng(EVAL_CONFIG["bootstrap_seed"])
    rr = np.array([r["reciprocal_rank"] for r in rows])
    boots = np.mean(rr[rng.integers(0, len(rr), size=(2000, len(rr)))], axis=1)
    result = {"protocol": protocol, "query_count": len(rows), "full_corpus_passages": searcher.manifest["passages"],
              "mrr_at_10": float(rr.mean()), "recall_at_10": float(np.mean([r["recall"] for r in rows])),
              "hit_rate_at_10": float(np.mean([r["hit"] for r in rows])),
              "mrr_bootstrap_95_percent": np.quantile(boots, [.025,.975]).tolist(),
              "timeouts_counted_as_zero": timeouts, "median_ms": float(np.median(times)),
              "p95_ms": float(np.quantile(times, .95)), "rows": rows,
              "limits": "100-query development sample; incomplete relevance judgments; lexical ranking only; not final/production evidence",
              "classifier_test_evaluated": False, "answer_generation_evaluated": False}
    write_json(report, result)
    return result
