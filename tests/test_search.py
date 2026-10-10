import hashlib
from contextlib import closing
from pathlib import Path
import sqlite3

import pytest

from intentdesk.common import read_json, write_json
from intentdesk.search import INDEX_CONFIG, Searcher, build_index, match_query, ranking_metrics


def fixture_corpus(tmp_path, monkeypatch):
    directory = tmp_path/"data/retrieval/msmarco-v1"
    directory.mkdir(parents=True)
    content = b"0\tThe cat sleeps on a mat.\n1\tPython programming language tutorial.\n2\tDogs like running.\n3\tPython programming for beginners.\n"
    collection = directory/"collection.tsv"
    collection.write_bytes(content)
    corpus = {"collection": {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "passages": 4},
              "terms": "Test fixture only"}
    monkeypatch.setattr("intentdesk.search.prepare_retrieval", lambda root: corpus)
    monkeypatch.setitem(INDEX_CONFIG, "batch_rows", 2)
    return directory, collection, content


def test_index_search_ranking_and_verified_reuse(tmp_path, monkeypatch):
    directory, collection, content = fixture_corpus(tmp_path, monkeypatch)
    result = build_index(tmp_path)
    assert result["passages"] == 4 and result["integrity_check"] == "passed"
    searcher = Searcher(tmp_path, verify_hash=True)
    found = searcher.search("Python programming")
    assert {r["passage_id"] for r in found["results"]} == {1, 3}
    assert all(r["bm25_score"] > 0 for r in found["results"])
    assert not found["answer_generated"]
    assert build_index(tmp_path)["reused"]


def test_interrupted_index_resumes_from_committed_batch_without_duplicate_rows(tmp_path, monkeypatch):
    directory, collection, content = fixture_corpus(tmp_path, monkeypatch)
    collection.write_bytes(content.replace(b"2\tDogs", b"7\tDogs"))
    with pytest.raises(ValueError, match="ID order"):
        build_index(tmp_path)
    with closing(sqlite3.connect(directory/"search.building.sqlite3")) as db:
        assert db.execute("SELECT count(*) FROM passages").fetchone()[0] == 2
    collection.write_bytes(content)
    assert build_index(tmp_path)["passages"] == 4
    with closing(sqlite3.connect(directory/"search.sqlite3")) as db:
        assert db.execute("SELECT count(*) FROM passages").fetchone()[0] == 4


def test_search_tampered_index_is_rejected(tmp_path, monkeypatch):
    directory, collection, content = fixture_corpus(tmp_path, monkeypatch)
    build_index(tmp_path)
    index = directory/"search.sqlite3"
    with index.open("ab") as output:
        output.write(b"tampering")
    with pytest.raises(ValueError, match="incompatible"):
        Searcher(tmp_path)


@pytest.mark.parametrize("text", [None, "", "  ", "x"*1001])
def test_query_validation(text):
    with pytest.raises(ValueError):
        match_query(text)


def test_query_syntax_is_literal_and_bounded():
    assert match_query('python OR text:password* "secret"') == '"python" OR "text" OR "password" OR "secret"'
    assert match_query("the and or") == '"the" OR "and" OR "or"'
    assert match_query("!!!") == ""
    assert len(match_query(" ".join(f"term{i}" for i in range(100))).split(" OR ")) == 12


def test_ranking_metrics_handle_multiple_relevant_documents_and_no_hits():
    assert ranking_metrics([4, 7, 2], {7, 2}) == {"reciprocal_rank": .5, "recall": 1., "hit": 1.}
    assert ranking_metrics([], {7}) == {"reciprocal_rank": 0., "recall": 0., "hit": 0.}
    with pytest.raises(ValueError):
        ranking_metrics([1, 1], {1})


def test_query_timeout_is_explicit_and_not_partial_results(tmp_path, monkeypatch):
    fixture_corpus(tmp_path, monkeypatch)
    build_index(tmp_path)
    searcher = Searcher(tmp_path)
    assert searcher.search("???")["results"] == []
    with pytest.raises(ValueError):
        searcher.search("python", limit=100)
    original = sqlite3.connect
    class CheckEveryStep(sqlite3.Connection):
        def set_progress_handler(self, callback, instructions):
            super().set_progress_handler(callback, 1)
    monkeypatch.setattr("intentdesk.search.sqlite3.connect", lambda *a, **kw: original(*a, factory=CheckEveryStep, **kw))
    with pytest.raises(TimeoutError, match="specific terms"):
        searcher.search("python", timeout_seconds=1e-9)
