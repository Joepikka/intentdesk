import hashlib
import io
import tarfile

import pytest

from intentdesk.retrieval_data import audit_collection, extract_collection


def make_archive(path, name, data):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))


def test_stream_extract_and_audit(tmp_path):
    data = b"0\tfirst passage\n1\tsecond passage\n"
    archive, target = tmp_path / "corpus.tar.gz", tmp_path / "collection.tsv"
    make_archive(archive, "collection.tsv", data)
    assert extract_collection(archive, target) == len(data)
    audit = audit_collection(target, expected_passages=2)
    assert audit["sha256"] == hashlib.sha256(data).hexdigest()
    assert audit["passages"] == 2


def test_archive_paths_cannot_escape_or_publish(tmp_path):
    archive, target = tmp_path / "corpus.tar.gz", tmp_path / "collection.tsv"
    make_archive(archive, "../escape.tsv", b"0\ttext\n")
    with pytest.raises(ValueError, match="Unexpected archive member"):
        extract_collection(archive, target)
    assert not target.exists()
    assert not (tmp_path.parent / "escape.tsv").exists()


@pytest.mark.parametrize("data,count", [(b"0\ta\n0\tb\n", 2), (b"0\ta\n", 2), (b"0\t\n", 1)])
def test_audit_rejects_duplicate_ids_bad_counts_and_empty_text(tmp_path, data, count):
    target = tmp_path / "collection.tsv"
    target.write_bytes(data)
    with pytest.raises(ValueError):
        audit_collection(target, expected_passages=count)
