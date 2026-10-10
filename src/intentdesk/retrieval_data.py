"""Stream the official MS MARCO passage corpus to disk for research retrieval.

This corpus has no CLINC intent labels. It never enters classifier training.
"""

import base64
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tarfile
import time
from urllib.request import Request, urlopen

from intentdesk.common import read_json, sha256_file, write_json

URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
SOURCE_REVISION = "21b4ece9f4fc10fa4cec2028bb74287629945df6"
SOURCE_URL = f"https://raw.githubusercontent.com/microsoft/msmarco/{SOURCE_REVISION}/Datasets.md"
TERMS_URL = "https://microsoft.github.io/msmarco/"
ARCHIVE_BYTES = 1_035_009_698
ARCHIVE_MD5_BASE64 = "h90Bgm2j4q1FRHulr1d2KA=="
ETAG = '"0x8DBE166D3DC101A"'
PASSAGES = 8_841_823


def md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def download_archive(directory: Path) -> Path:
    """Resume only against the pinned object; publish only a complete verified file."""
    archive = directory / "collection.tar.gz"
    if archive.exists():
        if archive.stat().st_size != ARCHIVE_BYTES or md5_base64(archive) != ARCHIVE_MD5_BASE64:
            raise ValueError("Existing archive failed verification; preserve/remove it before retrying.")
        return archive
    partial = archive.with_suffix(".gz.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > ARCHIVE_BYTES:
        raise ValueError("Partial download exceeds expected archive size.")
    if offset < ARCHIVE_BYTES:
        headers = {"User-Agent": "IntentDesk-research/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers.update({"Range": f"bytes={offset}-", "If-Range": ETAG})
        with urlopen(Request(URL, headers=headers), timeout=60) as response:
            if response.headers.get("ETag") != ETAG:
                raise ValueError("Upstream archive changed; review source metadata before downloading.")
            if offset and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{ARCHIVE_BYTES-1}/{ARCHIVE_BYTES}"):
                raise ValueError("Server did not honor the verified resume range.")
            if not offset and response.status != 200:
                raise ValueError("Unexpected archive response.")
            if int(response.headers.get("Content-Length", -1)) != ARCHIVE_BYTES-offset:
                raise ValueError("Archive response length changed.")
            last_report = time.monotonic()
            with partial.open("ab" if offset else "wb") as output:
                while block := response.read(4 * 1024 * 1024):
                    output.write(block)
                    offset += len(block)
                    if offset > ARCHIVE_BYTES:
                        raise ValueError("Download exceeded expected size.")
                    if time.monotonic()-last_report >= 10:
                        print(f"MS MARCO archive: {offset/1e6:.0f}/{ARCHIVE_BYTES/1e6:.0f} MB", flush=True)
                        last_report = time.monotonic()
    if partial.stat().st_size != ARCHIVE_BYTES or md5_base64(partial) != ARCHIVE_MD5_BASE64:
        raise ValueError("Downloaded archive failed size/MD5 verification; partial file retained.")
    partial.replace(archive)
    return archive


def extract_collection(archive: Path, target: Path) -> int:
    """Extract one regular file by streaming; never trust archive paths."""
    partial = target.with_suffix(".tsv.part")
    found = False
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            if member.name not in ("collection.tsv", "./collection.tsv") or not member.isfile() or found:
                raise ValueError(f"Unexpected archive member: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Collection member unreadable.")
            with source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            if partial.stat().st_size != member.size:
                raise ValueError("Extracted size mismatch.")
            found = True
    if not found:
        raise ValueError("Collection missing from archive.")
    # Publish only after archive traversal and integrity checks complete.
    partial.replace(target)
    return target.stat().st_size


def audit_collection(path: Path, expected_passages: int = PASSAGES) -> dict:
    count = 0
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            identifier, delimiter, text = line.partition(b"\t")
            if not delimiter or not identifier.isdigit() or not text.strip():
                raise ValueError(f"Malformed passage row {count+1}.")
            # The official v1 collection uses contiguous ascending passage IDs.
            if int(identifier) != count:
                raise ValueError(f"Unexpected passage ID at row {count+1}.")
            text.decode("utf-8")
            count += 1
            if count % 1_000_000 == 0:
                print(f"Audited {count:,} passages", flush=True)
    if count != expected_passages:
        raise ValueError(f"Expected {expected_passages:,} passages, found {count:,}.")
    return {"passages": count, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "ids": "unique contiguous ascending integers, starting at zero", "encoding": "UTF-8"}


def prepare_retrieval(root: Path) -> dict:
    directory = root / "data" / "retrieval" / "msmarco-v1"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    collection = directory / "collection.tsv"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing["source_revision"] != SOURCE_REVISION or not collection.exists() or sha256_file(collection) != existing["collection"]["sha256"]:
            raise ValueError("Existing retrieval corpus changed.")
        return existing
    if shutil.disk_usage(directory).free < 10 * 1024**3:
        raise ValueError("Allow at least 10 GiB free for the archive, extraction and audit.")
    print("MS MARCO: non-commercial research terms; separate retrieval benchmark, not intent labels.", flush=True)
    with urlopen(SOURCE_URL, timeout=60) as response:
        source = response.read()
    if URL.encode() not in source:
        raise ValueError("Download is not listed in the pinned official dataset registry.")
    (directory / "upstream_Datasets.md").write_bytes(source)
    archive = download_archive(directory)
    print("Extracting the passage corpus in bounded memory...", flush=True)
    extract_collection(archive, collection)
    print("Auditing all passage IDs, UTF-8 rows and checksums...", flush=True)
    result = {"status": "downloaded_and_audited", "source_url": SOURCE_URL,
              "source_revision": SOURCE_REVISION, "download_url": URL, "terms_url": TERMS_URL,
              "terms": "Non-commercial research only; underlying content rights are not extended. Do not redistribute corpus in the portfolio repository.",
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "archive": {"bytes": archive.stat().st_size, "md5_base64": md5_base64(archive),
                          "sha256": sha256_file(archive), "etag": ETAG},
              "collection": audit_collection(collection),
              "used_for_classifier_training": False, "retrieval_index_built": False,
              "retrieval_quality_evaluated": False,
              "pretraining_overlap": "MPNet model card lists MS MARCO training pairs; retrieval scores using it must disclose exposure."}
    write_json(manifest_path, result)
    write_json(root / "reports" / "retrieval_data.json", result)
    return result
