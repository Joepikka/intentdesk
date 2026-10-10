"""Local review records. Human feedback is never automatically training data."""

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class ReviewConflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Incompatible review database version.")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, text TEXT NOT NULL,
                    analysis TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    version INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id),
                    created_at TEXT NOT NULL, label TEXT NOT NULL, note TEXT NOT NULL,
                    version INTEGER NOT NULL, UNIQUE(request_id, version)
                );
                CREATE INDEX IF NOT EXISTS request_queue ON requests(status, created_at, id);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        result = dict(row)
        result["analysis"] = json.loads(result["analysis"])
        return result

    def create(self, text: str, analysis: dict) -> dict:
        request_id = str(uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO requests(id,created_at,text,analysis) VALUES(?,?,?,?)",
                       (request_id, datetime.now(timezone.utc).isoformat(), text,
                        json.dumps(analysis, allow_nan=False)))
        return self.get(request_id)

    def get(self, request_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
            if row is None:
                raise KeyError(request_id)
            result = self.decode(row)
            result["decisions"] = [dict(r) for r in db.execute(
                "SELECT created_at,label,note,version FROM decisions WHERE request_id=? ORDER BY version",
                (request_id,))]
        return result

    def list(self, status: str, limit: int, offset: int) -> dict:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM requests WHERE status=? ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                              (status, limit, offset)).fetchall()
            total = db.execute("SELECT count(*) FROM requests WHERE status=?", (status,)).fetchone()[0]
        return {"items": [self.decode(r) for r in rows], "total": total, "limit": limit, "offset": offset}

    def review(self, request_id: str, label: str, note: str, expected_version: int) -> dict:
        # The update and history insert commit together. Stale tabs cannot overwrite a newer decision.
        with self.connect() as db:
            cursor = db.execute("UPDATE requests SET status='reviewed',version=version+1 WHERE id=? AND version=?",
                                (request_id, expected_version))
            if cursor.rowcount != 1:
                exists = db.execute("SELECT 1 FROM requests WHERE id=?", (request_id,)).fetchone()
                if not exists:
                    raise KeyError(request_id)
                raise ReviewConflict("This request changed. Reload it before saving a decision.")
            db.execute("INSERT INTO decisions(request_id,created_at,label,note,version) VALUES(?,?,?,?,?)",
                       (request_id, datetime.now(timezone.utc).isoformat(), label, note, expected_version+1))
        return self.get(request_id)

    def counts(self) -> dict:
        with self.connect() as db:
            values = dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status"))
        return {name: values.get(name, 0) for name in ("pending", "reviewed")}
