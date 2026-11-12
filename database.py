"""SQLite persistence layer."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class Database:
    def __init__(self, db_path: str | Path = "data/messages.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_schema(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    issue_number INTEGER,
                    title TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    username TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT,
                    updated_at TEXT,
                    html_url TEXT NOT NULL DEFAULT '',
                    labels_json TEXT NOT NULL DEFAULT '[]',
                    sentiment TEXT,
                    sentiment_score REAL,
                    processed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_repo ON messages(repo);
                CREATE INDEX IF NOT EXISTS idx_messages_sentiment ON messages(sentiment);

                CREATE TABLE IF NOT EXISTS repo_cursors (
                    repo TEXT PRIMARY KEY,
                    last_sync_at TEXT NOT NULL,
                    pending_since TEXT,
                    next_page INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS comment_cursors (
                    repo TEXT NOT NULL,
                    issue_number INTEGER NOT NULL,
                    last_comment_updated_at TEXT NOT NULL,
                    PRIMARY KEY (repo, issue_number)
                );
                """
            )
            # Lightweight migration for databases created by an earlier prototype version.
            cursor_columns = {row[1] for row in con.execute("PRAGMA table_info(repo_cursors)").fetchall()}
            if "pending_since" not in cursor_columns:
                con.execute("ALTER TABLE repo_cursors ADD COLUMN pending_since TEXT")
            if "next_page" not in cursor_columns:
                con.execute("ALTER TABLE repo_cursors ADD COLUMN next_page INTEGER NOT NULL DEFAULT 1")

    @staticmethod
    def _normalize_datetime(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return str(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()

    def upsert_raw_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert new messages and return those whose text changed and need analysis."""
        changed: list[dict[str, Any]] = []
        with self.connect() as con:
            for item in messages:
                message_id = str(item["message_id"])
                text_value = str(item.get("text") or "").strip()
                if not text_value:
                    continue
                existing = con.execute(
                    "SELECT text FROM messages WHERE message_id = ?", (message_id,)
                ).fetchone()
                labels_json = json.dumps(item.get("labels", []), ensure_ascii=False)
                values = (
                    str(item.get("repo", "")),
                    str(item.get("kind", "issue")),
                    item.get("issue_number"),
                    str(item.get("title", "")),
                    text_value,
                    str(item.get("username", "unknown")),
                    self._normalize_datetime(item.get("created_at")),
                    self._normalize_datetime(item.get("updated_at")),
                    str(item.get("html_url", "")),
                    labels_json,
                )

                if existing is None:
                    con.execute(
                        """INSERT INTO messages (
                            message_id, repo, kind, issue_number, title, text, username,
                            created_at, updated_at, html_url, labels_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (message_id, *values),
                    )
                    changed.append({**item, "text": text_value})
                else:
                    text_changed = existing["text"] != text_value
                    con.execute(
                        """UPDATE messages SET
                            repo=?, kind=?, issue_number=?, title=?, text=?, username=?,
                            created_at=?, updated_at=?, html_url=?, labels_json=?,
                            sentiment=CASE WHEN ? THEN NULL ELSE sentiment END,
                            sentiment_score=CASE WHEN ? THEN NULL ELSE sentiment_score END,
                            processed_at=CASE WHEN ? THEN NULL ELSE processed_at END
                        WHERE message_id=?""",
                        (*values, int(text_changed), int(text_changed), int(text_changed), message_id),
                    )
                    if text_changed:
                        changed.append({**item, "text": text_value})
        return changed

    def update_sentiments(self, results: list[dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as con:
            con.executemany(
                """UPDATE messages
                   SET sentiment=?, sentiment_score=?, processed_at=?
                   WHERE message_id=?""",
                [
                    (r["sentiment"], float(r["score"]), now, r["message_id"])
                    for r in results
                ],
            )

    def get_repo_cursor(self, repo: str) -> datetime | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT last_sync_at FROM repo_cursors WHERE repo=?", (repo,)
            ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["last_sync_at"])

    def get_repo_state(self, repo: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT last_sync_at, pending_since, next_page FROM repo_cursors WHERE repo=?",
                (repo,),
            ).fetchone()
        if not row:
            return None
        return {
            "last_sync_at": datetime.fromisoformat(row["last_sync_at"]),
            "pending_since": datetime.fromisoformat(row["pending_since"]) if row["pending_since"] else None,
            "next_page": int(row["next_page"] or 1),
        }

    def set_repo_cursor(
        self,
        repo: str,
        value: datetime,
        pending_since: datetime | None = None,
        next_page: int = 1,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO repo_cursors(repo, last_sync_at, pending_since, next_page)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(repo) DO UPDATE SET
                       last_sync_at=excluded.last_sync_at,
                       pending_since=excluded.pending_since,
                       next_page=excluded.next_page""",
                (
                    repo,
                    value.astimezone(timezone.utc).isoformat(),
                    pending_since.astimezone(timezone.utc).isoformat() if pending_since else None,
                    int(next_page),
                ),
            )

    def get_comment_cursor(self, repo: str, issue_number: int) -> datetime | None:
        with self.connect() as con:
            row = con.execute(
                """SELECT last_comment_updated_at FROM comment_cursors
                   WHERE repo=? AND issue_number=?""",
                (repo, issue_number),
            ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["last_comment_updated_at"])

    def set_comment_cursor(self, repo: str, issue_number: int, value: datetime) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO comment_cursors(repo, issue_number, last_comment_updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(repo, issue_number)
                   DO UPDATE SET last_comment_updated_at=excluded.last_comment_updated_at""",
                (repo, issue_number, value.astimezone(timezone.utc).isoformat()),
            )

    def load_messages(self) -> pd.DataFrame:
        with self.connect() as con:
            frame = pd.read_sql_query("SELECT * FROM messages ORDER BY created_at DESC", con)
        if not frame.empty:
            frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
            frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
            frame["labels"] = frame["labels_json"].apply(
                lambda value: json.loads(value) if isinstance(value, str) else []
            )
        return frame

    def clear_all(self) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM messages")
            con.execute("DELETE FROM repo_cursors")
            con.execute("DELETE FROM comment_cursors")

    def insert_demo_messages(self, messages: list[dict[str, Any]]) -> int:
        """Insert reproducible demo rows with precomputed labels; no model download needed."""
        inserted = 0
        with self.connect() as con:
            for item in messages:
                cursor = con.execute(
                    """INSERT OR IGNORE INTO messages (
                        message_id, repo, kind, issue_number, title, text, username,
                        created_at, updated_at, html_url, labels_json,
                        sentiment, sentiment_score, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item["message_id"], item["repo"], item["kind"], item["issue_number"],
                        item["title"], item["text"], item["username"],
                        self._normalize_datetime(item["created_at"]),
                        self._normalize_datetime(item["updated_at"]), item["html_url"],
                        json.dumps(item.get("labels", []), ensure_ascii=False),
                        item["sentiment"], float(item["score"]),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                inserted += cursor.rowcount
        return inserted
