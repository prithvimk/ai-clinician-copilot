"""Clinician feedback for the learning loop: ratings, corrections and review edits.

Postgres table when POSTGRES_URL is set (shared by all API replicas); otherwise append-only
JSONL for local runs. Both expose add() / list() and feed evaluation sets and prompt or rule
updates later.
"""

import datetime as dt
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from copilot.config import get_settings


class FeedbackIn(BaseModel):
    kind: Literal["chat", "analysis"]
    thread_id: str
    rating: Literal["up", "down"] | None = None
    comment: str = Field("", max_length=4000)
    correction: str = Field("", max_length=8000)  # what the answer should have said
    payload: dict = Field(default_factory=dict)  # e.g. question/answer, original vs edited draft


class Feedback(FeedbackIn):
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    source: Literal["user", "review"] = "user"  # "review": recorded from a clinician review


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def add(self, item: FeedbackIn, source: Literal["user", "review"] = "user") -> Feedback:
        record = Feedback(**item.model_dump(), source=source)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(record.model_dump_json() + "\n")
        return record

    def list(self, kind: str | None = None, limit: int = 100) -> list[Feedback]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text().splitlines()
        records = [Feedback.model_validate(json.loads(line)) for line in lines if line.strip()]
        if kind:
            records = [r for r in records if r.kind == kind]
        return records[-limit:][::-1]  # newest first


class PostgresFeedbackStore:
    DDL = """
        CREATE TABLE IF NOT EXISTS copilot_feedback (
            id text PRIMARY KEY,
            created_at timestamptz NOT NULL,
            kind text NOT NULL,
            source text NOT NULL,
            thread_id text NOT NULL,
            record jsonb NOT NULL
        );
        CREATE INDEX IF NOT EXISTS copilot_feedback_kind_created
            ON copilot_feedback (kind, created_at DESC);
    """

    def __init__(self, url: str):
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(
            url, kwargs={"autocommit": True}, min_size=1, max_size=4, open=True
        )
        with self.pool.connection() as conn:
            conn.execute(self.DDL)

    def add(self, item: FeedbackIn, source: Literal["user", "review"] = "user") -> Feedback:
        record = Feedback(**item.model_dump(), source=source)
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO copilot_feedback (id, created_at, kind, source, thread_id, record)"
                " VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                (
                    record.id,
                    record.created_at,
                    record.kind,
                    record.source,
                    record.thread_id,
                    record.model_dump_json(),
                ),
            )
        return record

    def list(self, kind: str | None = None, limit: int = 100) -> list[Feedback]:
        query = "SELECT record FROM copilot_feedback"
        params: list = []
        if kind:
            query += " WHERE kind = %s"
            params.append(kind)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self.pool.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [Feedback.model_validate(row[0]) for row in rows]


@lru_cache
def get_feedback_store() -> FeedbackStore | PostgresFeedbackStore:
    settings = get_settings()
    if settings.postgres_url:
        return PostgresFeedbackStore(settings.postgres_url)
    return FeedbackStore(settings.feedback_path)
