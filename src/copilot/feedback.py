"""Clinician feedback for the learning loop: ratings, corrections and review edits.

Stored as append-only JSONL (one record per line) so it can feed evaluation sets and prompt
or rule updates later; swap for a database table behind the same two methods in production.
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


@lru_cache
def get_feedback_store() -> FeedbackStore:
    return FeedbackStore(get_settings().feedback_path)
