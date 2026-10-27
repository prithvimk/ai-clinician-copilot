"""Production persistence paths against a real Postgres (CI service container).

Skipped unless TEST_POSTGRES_URL is set, e.g. postgresql://copilot:copilot@localhost:5432/copilot
"""

import os
from uuid import uuid4

import pytest

from copilot.feedback import FeedbackIn, PostgresFeedbackStore
from copilot.rag.chat import ChatService
from copilot.rag.qa import CitedQA
from tests.fakes import FakeLLM, StubRetriever

URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_POSTGRES_URL not set")


def postgres_saver():
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        URL,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver, pool


def test_feedback_store_round_trip():
    store, again = PostgresFeedbackStore(URL), None
    thread = uuid4().hex
    store.add(FeedbackIn(kind="chat", thread_id=thread, rating="up", payload={"q": "x"}))
    store.add(FeedbackIn(kind="analysis", thread_id=thread, comment="edited"), source="review")
    try:
        again = PostgresFeedbackStore(URL)  # DDL is idempotent
        mine = [f for f in store.list(limit=1000) if f.thread_id == thread]
        assert [f.kind for f in mine] == ["analysis", "chat"]  # newest first
        assert mine[0].source == "review" and mine[1].payload == {"q": "x"}
        assert all(f.kind == "chat" for f in store.list(kind="chat"))
    finally:
        for s in (store, again):
            if s:
                s.pool.close()


def test_chat_memory_survives_a_new_service_instance():
    """History lives in Postgres, so another replica (new process) resumes the thread."""
    thread = uuid4().hex
    retriever = StubRetriever("Stop metformin when eGFR falls below 30.")
    (saver1, pool1), (saver2, pool2) = postgres_saver(), postgres_saver()
    try:
        first = ChatService(CitedQA(retriever, FakeLLM("Stop below 30 [1]."), 0.0), saver1)
        first.ask("When should metformin be stopped?", thread)

        llm = FakeLLM("metformin dose reduction", "Reduce at 30-44 [1].")
        second = ChatService(CitedQA(retriever, llm, 0.0), saver2)
        follow_up = second.ask("And the dose?", thread)
        assert follow_up.standalone_question == "metformin dose reduction"
        assert "When should metformin be stopped?" in llm.calls[0]["messages"][0]["content"]
        assert len(second.history(thread)) == 4
    finally:
        pool1.close()
        pool2.close()
