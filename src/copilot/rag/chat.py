"""Conversational QA as a LangGraph graph with per-thread memory.

    START -> rewrite (follow-up -> standalone question) -> answer (retrieve, rerank, cite) -> END

The checkpointer persists each thread's history: in-process memory for dev, Postgres in prod.
"""

import operator
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from copilot.config import get_settings
from copilot.rag.qa import Answer, CitedQA, get_qa


class ChatState(TypedDict, total=False):
    history: Annotated[list[dict], operator.add]  # {"role", "content"} turns, appended
    question: str
    standalone_question: str
    answer: dict  # Answer.model_dump(); plain JSON keeps checkpoints serializer-agnostic


class ChatService:
    def __init__(self, qa: CitedQA, checkpointer: BaseCheckpointSaver):
        self.qa = qa
        graph = StateGraph(ChatState)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile(checkpointer=checkpointer)

    def _rewrite(self, state: ChatState) -> ChatState:
        return {"standalone_question": self.qa.rewrite(state.get("history", []), state["question"])}

    def _answer(self, state: ChatState) -> ChatState:
        result = self.qa.answer(state["question"], state["standalone_question"])
        return {
            "answer": result.model_dump(),
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": result.answer},
            ],
        }

    def ask(self, question: str, thread_id: str) -> Answer:
        state = self.graph.invoke(
            {"question": question}, config={"configurable": {"thread_id": thread_id}}
        )
        return Answer.model_validate(state["answer"])

    def history(self, thread_id: str) -> list[dict]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("history", [])


def get_checkpointer() -> BaseCheckpointSaver:
    settings = get_settings()
    if not settings.postgres_url:
        return InMemorySaver()

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        settings.postgres_url,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()  # idempotent table migrations
    return saver


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(get_qa(), get_checkpointer())
