"""Scripted LLM and fixed-hit retriever for testing without API keys or embedding models."""

from copilot.ingestion.models import Chunk
from copilot.rag.vector_store import SearchHit


class FakeLLM:
    """Returns scripted replies (text or schema objects) in order and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def _next(self, system, messages, effort):
        self.calls.append({"system": system, "messages": messages, "effort": effort})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def complete(self, system, messages, effort, max_tokens=16000):
        return self._next(system, messages, effort)

    def parse(self, system, messages, schema, effort, max_tokens=16000):
        reply = self._next(system, messages, effort)
        return reply if isinstance(reply, schema) else schema.model_validate(reply)


class StubRetriever:
    """Fixed hits, so QA / memory / API logic is tested without embedding models."""

    def __init__(self, *texts: str, score: float = 0.9, miss=lambda query: False):
        self.hits = [
            SearchHit(
                chunk=Chunk(
                    chunk_id=f"c{i}",
                    doc_id="pathway",
                    chunk_index=i,
                    title="Care Pathway",
                    source_file="pathway.docx",
                    section_path=("Diabetes", "Metformin"),
                    page_start=None,
                    page_end=None,
                    text=text,
                    token_count=len(text.split()),
                ),
                score=score - i * 0.1,
            )
            for i, text in enumerate(texts)
        ]
        self.miss = miss  # queries for which nothing is found
        self.queries: list[str] = []

    def retrieve(self, query, k=None, query_filter=None):
        self.queries.append(query)
        return [] if self.miss(query) else self.hits[: k or len(self.hits)]
