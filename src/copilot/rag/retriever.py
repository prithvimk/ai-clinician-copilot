"""Two-stage retrieval: hybrid (dense + BM25, RRF) candidates, then cross-encoder rerank.

RRF scores only rank; the cross-encoder gives a calibrated-ish 0-1 relevance per chunk, which
is what the low-evidence refusal threshold is applied to.
"""

import math
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from copilot.config import get_settings
from copilot.rag.vector_store import SearchHit, VectorStore, get_vector_store


class Reranker:
    def __init__(self, model: str):
        self.model = TextCrossEncoder(model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance in [0, 1] (sigmoid of the cross-encoder logit), one per text."""
        return [1 / (1 + math.exp(-logit)) for logit in self.model.rerank(query, texts)]


class Retriever:
    def __init__(self, store: VectorStore, reranker: Reranker, candidates: int, top_k: int):
        self.store = store
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k

    def retrieve(
        self, query: str, k: int | None = None, query_filter: models.Filter | None = None
    ) -> list[SearchHit]:
        """Top-k chunks by rerank score (highest first); `score` is the rerank relevance."""
        hits = self.store.search(query, k=self.candidates, mode="hybrid", query_filter=query_filter)
        if not hits:
            return []
        scores = self.reranker.score(query, [h.chunk.embed_text for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [SearchHit(chunk=h.chunk, score=s) for h, s in ranked[: k or self.top_k]]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings().rerank_model)


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings()
    return Retriever(
        get_vector_store(),
        get_reranker(),
        candidates=settings.retrieval_candidates,
        top_k=settings.retrieval_top_k,
    )
