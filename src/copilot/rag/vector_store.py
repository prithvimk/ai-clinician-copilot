"""Qdrant-backed chunk store with dense, sparse (BM25) and hybrid (RRF) search.

Runs embedded (on-disk, no server) when QDRANT_URL is unset; the same code talks to a
Qdrant server/cluster in production.
"""

import atexit
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from qdrant_client import QdrantClient, models
from qdrant_client.local.qdrant_local import QdrantLocal

from copilot.config import get_settings
from copilot.ingestion.models import Chunk
from copilot.rag.embeddings import Embedder, get_embedder

DENSE = "dense"
SPARSE = "bm25"
SearchMode = Literal["dense", "sparse", "hybrid"]


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.embedder.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if not isinstance(self.client._client, QdrantLocal):  # indexes are a no-op when embedded
            for field in ("doc_id", "content_hash"):
                self.client.create_payload_index(
                    self.collection, field, models.PayloadSchemaType.KEYWORD
                )

    # --- writes --------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk], content_hash: str, batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            dense, sparse = self.embedder.embed_documents([c.embed_text for c in batch])
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=c.chunk_id,
                        vector={DENSE: d, SPARSE: s},
                        payload={**c.model_dump(), "content_hash": content_hash},
                    )
                    for c, d, s in zip(batch, dense, sparse, strict=True)
                ],
            )

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.collection, points_selector=_doc_filter(doc_id))

    # --- reads ---------------------------------------------------------------------

    def document_hash(self, doc_id: str) -> str | None:
        points, _ = self.client.scroll(
            self.collection, scroll_filter=_doc_filter(doc_id), limit=1, with_payload=True
        )
        return points[0].payload.get("content_hash") if points else None

    def search(
        self,
        query: str,
        k: int = 5,
        mode: SearchMode = "hybrid",
        query_filter: models.Filter | None = None,
        prefetch_k: int = 30,
    ) -> list[SearchHit]:
        dense, sparse = self.embedder.embed_query(query)
        if mode == "dense":
            result = self.client.query_points(
                self.collection, query=dense, using=DENSE, limit=k, query_filter=query_filter
            )
        elif mode == "sparse":
            result = self.client.query_points(
                self.collection, query=sparse, using=SPARSE, limit=k, query_filter=query_filter
            )
        else:
            result = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense, using=DENSE, limit=prefetch_k, filter=query_filter
                    ),
                    models.Prefetch(
                        query=sparse, using=SPARSE, limit=prefetch_k, filter=query_filter
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
            )
        return [
            SearchHit(chunk=Chunk.model_validate(p.payload), score=p.score) for p in result.points
        ]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=["doc_id"]
            )
            for p in points:
                counts[p.payload["doc_id"]] = counts.get(p.payload["doc_id"], 0) + 1
            if offset is None:
                return counts


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    else:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(settings.qdrant_path))
    atexit.register(client.close)  # embedded mode otherwise errors while the interpreter exits
    return VectorStore(client, settings.qdrant_collection, get_embedder())
