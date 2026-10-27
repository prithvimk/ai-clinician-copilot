"""Dense + sparse (BM25) embedders, run locally via fastembed (ONNX, CPU).

Local models keep PHI-adjacent text on-prem and need no API key; swapping to a hosted
embedder only requires another class with the same two methods.
"""

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from copilot.config import get_settings


class Embedder:
    def __init__(self, dense_model: str, sparse_model: str, cache_dir: str | None = None):
        self.dense = TextEmbedding(dense_model, cache_dir=cache_dir)
        self.sparse = SparseTextEmbedding(sparse_model, cache_dir=cache_dir)
        self.dense_dim = len(next(iter(self.dense.embed(["dimension probe"]))))

    def embed_documents(
        self, texts: list[str], batch_size: int = 32
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        dense = [v.tolist() for v in self.dense.embed(texts, batch_size=batch_size)]
        sparse = [
            models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            for s in self.sparse.embed(texts, batch_size=batch_size)
        ]
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        dense = next(iter(self.dense.query_embed(text))).tolist()
        s = next(iter(self.sparse.query_embed(text)))
        return dense, models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    cache = str(settings.model_cache_dir) if settings.model_cache_dir else None
    return Embedder(settings.dense_model, settings.sparse_model, cache)
