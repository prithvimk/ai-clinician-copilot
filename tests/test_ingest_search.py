import pytest
from qdrant_client import QdrantClient

from copilot.ingestion.indexer import ingest_path
from copilot.rag.embeddings import get_embedder
from copilot.rag.vector_store import VectorStore


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"), "test", get_embedder())


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_ingest_then_search(store, sample_pdf, sample_docx, mode):
    results = ingest_path(sample_pdf.parent, store)
    assert {r.status for r in results} == {"indexed"}

    hits = store.search("When should metformin be stopped?", k=3, mode=mode)
    assert hits[0].chunk.doc_id == "pathway"
    assert "Stop" in hits[0].chunk.text


def test_reingest_is_idempotent(store, sample_pdf):
    first = ingest_path(sample_pdf, store)
    assert first[0].status == "indexed"
    assert ingest_path(sample_pdf, store)[0].status == "unchanged"

    forced = ingest_path(sample_pdf, store, force=True)
    assert store.stats() == {"guideline": forced[0].chunks}


def test_bad_file_does_not_abort_batch(store, sample_docx, tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    results = {r.source_file: r.status for r in ingest_path(tmp_path, store)}
    assert results == {"broken.pdf": "failed", "pathway.docx": "indexed"}
