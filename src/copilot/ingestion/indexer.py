"""Ingest documents into the vector store: load → chunk → embed → upsert.

Idempotent: a document whose content hash is unchanged is skipped; a changed document has
its old chunks deleted before re-indexing, so stale guidance never lingers in the index.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from copilot.config import get_settings
from copilot.ingestion.chunker import chunk_document
from copilot.ingestion.loaders import SUPPORTED_SUFFIXES, load_document
from copilot.rag.vector_store import VectorStore

log = logging.getLogger(__name__)

SOURCES_FILE = "sources.yaml"


@dataclass
class IngestResult:
    source_file: str
    doc_id: str
    status: str  # indexed | unchanged | failed
    chunks: int = 0
    error: str = ""


def discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)


def load_source_metadata(directory: Path) -> dict[str, dict]:
    """Per-file metadata (title, publisher, year, url, conditions) keyed by filename."""
    sources = directory / SOURCES_FILE
    if not sources.exists():
        return {}
    entries = yaml.safe_load(sources.read_text(encoding="utf-8")) or []
    return {e["file"]: {k: v for k, v in e.items() if k != "file"} for e in entries}


def ingest_path(path: Path, store: VectorStore, force: bool = False) -> list[IngestResult]:
    settings = get_settings()
    directory = path if path.is_dir() else path.parent
    source_meta = load_source_metadata(directory)
    results = []

    for file in discover(path):
        try:
            doc = load_document(file, source_meta.get(file.name))
            if not force and store.document_hash(doc.doc_id) == doc.content_hash:
                results.append(IngestResult(file.name, doc.doc_id, "unchanged"))
                continue
            chunks = chunk_document(
                doc,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            store.delete_document(doc.doc_id)
            store.upsert_chunks(chunks, doc.content_hash)
            results.append(IngestResult(file.name, doc.doc_id, "indexed", len(chunks)))
        except Exception as exc:  # one bad file must not abort the batch
            log.exception("Failed to ingest %s", file)
            results.append(IngestResult(file.name, "", "failed", error=str(exc)))
    return results
