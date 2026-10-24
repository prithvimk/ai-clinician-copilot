"""Section-aware chunking.

Chunks never span two sections (so a citation's section label is always accurate), except
that very small sections are folded into the following one to avoid near-empty chunks.
Within a section, paragraphs are packed up to `max_tokens`, with `overlap_tokens` of trailing
context carried into the next chunk. Oversized paragraphs are split on sentence boundaries.
"""

import re
import uuid
from functools import lru_cache
from itertools import groupby

import tiktoken

from copilot.ingestion.models import Block, Chunk, ParsedDocument

_NAMESPACE = uuid.UUID("6f1c2b1e-6a0e-4d4b-9a57-5c0b1e7d2a10")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9•\-(])")


@lru_cache
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def chunk_id(doc_id: str, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{index}"))


def chunk_document(
    doc: ParsedDocument,
    max_tokens: int = 600,
    overlap_tokens: int = 100,
    min_tokens: int = 60,
) -> list[Chunk]:
    sections = [
        (path, list(blocks)) for path, blocks in groupby(doc.blocks, key=lambda b: b.section_path)
    ]

    pieces: list[tuple[tuple[str, ...], list[Block]]] = []  # (section_path, blocks) per chunk
    carry: list[Block] = []
    for path, blocks in sections:
        blocks = carry + blocks
        carry = []
        if sum(count_tokens(b.text) for b in blocks) < min_tokens:
            carry = blocks  # too small on its own: prepend to the next section
            continue
        pieces.extend((path, group) for group in _pack(blocks, max_tokens, overlap_tokens))
    if carry:
        if pieces:
            path, last = pieces[-1]
            pieces[-1] = (path, last + carry)
        else:
            pieces.append((carry[-1].section_path, carry))

    chunks = []
    for index, (path, blocks) in enumerate(pieces):
        text = "\n\n".join(b.text for b in blocks)
        pages = [b.page for b in blocks if b.page is not None]
        chunks.append(
            Chunk(
                chunk_id=chunk_id(doc.doc_id, index),
                doc_id=doc.doc_id,
                chunk_index=index,
                title=doc.title,
                source_file=doc.source_file,
                section_path=path,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                text=text,
                token_count=count_tokens(text),
                metadata=doc.metadata,
            )
        )
    return chunks


def _pack(blocks: list[Block], max_tokens: int, overlap_tokens: int) -> list[list[Block]]:
    units = [u for b in blocks for u in _split_oversized(b, max_tokens)]
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0

    for unit in units:
        n = count_tokens(unit.text)
        if current and current_tokens + n > max_tokens:
            groups.append(current)
            current = _overlap_tail(current, min(overlap_tokens, max_tokens // 3))
            current_tokens = sum(count_tokens(b.text) for b in current)
            if current_tokens + n > max_tokens:  # overlap must never push a chunk over the limit
                current, current_tokens = [], 0
        current.append(unit)
        current_tokens += n
    if current:
        groups.append(current)
    return groups


def _overlap_tail(blocks: list[Block], overlap_tokens: int) -> list[Block]:
    """Trailing sentences of the previous chunk, up to `overlap_tokens`."""
    if overlap_tokens <= 0:
        return []
    last = blocks[-1]
    tail: list[str] = []
    total = 0
    for sentence in reversed(_SENTENCE_SPLIT.split(last.text)):
        n = count_tokens(sentence)
        if total + n > overlap_tokens:
            break
        tail.insert(0, sentence)
        total += n
    return [last.model_copy(update={"text": " ".join(tail)})] if tail else []


def _split_oversized(block: Block, max_tokens: int) -> list[Block]:
    if count_tokens(block.text) <= max_tokens:
        return [block]
    parts: list[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(block.text):
        candidate = f"{buf} {sentence}".strip()
        if buf and count_tokens(candidate) > max_tokens:
            parts.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        parts.append(buf)
    # A single sentence longer than max_tokens: hard-split on tokens.
    out = []
    enc = _encoder()
    for part in parts:
        ids = enc.encode(part)
        for i in range(0, len(ids), max_tokens):
            out.append(block.model_copy(update={"text": enc.decode(ids[i : i + max_tokens])}))
    return out
