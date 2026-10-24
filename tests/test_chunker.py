from copilot.ingestion.chunker import chunk_document, count_tokens
from copilot.ingestion.models import Block, ParsedDocument

SENTENCE = "Metformin should be stopped when eGFR falls below 30 ml/min. "


def make_doc(blocks: list[Block]) -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc", title="Guideline", source_file="g.pdf", content_hash="h", blocks=blocks
    )


def test_chunks_respect_max_tokens_and_never_span_sections():
    blocks = [Block(text=SENTENCE * 10, page=p, section_path=("A",)) for p in (1, 2, 3)]
    blocks += [Block(text=SENTENCE * 10, page=4, section_path=("B",))]
    chunks = chunk_document(make_doc(blocks), max_tokens=200, overlap_tokens=30)

    assert all(c.token_count <= 200 + 30 for c in chunks)
    assert {c.section_path for c in chunks} == {("A",), ("B",)}
    b_chunks = [c for c in chunks if c.section_path == ("B",)]
    assert all(c.page_start == 4 for c in b_chunks)


def test_overlap_carries_previous_sentences():
    blocks = [Block(text=f"Sentence number {i}. " * 1 + SENTENCE * 3) for i in range(12)]
    chunks = chunk_document(make_doc(blocks), max_tokens=120, overlap_tokens=40, min_tokens=0)

    assert len(chunks) > 1
    tail_of_first = chunks[0].text.split("\n\n")[-1].split(". ")[-2]
    assert tail_of_first in chunks[1].text


def test_small_sections_fold_into_next():
    blocks = [
        Block(text="Short intro.", section_path=("Intro",)),
        Block(text=SENTENCE * 5, section_path=("Body",)),
    ]
    chunks = chunk_document(make_doc(blocks), max_tokens=600, min_tokens=60)

    assert len(chunks) == 1
    assert chunks[0].section_path == ("Body",)
    assert chunks[0].text.startswith("Short intro.")


def test_oversized_paragraph_is_split():
    chunks = chunk_document(make_doc([Block(text=SENTENCE * 100)]), max_tokens=100)
    assert len(chunks) > 5
    assert all(count_tokens(c.text) <= 140 for c in chunks)


def test_chunk_ids_are_deterministic():
    doc = make_doc([Block(text=SENTENCE * 30)])
    first = [c.chunk_id for c in chunk_document(doc, max_tokens=100)]
    assert first == [c.chunk_id for c in chunk_document(doc, max_tokens=100)]
    assert len(set(first)) == len(first)


def test_citation_format():
    chunk = chunk_document(
        make_doc([Block(text=SENTENCE * 5, page=7, section_path=("3 Targets", "3.1 HbA1c"))])
    )[0]
    assert chunk.citation() == "Guideline, p.7 (3 Targets > 3.1 HbA1c)"
    assert chunk.embed_text.startswith("Guideline > 3 Targets > 3.1 HbA1c\n\n")
