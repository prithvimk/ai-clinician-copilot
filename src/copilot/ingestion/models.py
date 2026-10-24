from pydantic import BaseModel, Field


class Block(BaseModel):
    """A paragraph-level unit of text with its location in the source document."""

    text: str
    page: int | None = None  # 1-based; None for formats without pages (DOCX)
    section_path: tuple[str, ...] = ()


class ParsedDocument(BaseModel):
    doc_id: str
    title: str
    source_file: str
    content_hash: str
    blocks: list[Block]
    metadata: dict = Field(default_factory=dict)  # publisher, year, url, conditions...


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    source_file: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    text: str
    token_count: int
    metadata: dict = Field(default_factory=dict)

    @property
    def section(self) -> str:
        return " > ".join(self.section_path)

    @property
    def embed_text(self) -> str:
        """Text sent to the embedder: a contextual header improves retrieval of terse chunks."""
        header = self.title + (f" > {self.section}" if self.section_path else "")
        return f"{header}\n\n{self.text}"

    def citation(self) -> str:
        pages = ""
        if self.page_start is not None:
            pages = (
                f", p.{self.page_start}"
                if self.page_start == self.page_end
                else f", pp.{self.page_start}-{self.page_end}"
            )
        section = f" ({self.section})" if self.section_path else ""
        return f"{self.title}{pages}{section}"
