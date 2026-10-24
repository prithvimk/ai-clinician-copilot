"""Parse PDF and DOCX clinical documents into section-tagged blocks.

PDFs carry no explicit structure, so headings are inferred from font size and weight
relative to the dominant body font. DOCX headings come from paragraph styles.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import docx
import pymupdf
from docx.table import Table
from docx.text.paragraph import Paragraph

from copilot.ingestion.models import Block, ParsedDocument

SUPPORTED_SUFFIXES = {".pdf", ".docx"}

_MAX_HEADING_CHARS = 120
_HEADING_SIZE_RATIO = 1.12
_MAX_HEADING_LEVELS = 3
_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+[A-Za-z]")
# Word-internal bookmarks (_GoBack, _Toc123), link anchors (flink1), figure/table captions.
_JUNK_BOOKMARK = re.compile(r"^(_\w+|[a-z]+\d+|(Fig\.?|Figure|Table|Box)\s*\d.*)$")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_document(path: Path, metadata: dict | None = None) -> ParsedDocument:
    metadata = dict(metadata or {})
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        blocks, detected_title = _load_pdf(path)
    elif suffix == ".docx":
        blocks, detected_title = _load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return ParsedDocument(
        doc_id=metadata.pop("doc_id", None) or slugify(path.stem),
        title=metadata.pop("title", None) or detected_title or path.stem,
        source_file=path.name,
        content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        blocks=blocks,
        metadata=metadata,
    )


# --- PDF -------------------------------------------------------------------------------


@dataclass
class _Marker:
    """A section start: from the PDF outline, or inferred from heading typography."""

    level: int
    title: str
    page: int
    y: float


def _load_pdf(path: Path) -> tuple[list[Block], str | None]:
    with pymupdf.open(path) as pdf:
        lines = [line for page in pdf for line in _pdf_lines(page)]
        title = (pdf.metadata or {}).get("title") or None
        markers = _outline_markers(pdf)

    heading_lines: set[int] = set()
    if not markers:
        markers, heading_lines = _typography_markers(lines)
    markers.sort(key=lambda m: (m.page, m.y))

    blocks: list[Block] = []
    path_stack: list[_Marker] = []
    para: list[str] = []
    para_page: int | None = None

    def flush() -> None:
        nonlocal para, para_page
        text = _clean(" ".join(para))
        if text:
            blocks.append(
                Block(text=text, page=para_page, section_path=tuple(m.title for m in path_stack))
            )
        para, para_page = [], None

    next_marker = 0
    prev_y1: float | None = None
    prev_page: int | None = None
    for i, ln in enumerate(lines):
        # Enter every section that starts at or before this line.
        while next_marker < len(markers) and (
            markers[next_marker].page,
            markers[next_marker].y,
        ) <= (
            ln["page"],
            ln["y0"] + 1,
        ):
            flush()
            marker = markers[next_marker]
            while path_stack and path_stack[-1].level >= marker.level:
                path_stack.pop()
            path_stack.append(marker)
            next_marker += 1
        if i in heading_lines:
            prev_y1, prev_page = ln["y1"], ln["page"]
            continue

        # New paragraph on page change or a vertical gap larger than ~one line.
        gap = ln["y0"] - prev_y1 if prev_y1 is not None and ln["page"] == prev_page else None
        if para and (ln["page"] != prev_page or (gap is not None and gap > ln["size"] * 0.9)):
            flush()
        if not para:
            para_page = ln["page"]
        para.append(ln["text"])
        prev_y1, prev_page = ln["y1"], ln["page"]
    flush()

    return blocks, title


def _outline_markers(pdf: pymupdf.Document) -> list[_Marker]:
    """Section starts from the PDF's bookmarks, positioned by finding the title on its page."""
    entries = [
        (level, _clean(title), page)
        for level, title, page in pdf.get_toc(simple=True)
        if page >= 1 and re.search(r"[A-Za-z]{3}", title) and not _JUNK_BOOKMARK.match(title)
    ]
    # A single top-level entry is the document title itself: drop it and promote its children.
    if sum(1 for level, _, _ in entries if level == 1) == 1 and len(entries) > 1:
        entries = [(level - 1, t, p) for level, t, p in entries if level > 1]
    # Outline levels are often wrong in exported PDFs; numbered headings ("2.3 ...") are not.
    entries = [
        (len(m.group(1).split(".")) if (m := _SECTION_NUMBER.match(t)) else level, t, p)
        for level, t, p in entries
    ]

    markers = []
    for level, title, page in entries:
        hits = pdf[page - 1].search_for(title[:60]) if title else []
        markers.append(_Marker(level, title, page, hits[0].y0 if hits else 0.0))
    return markers


def _typography_markers(lines: list[dict]) -> tuple[list[_Marker], set[int]]:
    """Infer headings from font size when a PDF has no outline.

    Consecutive heading-sized lines are merged (multi-line headings), the cover page is
    ignored, and only sizes recurring across pages count as heading levels.
    """
    body_size = _dominant_font_size(lines)
    candidates = [
        i for i, ln in enumerate(lines) if ln["page"] > 1 and _is_heading_candidate(ln, body_size)
    ]
    pages_by_size: dict[float, set[int]] = {}
    for i in candidates:
        pages_by_size.setdefault(round(lines[i]["size"], 1), set()).add(lines[i]["page"])
    sizes = sorted((s for s, pages in pages_by_size.items() if len(pages) >= 2), reverse=True)
    levels = {s: n + 1 for n, s in enumerate(sizes[:_MAX_HEADING_LEVELS])}

    markers: list[_Marker] = []
    heading_lines: set[int] = set()
    prev: int | None = None
    for i in candidates:
        ln = lines[i]
        size = round(ln["size"], 1)
        if size not in levels:
            continue
        heading_lines.add(i)
        p = lines[prev] if prev is not None else None
        if (
            p is not None
            and prev == i - 1
            and round(p["size"], 1) == size
            and p["page"] == ln["page"]
            and ln["y0"] - p["y1"] < ln["size"] * 0.8
        ):
            markers[-1].title = f"{markers[-1].title} {ln['text']}"
        else:
            markers.append(_Marker(levels[size], ln["text"], ln["page"], ln["y0"]))
        prev = i
    return markers, heading_lines


def _pdf_lines(page: pymupdf.Page) -> list[dict]:
    """Flatten a page into text lines with their dominant font size and boldness."""
    out = []
    height = page.rect.height
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            # Keep whitespace-only spans: some PDFs encode inter-word spaces as their own spans.
            text = _clean("".join(s["text"] for s in line["spans"]))
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            y0, y1 = line["bbox"][1], line["bbox"][3]
            # Drop running headers/footers and bare page numbers.
            if (y1 < height * 0.05 or y0 > height * 0.95) or re.fullmatch(r"\d{1,4}", text):
                continue
            main = max(spans, key=lambda s: len(s["text"]))
            out.append(
                {
                    "text": text,
                    "size": main["size"],
                    "bold": bool(main["flags"] & 16) or "bold" in main["font"].lower(),
                    "page": page.number + 1,
                    "y0": y0,
                    "y1": y1,
                }
            )
    return out


def _dominant_font_size(lines: list[dict]) -> float:
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[round(ln["size"], 1)] += len(ln["text"])
    return weights.most_common(1)[0][0] if weights else 10.0


def _is_heading_candidate(ln: dict, body_size: float) -> bool:
    text = ln["text"]
    if len(text) > _MAX_HEADING_CHARS or len(text) < 3 or not re.search(r"[A-Za-z]{3}", text):
        return False
    if text.endswith((".", ",", ";")) and not re.match(r"^\d+(\.\d+)*\.?\s", text):
        return False
    return ln["size"] >= body_size * _HEADING_SIZE_RATIO


# --- DOCX ------------------------------------------------------------------------------


def _load_docx(path: Path) -> tuple[list[Block], str | None]:
    document = docx.Document(str(path))
    title = document.core_properties.title or None
    blocks: list[Block] = []
    path_stack: list[tuple[int, str]] = []

    for item in _iter_docx_items(document):
        if isinstance(item, Table):
            rows = [" | ".join(_clean(c.text) for c in row.cells) for row in item.rows]
            text = "\n".join(r for r in rows if r.strip(" |"))
            if text:
                blocks.append(Block(text=text, section_path=tuple(h for _, h in path_stack)))
            continue

        text = _clean(item.text)
        if not text:
            continue
        style = (item.style.name or "") if item.style is not None else ""
        level = _docx_heading_level(style)
        if level == 0:
            title = title or text
            continue
        if level is not None:
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, text))
            continue
        blocks.append(Block(text=text, section_path=tuple(h for _, h in path_stack)))

    return blocks, title


def _docx_heading_level(style: str) -> int | None:
    if style == "Title":
        return 0
    match = re.match(r"Heading (\d)", style)
    return int(match.group(1)) if match else None


def _iter_docx_items(document):
    """Yield paragraphs and tables in document order."""
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


def _clean(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()
