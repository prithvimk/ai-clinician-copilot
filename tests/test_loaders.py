from copilot.ingestion.loaders import load_document


def test_pdf_sections_from_typography(sample_pdf):
    doc = load_document(sample_pdf)

    sections = {b.section_path for b in doc.blocks}
    assert ("1 Glycaemic control",) in sections
    assert ("2 Blood pressure",) in sections
    glycaemic = next(b for b in doc.blocks if b.section_path == ("1 Glycaemic control",))
    assert glycaemic.page == 2
    assert "HbA1c" in glycaemic.text


def test_pdf_drops_page_numbers(sample_pdf):
    doc = load_document(sample_pdf)
    assert not any(b.text.strip() in {"2", "3"} for b in doc.blocks)


def test_metadata_overrides(sample_pdf):
    doc = load_document(sample_pdf, {"doc_id": "dm-1", "title": "DM Guideline", "year": 2024})
    assert (doc.doc_id, doc.title, doc.metadata) == ("dm-1", "DM Guideline", {"year": 2024})


def test_docx_headings_tables_and_title(sample_docx):
    doc = load_document(sample_docx)

    assert doc.title == "Care Pathway"
    assert doc.doc_id == "pathway"
    monitoring = next(b for b in doc.blocks if "HbA1c" in b.text)
    assert monitoring.section_path == ("Diabetes", "Monitoring")
    table = next(b for b in doc.blocks if "Metformin" in b.text)
    assert table.section_path == ("Hypertension",)
    assert "<30 | Stop" in table.text
