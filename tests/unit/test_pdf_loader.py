from reportlab.pdfgen import canvas

from knowledge_assistant.ingestion.loaders import load_document


def create_test_pdf(path):
    """Create a two-page PDF for testing."""
    pdf = canvas.Canvas(str(path))

    pdf.drawString(72, 750, "Solar Inverter Fundamentals")
    pdf.drawString(72, 720, "An inverter converts DC power into AC power.")
    pdf.showPage()

    pdf.drawString(72, 750, "Grid Support Functions")
    pdf.drawString(72, 720, "Volt-VAR controls reactive power.")
    pdf.showPage()

    pdf.save()


def test_load_pdf_extracts_text_and_metadata(tmp_path):
    pdf_path = tmp_path / "solar_guide.pdf"
    create_test_pdf(pdf_path)

    document = load_document(pdf_path)

    assert document.source == "solar_guide.pdf"
    assert document.file_type == "pdf"
    assert document.document_id
    assert "Solar Inverter Fundamentals" in document.text
    assert "Volt-VAR controls reactive power." in document.text
    assert "[Page 1]" in document.text
    assert "[Page 2]" in document.text
    assert document.metadata["page_count"] == 2
    assert document.metadata["file_size"] > 0


def test_pdf_document_id_is_content_based(tmp_path):
    pdf_path = tmp_path / "solar_guide.pdf"
    create_test_pdf(pdf_path)

    first = load_document(pdf_path)
    second = load_document(pdf_path)

    assert first.document_id == second.document_id


def test_empty_pdf_is_rejected(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    pdf.showPage()
    pdf.save()

    try:
        load_document(pdf_path)
        raise AssertionError("Expected ValueError for PDF without extractable text")
    except ValueError as error:
        assert "No extractable text" in str(error)
