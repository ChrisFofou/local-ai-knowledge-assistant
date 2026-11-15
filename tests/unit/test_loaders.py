from pathlib import Path

import pytest

from knowledge_assistant.ingestion.loaders import (
    calculate_document_id,
    load_document,
    normalize_text,
)


def test_normalize_text():
    text = "Hello   world\r\n\r\n\r\nSecond paragraph"

    assert normalize_text(text) == "Hello world\n\nSecond paragraph"


def test_load_markdown(tmp_path: Path):
    path = tmp_path / "guide.md"
    path.write_text("# Solar Inverter\n\nConverts DC to AC.", encoding="utf-8")

    document = load_document(path)

    assert document.source == "guide.md"
    assert document.file_type == "md"
    assert "Converts DC to AC" in document.text
    assert document.metadata["file_size"] > 0


def test_document_id_is_content_based(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_text("Same content", encoding="utf-8")
    second.write_text("Same content", encoding="utf-8")

    assert calculate_document_id(first) == calculate_document_id(second)


def test_unsupported_format(tmp_path: Path):
    path = tmp_path / "guide.docx"
    path.write_text("Test", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(path)


def test_empty_document(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        load_document(path)


def test_missing_document():
    with pytest.raises(FileNotFoundError):
        load_document("/tmp/this-file-does-not-exist.txt")
