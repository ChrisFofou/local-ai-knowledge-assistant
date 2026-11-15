import pytest

from knowledge_assistant.ingestion.chunker import (
    ChunkingConfig,
    chunk_document,
    split_text,
)
from knowledge_assistant.ingestion.models import IngestedDocument


def make_document(text):
    return IngestedDocument(
        document_id="test-document-123",
        source="guide.md",
        file_type="md",
        text=text,
        metadata={"category": "technical"},
    )


def test_split_text_respects_chunk_size():
    text = "A" * 250
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)

    chunks = split_text(text, config)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert len(chunks) > 1


def test_split_text_preserves_content():
    text = "Solar inverters convert DC power into AC power."
    config = ChunkingConfig(chunk_size=100, chunk_overlap=10)

    chunks = split_text(text, config)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_handles_empty_text():
    assert split_text("   ", ChunkingConfig()) == []


def test_chunk_document_preserves_metadata():
    document = make_document("Solar inverter technical documentation. " * 30)
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)

    chunks = chunk_document(document, config)

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):
        assert chunk.document_id == document.document_id
        assert chunk.source == document.source
        assert chunk.chunk_index == index
        assert chunk.metadata["category"] == "technical"
        assert chunk.metadata["file_type"] == "md"


def test_chunk_ids_are_deterministic():
    document = make_document("Solar inverter technical documentation. " * 10)

    first = chunk_document(document)
    second = chunk_document(document)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_invalid_chunking_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
