from dataclasses import dataclass
from typing import Any

import pytest

from knowledge_assistant.rag.context import (
    AssembledContext,
    CharacterTokenCounter,
    ContextAssembler,
)
from knowledge_assistant.retrieval.service import RetrievedChunk


def make_chunk(
    chunk_id: str = "chunk-1",
    text: str = "Solar inverters convert DC into AC.",
    source: str = "solar_guide.pdf",
    document_id: str = "doc-1",
    chunk_index: int = 0,
    metadata: dict[str, Any] | None = None,
    distance: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        document_id=document_id,
        chunk_index=chunk_index,
        distance=distance,
        metadata=metadata or {},
    )


def test_character_token_counter_estimates_tokens():
    counter = CharacterTokenCounter()

    assert counter.count("") == 0
    assert counter.count("abc") == 1
    assert counter.count("abcdef") == 2
    assert counter.count("abcdefghij") == 4


def test_assemble_returns_empty_context_for_empty_input():
    assembler = ContextAssembler()

    result = assembler.assemble([])

    assert isinstance(result, AssembledContext)
    assert result.entries == []
    assert result.rendered == ""
    assert result.estimated_tokens == 0
    assert result.excluded_chunk_ids == []


def test_assemble_includes_single_chunk():
    chunk = make_chunk()
    assembler = ContextAssembler()

    result = assembler.assemble([chunk])

    assert len(result.entries) == 1
    assert result.entries[0].chunk_id == "chunk-1"
    assert result.entries[0].text == chunk.text
    assert "Solar inverters convert DC into AC." in result.rendered


def test_assemble_preserves_retrieval_order():
    chunks = [
        make_chunk(chunk_id="chunk-1", text="First result", chunk_index=0),
        make_chunk(chunk_id="chunk-2", text="Second result", chunk_index=1),
        make_chunk(chunk_id="chunk-3", text="Third result", chunk_index=2),
    ]
    assembler = ContextAssembler()

    result = assembler.assemble(chunks)

    assert [entry.chunk_id for entry in result.entries] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]


def test_assemble_assigns_sequential_citation_ids():
    chunks = [
        make_chunk(chunk_id="chunk-1", text="First source"),
        make_chunk(chunk_id="chunk-2", text="Second source", chunk_index=1),
    ]
    assembler = ContextAssembler()

    result = assembler.assemble(chunks)

    assert [entry.citation_id for entry in result.entries] == ["S1", "S2"]
    assert "[S1]" in result.rendered
    assert "[S2]" in result.rendered


def test_assemble_preserves_source_and_metadata():
    chunk = make_chunk(
        source="inverter_manual.pdf",
        document_id="doc-42",
        chunk_index=7,
        metadata={"file_type": "pdf", "language": "en"},
    )
    assembler = ContextAssembler()

    result = assembler.assemble([chunk])

    entry = result.entries[0]

    assert entry.source == "inverter_manual.pdf"
    assert entry.document_id == "doc-42"
    assert entry.chunk_index == 7
    assert entry.metadata["file_type"] == "pdf"
    assert entry.metadata["language"] == "en"


def test_assemble_reports_token_usage():
    chunks = [
        make_chunk(chunk_id="chunk-1", text="Short text"),
        make_chunk(chunk_id="chunk-2", text="Another short text", chunk_index=1),
    ]
    assembler = ContextAssembler()

    result = assembler.assemble(chunks)

    assert result.estimated_tokens > 0
    assert result.estimated_tokens <= result.max_tokens


def test_assemble_respects_context_budget():
    chunks = [
        make_chunk(chunk_id="chunk-1", text="A" * 300),
        make_chunk(chunk_id="chunk-2", text="B" * 300, chunk_index=1),
        make_chunk(chunk_id="chunk-3", text="C" * 300, chunk_index=2),
    ]
    assembler = ContextAssembler(max_tokens=120)

    result = assembler.assemble(chunks)

    assert result.estimated_tokens <= 120
    assert len(result.entries) < len(chunks)


def test_assemble_skips_oversized_chunk_and_considers_later_chunks():
    chunks = [
        make_chunk(chunk_id="large", text="X" * 1000),
        make_chunk(chunk_id="small-1", text="Useful short content", chunk_index=1),
        make_chunk(chunk_id="small-2", text="More useful content", chunk_index=2),
    ]
    assembler = ContextAssembler(max_tokens=100)

    result = assembler.assemble(chunks)

    included_ids = [entry.chunk_id for entry in result.entries]

    assert "large" not in included_ids
    assert "small-1" in included_ids
    assert "small-2" in included_ids


def test_assemble_tracks_excluded_chunk_ids():
    chunks = [
        make_chunk(chunk_id="included", text="Short content"),
        make_chunk(chunk_id="excluded", text="X" * 1000, chunk_index=1),
    ]
    assembler = ContextAssembler(max_tokens=100)

    result = assembler.assemble(chunks)

    assert "excluded" in result.excluded_chunk_ids
    assert "included" not in result.excluded_chunk_ids


@dataclass
class FixedTokenCounter:
    """A test token counter with deterministic costs."""

    token_cost: int

    def count(self, text: str) -> int:
        return self.token_cost


def test_assemble_accepts_custom_token_counter():
    chunks = [
        make_chunk(chunk_id="chunk-1", text="First source"),
        make_chunk(chunk_id="chunk-2", text="Second source", chunk_index=1),
    ]
    assembler = ContextAssembler(
        max_tokens=100,
        token_counter=FixedTokenCounter(token_cost=10),
    )

    result = assembler.assemble(chunks)

    assert len(result.entries) == 2
    assert result.estimated_tokens <= 100


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_context_assembler_rejects_invalid_budget(max_tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        ContextAssembler(max_tokens=max_tokens)
