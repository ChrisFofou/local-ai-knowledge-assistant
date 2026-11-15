import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from knowledge_assistant.ingestion.models import IngestedDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


def split_text(text: str, config: ChunkingConfig) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + config.chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - config.chunk_overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_document(
    document: IngestedDocument,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Convert an ingested document into metadata-preserving chunks."""
    config = config or ChunkingConfig()

    texts = split_text(document.text, config)
    chunks = []

    for index, text in enumerate(texts):
        chunk_id = hashlib.sha256(
            f"{document.document_id}:{index}:{text}".encode("utf-8")
        ).hexdigest()

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                source=document.source,
                text=text,
                chunk_index=index,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "chunk_count": len(texts),
                    "file_type": document.file_type,
                },
            )
        )

    return chunks
